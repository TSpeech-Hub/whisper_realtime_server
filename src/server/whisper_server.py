#!/usr/bin/env python3
import logging, os, sys
from datetime import datetime
from concurrent import futures
from contextlib import asynccontextmanager

import numpy as np
from grpc import aio
import asyncio
import itertools
from src.generated import speech_pb2_grpc, speech_pb2
from src.parallel_whisper_online import *

# LOGGING SETUP FUNCTION
def setup_logging(log_name, use_stdout=False, log_folder="server_logs"):
    os.makedirs(log_folder, exist_ok=True)

    log_path = os.path.join(log_folder, f"{datetime.now():%Y%m%d_%H%M%S}_{log_name}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    handlers = [logging.FileHandler(log_path)]
    if use_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Config Constants for th server 
# Not everything is used 
WHISPER_CONFIG = SimpleNamespace(
    warmup_file="resources/sample1.wav",
    model="large-v3-turbo",
    backend="faster-whisper",
    language=None,
    min_chunk_size=1.0,
    model_cache_dir=None,
    model_dir=None,
    lan="en",
    task="transcribe",
    vac=False,
    vac_chunk_size=0.04,
    vad=True,
    buffer_trimming="segment",
    buffer_trimming_sec=15,
    log_level="DEBUG",
)

PARALLEL_ASR = ParallelRealtimeASR(modelsize=WHISPER_CONFIG.model, logger=setup_logging("asr"), warmup_file=WHISPER_CONFIG.warmup_file)

LOGGER = setup_logging("Layer-server", use_stdout=True)
SERVER_CONFIG = SimpleNamespace(
    max_workers=20, # max number of concurrent clients for the grpc server
    log_every_processor=False, # set to true id you want to log every asr on separate files, care for filedescriptor limit error (too many files opened)
    ports=[50051, 50052],
)

# should be used only inside one single service 
class TrascriptionManager:

    def __init__(self):
        self.last_end = None

    def format_transcript(self, t): 
        """
        Code from original whisper_online_server.py
        Formats the asr output properly for the client:
            eg: 1500 3600 this is a message
        the first three words are:
        - Emission time from beginning of processing, in milliseconds
        - beginnign and end timestamp of the text segment, as estimated by 
          Whisper model. The timestamps are not accurate, but they're useful anyway
        - Corresponing text segment transcription.
        """
        if t[0] is not None:
            beg, end = t[0]*1000, t[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            if beg < 0: beg = 0 
            self.last_end = end
            return (int(round(beg)), int(round(end)), t[2])
        else:
            return None


# should be used only inside one single service 
class ProcessorManager:

    def __init__(self, id, logger, timeout=5.0):
        self.processor = ParallelOnlineASRProcessor(asr=PARALLEL_ASR.asr, logger=logger) #ParallelOnlineASRProcessor will not modify the asr object 
        self.audio_queue = asyncio.Queue() # received thread setup, these queues are thread sage
        self.id = id
        self.logger = logger
        self.__timeout = timeout

    async def insert_audio(self):
        """
        Insert audio chunks in the processor.
        """
        audio_batch = []

        while not self.audio_queue.empty(): #Read all the audio chunks in the queue
            chunk = await self.audio_queue.get()
            audio_batch.extend(chunk)

        self.processor.insert_audio_chunk(np.array(audio_batch, dtype=np.float32))

    @asynccontextmanager
    async def context(self):
        """
        Context manager to use the processor using the 'with' context
        register, process inside thw 'with' statement and finally unregister the processor.
        """
        self.processor.init()
        # this is made so that the service is registered only when it get some audio 
        # for the first time and not at the exact moment the client connects
        # this ensure the running services are not slowed by new incoming connections 
        try: #TODO: timeout
            while self.audio_queue.qsize() < 2: # wait for the first two audio chunks
                await asyncio.sleep(0.01)
            PARALLEL_ASR.register_processor(self.id, self.processor)
            LOGGER.debug(f"{self.id} accumulated {self.audio_queue.qsize()} chunks for the first time")

            yield #here is where the processor is used

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for audio chunks in {self.id}")
            LOGGER.warning(f"Timeout waiting for audio chunks in {self.id}")
            raise
        except Exception as e:
            self.logger.exception(f"Error {e} in {self.id}")  
            LOGGER.exception(f"Error {e} in {self.id}")
        finally:
            PARALLEL_ASR.unregister_processor(self.id)

# gRPC service class
class SpeechToTextServicer(speech_pb2_grpc.SpeechToTextServicer):
    """
    Implements the server part of the real-time speech-to-text 
    bidirectional streaming service (The whisper streaming service).
    """

    service_id = itertools.count()

    @classmethod
    def __get_unique_name(cls): 
        # this is safe thanks to cpython GIL running in 
        return f"Whisper-service-{next(cls.service_id)}"
    
    def __log_setup(self, id):
        if SERVER_CONFIG.log_every_processor:
            return setup_logging(f"{id}") 
        else:
            return logging.getLogger(__name__)

    async def __request_enqueuer(self, request_iterator, processor_manager):
        """
        Run in a separate thread.
        Reads audio chunk from iterator and puts it in the queue,
        this is a separate thread, to have a non blocking receiver 
        for StreamingRecognize.
        """
        try:
            async for audio_chunk in request_iterator:
                await processor_manager.audio_queue.put(audio_chunk.samples)
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in Request Enqueuer {processor_manager.id}")

    def create_grpc_response(self, start, end, text):
        return speech_pb2.Transcript(
            start_time_millis=start,
            end_time_millis=end,
            text=text
        )

    ### Main service function ###

    async def StreamingRecognize(self, request_iterator, context):
        """
        Async real-time speech-to-text streaming service.
        Keep transcribing until client stops streaming audio.
        Starts the request_enqueuer (non blocking receive)
        """
        id = self.__get_unique_name()
        logger = self.__log_setup(id)
        logger.info("Server started") # starting the online asr
        LOGGER.info(f"Started connection on {id}")

        processor_manager = ProcessorManager(id, logger) # this is the processor manager, it will handle the audio chunks and the asr processor
        transcript_manager = TrascriptionManager() 

        request_task = asyncio.create_task(self.__request_enqueuer(
            request_iterator,
            processor_manager
        )) # unblocking receiver coroutine

        # this is the main loop, it will run until the client stops streaming
        async with processor_manager.context(): 
            while not request_task.done():
                if processor_manager.audio_queue.empty(): 
                    await asyncio.sleep(0.01)
                    continue
                await processor_manager.insert_audio()

                PARALLEL_ASR.set_processor_ready(id)

                await PARALLEL_ASR.wait()# async wait for transcription event to be signaled 

                transcript = processor_manager.processor.results
                if transcript:
                    # but this means the asr did dont trasctibe our buffer we need to restart 
                    # (this can only happen upon first connection) #TODO: a better way to solve this
                    fmt_t = transcript_manager.format_transcript(transcript)
                    if fmt_t: # send actual result back to the client
                        yield self.create_grpc_response(*fmt_t)  

        # send the final result and close 
        final_transcript = processor_manager.processor.flush_everything()
        fmt_final_t = transcript_manager.format_transcript(final_transcript)
        if fmt_final_t:
            yield self.create_grpc_response(*fmt_final_t)  

        logger.info(f"Finished streaming process of {id}")
        LOGGER.info(f"Finished streaming process in {id}")

### gRPC SERVER SETUP ###

async def serve():
    """
    Starts the asr, starts the grpc service pool and opens ports,
    then waits until termination.
    """
    PARALLEL_ASR.start()
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=SERVER_CONFIG.max_workers), 
    )

    speech_pb2_grpc.add_SpeechToTextServicer_to_server(SpeechToTextServicer(), server)
    for port in SERVER_CONFIG.ports: #TODO: add secure port 
        server.add_insecure_port(f'[::]:{port}')

    await server.start()
    LOGGER.info("Server started")

    try: await server.wait_for_termination()
    except KeyboardInterrupt: LOGGER.error("Interrupted")
    finally: await server.stop(0)

def main():
    asyncio.run(serve())

