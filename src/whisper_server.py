#!/usr/bin/env python3
import logging, os, sys, json, threading,  queue
from datetime import datetime
from concurrent import futures
from argparse import Namespace

import numpy as np
import grpc
from grpc import aio
import asyncio
import itertools
from generated import speech_pb2_grpc, speech_pb2
from parallel_whisper_online import *

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

LOGGER = setup_logging("Layer-server", use_stdout=True)

# Config Constants
# contains the default values for the server (check in config.json)
with open("config.json") as file: 
    WHISPER_CONFIG = Namespace(**json.load(file))

# should keep the streaming alive, there can be silence during the audio stream
GRPC_SERVER_OPTIONS = [
    ('grpc.keepalive_time_ms', 10000),           # ping every 10 seconds  
    ('grpc.keepalive_timeout_ms', 5000),           # timeout of 5 seconds for the pong response 
    ('grpc.keepalive_permit_without_calls', 1),    # ping without activer rpc  
]

PARALLEL_ASR = ParallelRealtimeASR(modelsize=WHISPER_CONFIG.model, logger=setup_logging("asr"), warmup_file=WHISPER_CONFIG.warmup_file)

# gRPC service class
class SpeechToTextServicer(speech_pb2_grpc.SpeechToTextServicer):
    """
    Implements the server part of the real-time speech-to-text 
    bidirectional streaming service (The whisper streaming service).

    Class attribures
    ----------
    MAX_WORKERS: 
        The maximum concurrent StreamingRecognize services
    service_id: 
        A thread safe cycle for unique name assignment to services
        Ranging from 0 to MAX_WORKERS
    """

    MAX_WORKERS = 20
    service_id = itertools.cycle(range(1, MAX_WORKERS+1))

    @classmethod
    def get_unique_logger_name(cls): 
        # this is safe thanks to cpython GIL running in 
        return f"Whisper-service-{next(cls.service_id)}"

    @staticmethod
    def format_output_transcript(last_end, o):
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
        if o[0] is not None:
            beg, end = o[0]*1000, o[1]*1000
            if last_end is not None:
                beg = max(beg, last_end)

            last_end = end
            return (int(round(beg)), int(round(end)), o[2]), last_end
        else:
            return None, last_end

    async def __request_enqueuer(self, request_iterator, audio_queue, logger):
        """
        Run in a separate thread.
        Reads audio chunk from iterator and puts it in the queue,
        this is a separate thread, to have a non blocking receiver 
        for StreamingRecognize.
        """
        try:
            async for audio_chunk in request_iterator:
                await audio_queue.put(audio_chunk.samples)
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in {logger.name}")

    async def StreamingRecognize(self, request_iterator, context):
        """
        Async real-time speech-to-text streaming service.
        Keep transcribing until client stops streaming audio.
        Starts the request_enqueuer (non blocking receive)
        """

        id = self.get_unique_logger_name()
        logger = setup_logging(f"{id}") 
        logger.info("Server started") # starting the online asr
        LOGGER.info(f"Started connection on {logger.name}")
        #setting up the processor, should not call asr outside, but this is an exceptional case to make a single model work with legacy code
        #ParallelOnlineASRProcessor will not modify the asr object 
        last_end = None # for formatting the output, same format used in original whisper-streaming server 
        audio_batch = [] # used to collect chunks in the audio batch
        is_first = True # understand if first iteration

        online = ParallelOnlineASRProcessor(asr=PARALLEL_ASR.asr, logger=logger)
        audio_queue = asyncio.Queue() # received thread setup, these queues are thread sage
        request_task = asyncio.create_task(self.__request_enqueuer(request_iterator, audio_queue, logger)) # unblocking receiver coroutine

        try:
            # the asyncio.sleeps return the control to the event loop,
            # helping other coroutines (services) to progress
            online.init()
            while True:
                if audio_queue.empty(): # break if receiver dies 
                    if request_task.done():
                        break
                    await asyncio.sleep(0.01)
                    continue
                # this is made so that the service is registered only when it get some audio 
                # for the first time and not at the exact moment the client connects
                # this ensure the running services are not slowed by new incoming connections 
                if is_first: 
                    if audio_queue.qsize() > 1:
                        PARALLEL_ASR.register_processor(id, online)
                        is_first = False
                        LOGGER.debug(f"whisper-service-{id} I accumulated {audio_queue.qsize()} chunks for the first time, ready!")
                    else:
                        await asyncio.sleep(0.01)
                        continue

                while not audio_queue.empty(): #Read all the audio chunks in the queue
                    chunk = await audio_queue.get()
                    audio_batch.extend(chunk)

                online.insert_audio_chunk(np.array(audio_batch, dtype=np.float32))
                audio_batch.clear()

                PARALLEL_ASR.set_processor_ready(id)
                await asyncio.to_thread(PARALLEL_ASR.wait)# wait for transcription event to be signaled (blocking call converted to async sleep)
                t = online.results
                if t is None:
                    continue
                # but this means the asr did dont trasctibe our buffer we need to restart 
                # (this can only happen upon first connection) #TODO: a better way to solve this
                fmt_t, last_end = self.format_output_transcript(last_end, t)
                if fmt_t is not None: # send actual result back to the client
                    yield speech_pb2.Transcript(start_time_millis=fmt_t[0],
                                                  end_time_millis=fmt_t[1],
                                                  text=fmt_t[2])
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in {logger.name}")
        finally:
            logger.info("Finished streaming process")
            LOGGER.info(f"Finished streaming process in {logger.name}")
            PARALLEL_ASR.unregister_processor(id)

async def serve():
    """
    Starts the asr, starts the grpc service pool and opens ports,
    then waits until termination.
    """
    PARALLEL_ASR.start()
    server = aio.server(futures.ThreadPoolExecutor(max_workers=SpeechToTextServicer.MAX_WORKERS),
                        options=GRPC_SERVER_OPTIONS)
    speech_pb2_grpc.add_SpeechToTextServicer_to_server(SpeechToTextServicer(), server)
    server.add_insecure_port('[::]:50051') #TODO: add secure port 
    server.add_insecure_port('[::]:50052') #TODO: find optimal port range 
    await server.start()
    LOGGER.info("Server started")
    try: 
        await server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
    finally:
        await server.stop(0)

def main():
    asyncio.run(serve())

if __name__ == '__main__':
    main()

