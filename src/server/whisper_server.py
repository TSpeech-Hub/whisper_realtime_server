#!/usr/bin/env python3
import logging, os, sys
from datetime import datetime
from concurrent import futures
from contextlib import asynccontextmanager
import traceback

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

class TranscriptionManager:
    def __init__(self):
        self.last_end = None

    def format_transcript(self, t):
        if t and t[0] is not None: # what if t is null
            beg, end = t[0]*1000, t[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            if beg < 0: beg = 0
            self.last_end = end
            return True, (int(round(beg)), int(round(end)), t[2])
        else:
            return False, (0, 0, "")

class ProcessorManager:
    def __init__(self, id, logger, timeout=5.0):
        self.processor = ParallelOnlineASRProcessor(asr=PARALLEL_ASR.asr, logger=logger)
        self.audio_queue = asyncio.Queue()
        self.id = id
        self.logger = logger
        self.__timeout = timeout

    async def insert_audio(self):
        audio_batch = []
        while not self.audio_queue.empty():
            chunk = await self.audio_queue.get()
            audio_batch.extend(chunk)
        self.processor.insert_audio_chunk(np.array(audio_batch, dtype=np.float32))

    @asynccontextmanager
    async def context(self):
        self.processor.init()
        try:
            while self.audio_queue.qsize() < 2:
                await asyncio.sleep(0.01)
            PARALLEL_ASR.register_processor(self.id, self.processor)
            LOGGER.debug(f"{self.id} accumulated {self.audio_queue.qsize()} chunks")
            yield
        except Exception as e:
            LOGGER.error(f"Exception in ProcessorManager {self.id}: ") 
            print(traceback.format_exception(e))
        finally:
            PARALLEL_ASR.unregister_processor(self.id)
            LOGGER.debug(f"{self.id} finished processing")

class CommonStreamingUtils:
    service_id = itertools.count()

    @classmethod
    def get_unique_name(cls):
        return f"Whisper-service-{next(cls.service_id)}"

    @staticmethod
    def log_setup(id):
        if SERVER_CONFIG.log_every_processor:
            return setup_logging(f"{id}")
        else:
            return logging.getLogger(__name__)

    @staticmethod
    async def request_enqueuer(request_iterator, processor_manager):
        try:
            async for audio_chunk in request_iterator:
                await processor_manager.audio_queue.put(audio_chunk.samples)
        except Exception as e:
            processor_manager.logger.error(f"Exception in request_enqueuer {processor_manager.id}: {e}")

    @staticmethod
    def create_response(start, end, text):
        return speech_pb2.Transcript(
            start_time_millis=start,
            end_time_millis=end,
            text=text
        )

    @staticmethod
    def create_response_with_hypothesis(start_t, end_t, text, start_h, end_h, hypothesis):
        return speech_pb2.TranscriptWithHypothesis(
            confirmed=speech_pb2.Transcript(
                start_time_millis=start_t,
                end_time_millis=end_t,
                text=text
            ),
            hypothesis=speech_pb2.Transcript(
                start_time_millis=start_h,
                end_time_millis=end_h,
                text=hypothesis
            )
        )

class StandardSpeechToTextServicer(speech_pb2_grpc.SpeechToTextServicer):

    async def StreamingRecognize(self, request_iterator, context):
        id = CommonStreamingUtils.get_unique_name()
        logger = CommonStreamingUtils.log_setup(id)
        LOGGER.info(f"Started connection on {id}")

        processor_manager = ProcessorManager(id, logger)
        transcript_manager = TranscriptionManager()
        request_task = asyncio.create_task(CommonStreamingUtils.request_enqueuer(request_iterator, processor_manager))

        async with processor_manager.context():
            while not request_task.done():
                if processor_manager.audio_queue.empty():
                    await asyncio.sleep(0.01)
                    continue
                await processor_manager.insert_audio()
                PARALLEL_ASR.set_processor_ready(id)
                await PARALLEL_ASR.wait()
                transcript = processor_manager.processor.results
                if transcript:
                    ok, fmt = transcript_manager.format_transcript(transcript)
                    if ok:
                        yield CommonStreamingUtils.create_response(*fmt)

        final_transcript = processor_manager.processor.flush_everything()
        ok, fmt_final = transcript_manager.format_transcript(final_transcript)
        if ok:
            yield CommonStreamingUtils.create_response(*fmt_final)

class HypothesisSpeechToTextServicer(speech_pb2_grpc.SpeechToTextWithHypothesisServicer):

    async def StreamingRecognizeWithHypothesis(self, request_iterator, context):
        id = CommonStreamingUtils.get_unique_name()
        logger = CommonStreamingUtils.log_setup(id)
        LOGGER.info(f"Started connection on {id}")

        processor_manager = ProcessorManager(id, logger)
        transcript_manager = TranscriptionManager()
        hyp_transcript_manager = TranscriptionManager()
        request_task = asyncio.create_task(CommonStreamingUtils.request_enqueuer(request_iterator, processor_manager))

        async with processor_manager.context():
            while not request_task.done():
                if processor_manager.audio_queue.empty():
                    await asyncio.sleep(0.01)
                    continue
                await processor_manager.insert_audio()
                PARALLEL_ASR.set_processor_ready(id)
                await PARALLEL_ASR.wait()
                t = processor_manager.processor.results
                h = processor_manager.processor.hypothesis
                ok_t, fmt_t = transcript_manager.format_transcript(t)
                ok_h, fmt_h = hyp_transcript_manager.format_transcript(h)
                if ok_t or ok_h:
                    yield CommonStreamingUtils.create_response_with_hypothesis(*fmt_t, *fmt_h)

        final_t = processor_manager.processor.flush_everything()
        ok, fmt_final = transcript_manager.format_transcript(final_t)
        if ok:
            yield CommonStreamingUtils.create_response_with_hypothesis(*fmt_final, 0, 0, "")

async def serve():
    PARALLEL_ASR.start()
    server = aio.server(futures.ThreadPoolExecutor(max_workers=SERVER_CONFIG.max_workers))

    speech_pb2_grpc.add_SpeechToTextServicer_to_server(StandardSpeechToTextServicer(), server)
    speech_pb2_grpc.add_SpeechToTextWithHypothesisServicer_to_server(HypothesisSpeechToTextServicer(), server)

    for port in SERVER_CONFIG.ports:
        server.add_insecure_port(f"[::]:{port}")

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
