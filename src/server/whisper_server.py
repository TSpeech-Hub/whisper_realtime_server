#!/usr/bin/env python3
import logging, os, sys
from datetime import datetime
from concurrent import futures
from contextlib import asynccontextmanager
import argparse 

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
    def __init__(self, id, shared_asr, logger=logging.getLogger(__name__), server_logger=logging.getLogger(__name__), **kwargs):
        self.kwargs = kwargs
        self.id = id
        self.server_logger = server_logger
        self.processor = ParallelOnlineASRProcessor(asr=shared_asr.asr, logger=logger, **self.kwargs)
        self.audio_queue = asyncio.Queue()
        self.__shared_asr = shared_asr 
        self.logger = logger # same as the processor logger
        #self.__timeout = timeout

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
                await asyncio.sleep(0.1)
            self.__shared_asr.register_processor(self.id, self.processor)
            self.server_logger.debug(f"{self.id} accumulated {self.audio_queue.qsize()} chunks for the first time")
            yield
        except Exception as e:
            self.server_logger.error(f"Exception in ProcessorManager {self.id}: ") 
            self.server_logger.exception(e)
        finally:
            self.__shared_asr.unregister_processor(self.id)
            self.server_logger.debug(f"{self.id} finished processing")

class CommonStreamingUtils:
    __service_id = itertools.count()
    log_every_processor = False

    @classmethod
    def get_unique_name(cls):
        return f"Whisper-service-{next(cls.__service_id)}"

    @classmethod
    def log_setup(cls, id):
        if cls.log_every_processor:
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

    def __init__(self, shared_asr, main_server_logger, **kwargs): 
        super().__init__()
        self.__shared_asr = shared_asr
        self.__main_server_logger = main_server_logger
        self.__kwargs = kwargs

    async def StreamingRecognize(self, request_iterator, context):
        id = CommonStreamingUtils.get_unique_name()
        logger = CommonStreamingUtils.log_setup(id)
        self.__main_server_logger.info(f"Started connection on {id}")

        processor_manager = ProcessorManager(id, self.__shared_asr, logger=logger, server_logger=self.__main_server_logger, **self.__kwargs)
        transcript_manager = TranscriptionManager()
        request_task = asyncio.create_task(CommonStreamingUtils.request_enqueuer(request_iterator, processor_manager))

        async with processor_manager.context():
            while not request_task.done():
                if processor_manager.audio_queue.empty():
                    await asyncio.sleep(0.001)
                    continue
                await processor_manager.insert_audio()
                await asyncio.sleep(0.001)
                self.__shared_asr.set_processor_ready(id)

                # can happen that a new chunk is reveived after async waiting 
                await processor_manager.insert_audio()
                await self.__shared_asr.wait()
                transcript = processor_manager.processor.results
                if transcript:
                    ok, fmt = transcript_manager.format_transcript(transcript)
                    if ok:
                        yield CommonStreamingUtils.create_response(*fmt)

        final_transcript = processor_manager.processor.finish()
        ok, fmt_final = transcript_manager.format_transcript(final_transcript)
        if ok:
            yield CommonStreamingUtils.create_response(*fmt_final)

class HypothesisSpeechToTextServicer(speech_pb2_grpc.SpeechToTextWithHypothesisServicer):

    def __init__(self, shared_asr, main_server_logger, **kwargs): 
        super().__init__()
        self.__shared_asr = shared_asr
        self.__main_server_logger = main_server_logger
        self.__kwargs = kwargs

    async def StreamingRecognizeWithHypothesis(self, request_iterator, context):
        id = CommonStreamingUtils.get_unique_name()
        logger = CommonStreamingUtils.log_setup(id)
        self.__main_server_logger.info(f"Started connection on {id}")

        processor_manager = ProcessorManager(id, self.__shared_asr, logger=logger, server_logger=self.__main_server_logger, **self.__kwargs)
        transcript_manager = TranscriptionManager()
        hyp_transcript_manager = TranscriptionManager()
        request_task = asyncio.create_task(CommonStreamingUtils.request_enqueuer(request_iterator, processor_manager))

        async with processor_manager.context():
            while not request_task.done():
                if processor_manager.audio_queue.empty():
                    await asyncio.sleep(0.001)
                    continue
                await processor_manager.insert_audio()
                await asyncio.sleep(0.001)
                self.__shared_asr.set_processor_ready(id)

                await processor_manager.insert_audio()
                await self.__shared_asr.wait()

                t = processor_manager.processor.results
                h = processor_manager.processor.hypothesis
                ok_t, fmt_t = transcript_manager.format_transcript(t)
                ok_h, fmt_h = hyp_transcript_manager.format_transcript(h)
                if ok_t or ok_h:
                    yield CommonStreamingUtils.create_response_with_hypothesis(*fmt_t, *fmt_h)

        final_transcript = processor_manager.processor.finish()
        ok, fmt_final = transcript_manager.format_transcript(final_transcript)
        if ok:
            yield CommonStreamingUtils.create_response_with_hypothesis(*fmt_final, 0, 0, "")

async def serve(args):

    server_logger = setup_logging("Layer-server", use_stdout=True)

    server_logger.info("Starting server...")


    if args.log_every_processor: #TODO: fing a more elegant way idea to make this a global (settable) variable?
        CommonStreamingUtils.log_every_processor = True
        server_logger.info("Logging every processor in a separate file, be careful with the number of files generated, this should be used for debugging reasons only")

    if args.qratio_threshold <= 0 or args.qratio_threshold > 100:
        server_logger.error("qratio threshold must be between 0 and 100")
        sys.exit(1)

    if args.fallback:
        server_logger.info("Fallback logic enabled")
        if args.fallback_threshold <= 0:
            server_logger.error("Fallback threshold must be greater than 0")
            sys.exit(1)

    if args.buffer_trimming_sec <= 0:
        server_logger.error("Buffer trimming must be greater than 0")
        sys.exit(1)


    server_logger.info(f"Using faster-whisper model {args.model}")
    shared_asr = ParallelRealtimeASR(modelsize=args.model, logger=setup_logging("asr"), warmup_file=args.warmup_file)
    server_logger.info("Model loaded")


    shared_asr.start()
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers), 
        maximum_concurrent_rpcs=args.max_workers,
        options=[
            ('grpc.keepalive_time_ms', 1000), #ms
            ('grpc.keepalive_timeout_ms', 1000),
            ('grpc.keepalive_permit_without_calls', True),
        ]
    )

    processor_args = {
        "use_fallback": args.fallback,
        "fallback_threshold": args.fallback_threshold,
        "qratio_threshold": args.qratio_threshold,
        "buffer_trimming_sec": args.buffer_trimming_sec
    }

    speech_pb2_grpc.add_SpeechToTextServicer_to_server(StandardSpeechToTextServicer(shared_asr, server_logger, **processor_args), server)
    speech_pb2_grpc.add_SpeechToTextWithHypothesisServicer_to_server(HypothesisSpeechToTextServicer(shared_asr, server_logger, **processor_args), server)

    for port in args.ports:
        server.add_insecure_port(f"[::]:{port}")

    await server.start()
    server_logger.info("Server started")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        server_logger.error("Interrupted")
    finally:
        await server.stop(0)

def main():

    ### Services Hypothesis buffer args
    parser = argparse.ArgumentParser(description="Argument parser for the whisper-realtme-server")
    parser.add_argument("--fallback", action="store_true", help="Enable fallback logic when similarity local agreement fails for a mltitude of times")
    parser.add_argument("--fallback-threshold", type=float, default=1, help="threshold t for fallback logic after t+1 similarity local agreement fails (ignored if --fallback is not set)")
    parser.add_argument("--qratio-threshold", type=float, default=95, help="Threshold for qratio to confirm and insert new words using the hypothesis buffer (between 0 and 100), lower values than 90 are not recommended")
    parser.add_argument("--buffer-trimming-sec", type=int, default=15, help="Buffer trimming is the threshold in seconds that triggers the service processor audio buffer to be trimmed. This is useful to avoid memory leaks and to keep the buffer size under control. Default value is 15 seconds")

    ### gRPC Layer server args 
    parser.add_argument("--ports", type=int, nargs="+", default=[50051, 50052], help="Ports to run the server on")
    parser.add_argument("--max-workers", type=int, default=20, help="Max workers for the server")
    parser.add_argument("--log-every-processor", action="store_true", help="Log every processor in a separate file")
    # log folder 

    ### Whisper model args
    parser.add_argument('--model', type=str, default='large-v3-turbo', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo,turbo".split(","),help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir")
    parser.add_argument("--model-cache-dir", type=str, default=None, help="Directory for the whisper model caching")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory for a custom ct2 whisper model skipping if --model provided")
    parser.add_argument("--warmup-file", type=str, default="resources/sample1.wav", help="File to warm up the model and speed up the first request")

    ### Other args unused at the moment
    parser.add_argument("--lan", type=str, default="en", help="Language for the whisper model to translate to (unused at the moment)") 
    parser.add_argument("--vad", action="store_true", help="Use VAD for the model (unused at the moment)")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level for the server (DEBUG, INFO, WARNING, ERROR, CRITICAL) unused at the moment")

    asyncio.run(serve(parser.parse_args()))

