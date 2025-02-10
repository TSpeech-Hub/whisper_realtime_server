#!/usr/bin/env python3
import time, logging, os, sys, json, threading,  queue
from datetime import datetime
from concurrent import futures
from argparse import Namespace

import numpy as np
import grpc
import speech_pb2, speech_pb2_grpc
from parallel_whisper_online import MultiProcessingFasterWhisperASR, ParallelOnlineASRProcessor

#NOTE: LOGGING SETUP FUNCTION
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

#NOTE: Config Constants
with open("config.json") as file: 
    WHISPER_CONFIG = Namespace(**json.load(file))

#NOTE: we should keep the streaming alive, there can be silence during the audio stream
GRPC_SERVER_OPTIONS = [
    ('grpc.keepalive_time_ms', 10000),           # ping every 10 seconds  
    ('grpc.keepalive_timeout_ms', 5000),           # timeout of 5 seconds for the pong response 
    ('grpc.keepalive_permit_without_calls', 1),    # ping without activer rpc  
]

SHARED_WHISPER_ASR = MultiProcessingFasterWhisperASR(
    lan="auto", 
    logger=setup_logging("asr"), 
        modelsize=WHISPER_CONFIG.model
)
SHARED_WHISPER_ASR.warmup(WHISPER_CONFIG.warmup_file)

#NOTE: gRPC service class
class SpeechToTextServicer(speech_pb2_grpc.SpeechToTextServicer):

    _lock = threading.Lock()
    MAX_WORKERS = 10
    _logger_counter = 0

    @classmethod
    def get_unique_logger_name(cls):
        with cls._lock:
            cls._logger_counter += 1
            return f"{cls._logger_counter}"

    @staticmethod
    def format_output_transcript(last_end, o):
        """code from original whisper_online_server.py"""
        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if last_end is not None:
                beg = max(beg, last_end)

            last_end = end
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            return None

    def request_enqueuer(self, request_iterator, audio_queue, logger):
        """
        reads audio chunk from iterator and puts it in the queue,
        this is a separate thread, to have a non blocking receiver for the audio chunks.
        This ensure that the services are able to process multiple audio chunks at the same time.
        """
        try:
            for audio_chunk in request_iterator:
                audio_queue.put(audio_chunk.samples)
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in {logger.name}")

    def StreamingRecognize(self, request_iterator, context):
        """
        The real-time speech-to-text streaming service.
        """

        logger = setup_logging(f"Whisper-server-{self.get_unique_logger_name()}") 
        logger.info("Server started") # starting the online asr
        LOGGER.info(f"Started connection on {logger.name}")

        online = ParallelOnlineASRProcessor( #setting up the asr 
            SHARED_WHISPER_ASR, 
            logger=logger,
        )

        try:
            online.init()
            last_end = None # for formatting the output, same format used in original whisper-streaming server

            audio_queue = queue.Queue() # received thread setup 
            receiver = threading.Thread(
                target=self.request_enqueuer, 
                args=(request_iterator, audio_queue, logger), 
                daemon=True
            )
            receiver.start()

            audio_batch = []
            while(True):
                if audio_queue.empty(): # break if receiver dies 
                    if not receiver.is_alive():
                        break
                    time.sleep(0.1) #wait for audio chunk to arrive next audio chunk.  TODO: find better solution
                    continue

                audio_batch.clear()
                while not audio_queue.empty(): #Read all the audio chunks in the queue
                    try:
                        audio_batch.extend(audio_queue.get())
                    except queue.Empty:
                        break            

                if len(audio_batch) != 0:
                    a = np.array(audio_batch, dtype=np.float32) #TODO: normalize only if needed (should be done directly by the client)
                    online.insert_audio_chunk(a)
                    raw = online.parallel_process_iter()
                    transcripted = self.format_output_transcript(last_end, raw)

                    if transcripted is not None: # send actual result back to the client
                        yield speech_pb2.Transcript(text=transcripted) 
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in {logger.name}")
        finally:
            logger.info("Finished streaming process")
            LOGGER.info(f"Finished streaming process in {logger.name}")
            online.finish()

def serve():
    threading.Thread(target=SHARED_WHISPER_ASR.realtime_parallel_asr_loop, daemon=True).start() #NOTE: this is a local shared whisper model working in parallel
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=SpeechToTextServicer.MAX_WORKERS), options=GRPC_SERVER_OPTIONS)
    speech_pb2_grpc.add_SpeechToTextServicer_to_server(SpeechToTextServicer(), server)
    server.add_insecure_port('[::]:50051') #TODO: add secure port 
    server.start()
    LOGGER.info("Server started")
    try: #TODO: remove timeout test
        while True:
            time.sleep(86400) 
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()

