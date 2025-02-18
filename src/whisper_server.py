#!/usr/bin/env python3
import time, logging, os, sys, json, threading,  queue
from datetime import datetime
from concurrent import futures
from argparse import Namespace

import numpy as np
import grpc
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

    _lock = threading.Lock()
    MAX_WORKERS = 20
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
            return (int(round(beg)),int(round(end)),o[2]), last_end
        else:
            return None, last_end

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

        id = self.get_unique_logger_name()
        logger = setup_logging(f"Whisper-server-{id}") 
        logger.info("Server started") # starting the online asr
        LOGGER.info(f"Started connection on {logger.name}")
        #setting up the processor, should not call asr outside, but this is an exceptional case to make a single model work with legacy code
        #ParallelOnlineASRProcessor will not modify the asr object 
        online = ParallelOnlineASRProcessor(asr=PARALLEL_ASR.asr, logger=logger)

        try:
            online.init()
            last_end = None # for formatting the output, same format used in original whisper-streaming server 

            audio_queue = queue.Queue() # received thread setup, these queues are thread sage
            result = {"result": None}  # resut holder
            receiver = threading.Thread(
                target=self.request_enqueuer, 
                args=(request_iterator, audio_queue, logger), 
                daemon=True
            )
            receiver.start()

            audio_batch = []
            is_first = True

            while(True):
                if audio_queue.empty(): # break if receiver dies 
                    if not receiver.is_alive():
                        break
                    continue
                # this is made so that the service is registered only when it get some audio 
                # for the first time and not at the exact moment the client connects
                # this ensure the running services are not slowed by new incoming connections 
                if is_first: 
                    if audio_queue.qsize() > 1:
                        PARALLEL_ASR.register_processor(id, online, result)
                        is_first = False
                        LOGGER.debug(f"whisper-service-{id} I accumulated 2 chunks for the first time, ready!")
                    continue

                while not audio_queue.empty(): #Read all the audio chunks in the queue
                    try:
                        audio_batch.extend(audio_queue.get())
                    except queue.Empty:
                        break            

                a = np.array(audio_batch, dtype=np.float32) #TODO: normalize only if needed (should be done directly by the client)
                audio_batch.clear()
                online.insert_audio_chunk(a)

                PARALLEL_ASR.set_processor_ready(id)
                PARALLEL_ASR.wait()
                t = result["result"]

                if t == None: continue # if we set ready while the asr is transcribing we will receive the event of transcription end adn break the waiting
                # but this means the asr did dont trasctibe our buffer we need to restart 
                # (this can only happend upon first connection) #TODO: a better way to solve this
                fmt_t, last_end = self.format_output_transcript(last_end, t) #TODO: is last end useful?
                if fmt_t is not None: # send actual result back to the client
                    yield speech_pb2.Transcript(start_time_millis=fmt_t[0], end_time_millis=fmt_t[1],text=fmt_t[2]) 
        except Exception as e:
            logger.exception(f"{e}")
            LOGGER.error(f"Error {e} in {logger.name}")
        finally:
            logger.info("Finished streaming process")
            LOGGER.info(f"Finished streaming process in {logger.name}")
            PARALLEL_ASR.unregister_processor(id) 
            # no need to online.finish() the processor, this thread is going to shut down, 
            # a new one will be instantiated for new connections in new threads

def serve(): #TODO: handle clients that take to long to send messages and are slowing everyone
    PARALLEL_ASR.start()
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

