import asyncio
import numpy as np, logging, os, threading, time, copy
from types import SimpleNamespace
from src.whisper_online import *

class ParallelAudioBuffer:
    """A shared audio buffer for parallel processing.
    This class is used to store audio chunks and their corresponding ids and segment timesfor each client (or segment).
    """

    def __init__(self):
        self._audio = np.array([],dtype=np.float32)        
        self._segment_times = []
        self._ids = []

    def reset(self):
        """resets all fields to constructor init"""
        self.__dict__.update(self.__class__().__dict__)

    def append_token(self, id, audio):
        audio_lenth = len(audio)
        if audio_lenth == 0:
            return
        buffer_length = self.size 
    
        self._segment_times.append({"start": buffer_length, "end":(buffer_length + audio_lenth)})
        self._ids.append(id)
        self._audio = np.append(self._audio, audio)
        self._audio = np.append(self._audio, np.zeros(100, dtype=np.float32))

    @property
    def size(self):
        return len(self._audio)

    def __len__(self):
        return self.size

    def parameters(self):
        ns = SimpleNamespace(ids=self._ids, audio=self._audio, segment_times=self._segment_times)
        return copy.deepcopy(ns)

class MultiProcessingFasterWhisperASR(FasterWhisperASR):
    """A paralell implementation of the whisper-streaming FasterWhisperASR legacy class.
    The transcribe method use a batched pipeline, and a SharedAudioBuffer to handle multiple clients.
    """

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=logging.getLogger(__name__)):
        self._client_events = [] # events to signal the clients when their transcription is done
        self._last_transcript_time = 0.0 # last transcription time
        self._log = logfile
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile)# cant use vad if segment times self.use_vad() 

    @staticmethod
    def normalize_segment(start, segment):
        """
        Normalizes the segment timestamps to their relative start time.
        """
        o = []
        for word in segment.words:
            if segment.no_speech_prob > 0.9:
                continue
            # not stripping the spaces -- should not be merged with them!
            t = (round(word.start - start, 5), round(word.end - start, 5), word.word)
            o.append(t)
        return o

    def warmup(self, filepath): 
        """
        original repo asr warmup code, (without logging) 
        warm up the ASR because the very first transcribe takes more time than the others. 
        Test results in https://github.com/ufal/whisper_streaming/pull/81
        """
        if filepath: 
            if os.path.isfile(filepath):
                audio = load_audio_chunk(filepath,0,1)
                buffer = ParallelAudioBuffer()
                buffer.append_token(1, audio)
                self.transcribe_parallel(buffer)
                self._log.info("asr is warmed up") 
            else: self._log.info(f"{filepath} not found")
        else: self._log.info("no warmup file provided or file not found")


    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        This methods is needed to override the model in the superclass 
        constructor to use the batched pipeline of faster-whisper
        """
        from faster_whisper import BatchedInferencePipeline
        model = super().load_model(modelsize, cache_dir, model_dir)
        pipe = BatchedInferencePipeline(model)
        return pipe 

    def transcribe_parallel(self, audio_buffer: ParallelAudioBuffer):
        """
        Given a ParallelAudioBuffer, transcribe and returns the 
        segments tagged with the the corresponding segments for each client.
        Should be used for multiple audio inference
        """
        if not len(audio_buffer): # not processing if empty  
            return [] 
        
        parameters = audio_buffer.parameters()  
        timestamp = time.time()

        segments, _ = self.model.transcribe(
            parameters.audio, 
            beam_size=5, 
            condition_on_previous_text=False, 
            multilingual=True, 
            word_timestamps=True, 
            clip_timestamps=parameters.segment_times,
            batch_size=16, 
        ) #check if info util

        # segments is the segment generator produced by transcribe, applying list generate the segments
        segment_list = list(segments) # list(segments) is the real consuming part of generating transcriptions

        #[self._log.info(s.text) for s in segment_list] #log the segments just transcribed
        # this part zip (processor_id, start_time, segment) to the corresponding id
        # then using the start time it shift the word's timestamp in segment to the real timestamp 
        # in the processor that requested the transcription using normalize_segment 
        results_tagged = [
            (id, self.normalize_segment(start, seg)) for (id, start, seg) in list(zip(
                parameters.ids,
                [time["start"]/OnlineASRProcessor.SAMPLING_RATE for time in parameters.segment_times], 
                segment_list
            )
        )]

        self._last_transcript_time = time.time() - timestamp 
        self._log.debug(f"transcription time: {self._last_transcript_time} seconds")

        return results_tagged 

class ParallelOnlineASRProcessor(OnlineASRProcessor):
    """An OnlineASRProcessor that can be used in parallel with other processors.

    This subclass is used to make OnlineASRProcessor compatible with the ParallelRealtimeASR class.
    Implements new methods reusing the original OnlineASRProcessor code, keeping only the necessary modifications.
    """

    def __init__(self, asr, logger=logging.getLogger(__name__)):
        super().__init__(asr)
        self.__logger = logger
        self.__result = None
        self.buffer_trimming_sec = 10 #overwriting default trimming sec 

    @property
    def buffer_time_seconds(self):
        return len(self.audio_buffer)/self.SAMPLING_RATE

    def flush_everything(self):
        """
        Flushes the buffer and returns the results.
        """
        self.__logger.debug("Flushing everything")
        return self.to_flush(self.transcript_buffer.complete())


    def update(self, results):
        self.__logger.debug("ITERATION START\n")
        self.__logger.debug(f"transcribing {self.buffer_time_seconds:2.2f} seconds from {self.buffer_time_offset:2.2f}")

        self.transcript_buffer.insert(results, self.buffer_time_offset)

        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        completed = self.to_flush(o)
        self.__logger.debug(f">>>>COMPLETE NOW: {completed}")

        the_rest = self.to_flush(self.transcript_buffer.complete())
        self.__logger.debug(f"INCOMPLETE: {the_rest}")

        self._chunk_buffer_at()

        self.__logger.info(f"len of buffer now: {self.buffer_time_seconds:2.2f}")
        self.__logger.debug("ITERATION END \n")

        self.__result = self.to_flush(o)

    @property
    def results(self):
        return self.__result

    def _chunk_buffer_at(self):
        """
        Chunking the audio buffer on the end timestamps of the last committed words.
        """
        # Chunking here check original OnlineASRProcessor: chunck the buffer on on last committed work  
        k = len(self.commited)-1
        s = self.buffer_trimming_sec
        if self.buffer_time_seconds > s and k >= 0:
            l = self.buffer_time_offset + self.buffer_time_seconds - (s//2) 
            while k>0 and self.commited[k][1] > l:
                k -= 1
            t = self.commited[k][1] 
            self.__logger.debug(f"chunking segment at word {self.commited[-1]} at {t}")
            self.chunk_at(t)

from typing import Dict
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class RegisteredProcess():
    """Store data used to progress the ParallalASRProcess in parallel
    """
    asr_processor : ParallelOnlineASRProcessor 
    ready_flag : bool = False

class ParallelRealtimeASR():
    """Implements an adaptation of whisper streaming parallelized to process multiple clients.

    An instance of this class will run on a separate thread 

    How to use:
    1. Create an instance of this class
    2. start the instance with the start method
    3. Register the processors with the register_processor method when they connect
    4. Set the processor ready with the set_processor_ready method when a processor has received new audio
    5. Wait for the transcription to be done with the wait method
    6. Get the results from the result_holder dict shared with the processor
    7. Unregister the processor with the unregister_processor method when the client disconnects

    Multiple processors can be registered at the same time, running asynchronously. 
    The asr will handle synchronization and transcription of the audio streams.

    Check out whisper_server.py for an example of this class usage.

    """

    def __init__(self, modelsize="large-v3-turbo", logger=logging.getLogger(__name__), warmup_file=None): # type specified for a more comprehensive code 
        self.__registered_pids: Dict[int, RegisteredProcess] = {} # dict of {processor_id; (processor, result_holder)}
        self.__register_lock = threading.RLock()
        self.__transcription_event = asyncio.Event()
        self.__audio_buffer = ParallelAudioBuffer()
        self.__logger = logger
        self.__thread = threading.Thread(target=self.__asr_loop, args=(), daemon=True)
        self.__asr = MultiProcessingFasterWhisperASR("auto", modelsize=modelsize, logfile=logger) 
        self.__logger.info("asr created") 
        if warmup_file:
            self.__asr.warmup(warmup_file)

    @property
    def asr(self):
        """
        this is only for parallel asr processors creation
        wont be used for inference, just for whisper-streaming legacy code
        do not use the asr object outside this class to avoid issues
        """
        return self.__asr

    def register_processor(self, id, asr_processor):
        with self.__register_lock:
            self.__registered_pids[id] = RegisteredProcess(
                asr_processor=asr_processor,
                ready_flag=False,
            )

    def unregister_processor(self, id):
        with self.__register_lock:
            return self.__registered_pids.pop(id)

    def append_audio(self, id, audio): 
        self.__audio_buffer.append_token(id, audio)

    def start(self):
        self.__thread.start()

    def set_processor_ready(self, id):
        with self.__register_lock:
            if id in self.__registered_pids:
                self.__registered_pids[id].ready_flag = True
            else: 
                raise ValueError(f"{id} is not a registered processor.") 

    async def wait(self):
        try:
            await asyncio.wait_for(self.__transcription_event.wait(), timeout=2) 
        except asyncio.TimeoutError:
            self.__logger.error("Timeout waiting for transcription")

    def __all_pid_ready(self):
        with self.__register_lock:
            return len(self.__registered_pids) > 0 and all([x.ready_flag for x in self.__registered_pids.values()])

    def __reset_ready_pids(self):
        with self.__register_lock:
            for key in self.__registered_pids: 
                self.__registered_pids[key].ready_flag = False

    def __asr_loop(self):
        """
        realtime parallel asr loop
        this function must be called in a thread specific for the parallel processing
        the other threads receveind the audio streams must push the audio chunks in the buffer and wait for the results 
        """
        self.__logger.info("asr started")
        timestamp = time.time()
        try:
            while True:
                self.__transcription_event.clear()

                with self.__register_lock: #If everyone is here trascribe
                    if not self.__all_pid_ready(): 
                        continue
                    self.__reset_ready_pids()

                    self.__logger.debug(f"Time lost waiting {time.time() - timestamp} seconds")
                    self.__logger.info("Transcribing") #TODO:remove

                    current_processors = self.__registered_pids.copy()
                    for processor_id in self.__registered_pids:
                        processor = self.__registered_pids[processor_id]
                        #result_holder["result"] = None 
                        self.append_audio(processor_id, processor.asr_processor.audio_buffer) 
                    self.__logger.debug(f"Shared buffer samples: {len(self.__audio_buffer)}")

                # this will block until the transcribe is done can take more than 1 second
                results = self.__asr.transcribe_parallel(self.__audio_buffer)
                self.__audio_buffer.reset()

                with self.__register_lock: 
                    for (processor_id, result) in results:
                        processors = current_processors[processor_id]
                        processors.asr_processor.update(result) 
                        self.__logger.debug(f"Result {processor_id}: {processors.asr_processor.results}")

                self.__transcription_event.set()
                timestamp = time.time()

        except KeyboardInterrupt:
            self.__logger.info("Interrupted")
            return 
        except Exception as e: #TODO: restart the asr properly in case of error
            self.__logger.exception(e)
            return

