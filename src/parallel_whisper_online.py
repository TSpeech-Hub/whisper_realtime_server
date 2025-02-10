import sys, numpy as np, logging, os, threading, time, copy
from types import SimpleNamespace
from whisper_online import FasterWhisperASR, OnlineASRProcessor, load_audio_chunk

#TODO: WRITE DOC: all these classes should be used toghether with parallel threads, explain best practices and why i used the already implemented classes 
# (example: this concurrecy inheritance batched inferecence can be used with old online asr sequential class) 

#NOTE: separated lists to avoid reappending data in getters 
class ParallelAudioBuffer:
 
    def __init__(self):
        self._audio = np.array([],dtype=np.float32)        
        self._segment_times = []
        self._ids = []
        self._init_prompts = "" 

    def reset(self):
        self._audio = np.array([],dtype=np.float32)
        self._segment_times = []
        self._ids = []
        self._init_prompts = "" 

    def append_token(self, id, audio, init_prompt=""):
        audio_lenth = len(audio)
        if audio_lenth == 0:
            return
        buffer_length = self.size 
    
        self._init_prompts += init_prompt
        self._segment_times.append({"start": buffer_length, "end":(buffer_length + audio_lenth)})
        self._ids.append(id)
        self._audio = np.append(self._audio, audio)

    @property
    def size(self):
        return len(self._audio)

    def is_empty(self):
        return self.size ==  0

    def parameters(self):
        return copy.deepcopy(SimpleNamespace(ids=self._ids, audio=self._audio, segment_times=self._segment_times, init_prompts=self._init_prompts))

#TODO: change fields visibility 
class MultiProcessingFasterWhisperASR(FasterWhisperASR):

    def __init__(self, lan, logger, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self._audio_buffer_lock = threading.Lock() #lock for the shared buffer
        self._transctiptions_lock = threading.Lock() #lock for the shared results

        self._last_transcribed = [] # [(id, start, segments), ...]
        self._buffer = ParallelAudioBuffer() # shared buffer for the parallel processing
        self._client_events = [] # events to signal the clients when their transcription is done
        self._last_transcript_time = 0.0 # last transcription time

        self._log = logger
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile=logfile)# cant use vad if segment times self.use_vad() 

    #also we use segment timing, if the segment time start is not zero it mean we have to normalize the words time by segment start 
    def ts_words_normalized(self, segments):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for (start, segment) in segments:
            #normalization for word times: when sharing the buffer, segments start wont is shifted 
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                t = (word.start - start, word.end - start, word.word)
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
                a = load_audio_chunk(filepath,0,1)
                self.transcribe(a)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import BatchedInferencePipeline
        model = super().load_model(modelsize, cache_dir, model_dir)
        pipe = BatchedInferencePipeline(model)
        return pipe 

    def append_audio(self, id, event, audio, init_prompt=""): #NOTE: for shared processing only
        """make use of the shared buffer"""
        with self._audio_buffer_lock:
            self._client_events.append(event)
            self._buffer.append_token(id, audio, init_prompt)

    def transcribe_parallel(self):
        """
        inference on the shared buffer, can be shared between parallel online processors
        online processors can get results with get_last_transcribe method if pushed audio before
        cant select a specific language, will have transcribe task by default, NOTE: lang will be global
        """
        if self._buffer.is_empty(): #NOTE: not processing if empty  
            return 
        # resetting buffer, must be done before transcribing, other threads can queue up and push audio while transcribing
        timestamp = time.time()
        with self._audio_buffer_lock: #WARNING: mututal exclusion
            buffer_args = self._buffer.parameters()
            events = self._client_events 
            self._client_events = []
            self._buffer.reset()

        segments, _ = self.model.transcribe(buffer_args.audio, batch_size=16, multilingual=True, initial_prompt=buffer_args.init_prompts,beam_size=5, word_timestamps=True, condition_on_previous_text=True, clip_timestamps=buffer_args.segment_times) #check if info util
        with self._transctiptions_lock: # assoc, client ids and time offset to segments for helping the processors syncronize
            self._last_transcribed.extend(list(zip(
                buffer_args.ids, 
                [time["start"]/ParallelOnlineASRProcessor.SAMPLING_RATE for time in buffer_args.segment_times],list(segments)
            ))) 
            [e.set() for e in events] # set all client event to let them get the transcription 

        self._last_transcript_time = time.time() - timestamp 
        self._log.debug(f"transcription time: {self._last_transcript_time} seconds")
        time.sleep(0.05) # help client syncronize... #TODO: find a better way to syncronize
    
    def get_last_transcribed(self, id):
        """
        return the last transcribed segments associated with id (in order), if any thread is using the parallel transcribe it will wait until the transcribe is done
        collateral: will remove (and return) all the segments corresponding to id  
        """
        with self._transctiptions_lock:
            self._log.debug(f"Requesting segments for id: {id}")
            user_segments = [(start, s) for (i, start, s) in self._last_transcribed if i == id]
            if user_segments:
                self._log.debug(f"Current last_transcribed: {self._last_transcribed}")
                self._last_transcribed = [(i, start, s) for (i, start, s) in self._last_transcribed if i != id]
            return user_segments 

    #TODO: define the loop structure for the parallel processing best practices using this asr template 
    # for the user: push the buffer and wait for it to be empty (with timeout to avoid deadlocks) 
    def realtime_parallel_asr_loop(self):
        """
        realtime parallel asr loop
        this function must be called in a thread specific for the parallel processing
        the other threads receveind the audio streams must push the audio chunks in the buffer and wait for the results 
        """
        try:
            while True:
                self.transcribe_parallel() # this will block until the transcribe is done can take more than 1 second
        except KeyboardInterrupt:
            self._log.info("interrupted")
            return 
        except Exception as e: #TODO: restart the asr properly in case of error
            self._log.exception(e)
            return

#NOTE: asr for parallel use
class ParallelOnlineASRProcessor(OnlineASRProcessor):

    def __init__(self, asr, logger=logging.getLogger(__name__)):
        super().__init__(asr)
        self._logger = logger
        self._transcription_done = threading.Event()

    @property
    def buffer_time_seconds(self):
        return len(self.audio_buffer)/self.SAMPLING_RATE

    #TODO: remove huge logging 
    def parallel_process_iter(self):
        """
        Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        Parallel version, will be syncronized with the asr running conurrently, 
        checkout examples
        """
        #NOTE: checkout original process iter for understanding the process, i removed some comments for a more compact code 
        prompt, _ = self.prompt()
        self._logger.debug(f"transcribing {self.buffer_time_seconds:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        self._logger.debug(f"confirmed text: {self.transcript_buffer.commited_in_buffer}")

        self.asr.append_audio(id(self), self._transcription_done, self.audio_buffer, prompt) #WARNING:Blocking: wait for shared asr's buffer lock to push audio
        x = self.asr._last_transcript_time
        t = x + 1 if 0 < x < 1 else x # timeout min is 2 second, we balance the timeout with the last transcription time
        self._transcription_done.wait(timeout=t*2) #WARNING: Blocking: wait for transcription to complete, timeout to avoid rare deadlocks
        self._transcription_done.clear()

        res = self.asr.get_last_transcribed(id(self))
        tsw = self.asr.ts_words_normalized(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        self._logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        self._logger.debug(f"INCOMPLETE: {the_rest}")

        # Chunking here check original OnlineASRProcessor: chunck the buffer on on last committed work  
        k = len(self.commited)-1
        s = self.buffer_trimming_sec
        if self.buffer_time_seconds > s and k >= 0:
            l = self.buffer_time_offset + self.buffer_time_seconds - (s // 2)
            while k>0 and self.commited[k][1] > l:
                k -= 1
            t = self.commited[k][1] 
            self._logger.debug(f"chunking segment at word {self.commited[-1]} at {t}")
            self.chunk_at(t)

        self._logger.info(f"len of buffer now: {self.buffer_time_seconds:2.2f}")
        return self.to_flush(o)

