import sys, numpy as np, logging, os, threading, time, copy
from types import SimpleNamespace
from whisper_online import FasterWhisperASR, OnlineASRProcessor, logger, load_audio_chunk
from whisper_online_server import ServerProcessor

#TODO: WRITE DOC: all these classes should be used toghether with parallel threads, explain best practices and why i used the already implemented classes 
# (example with this correcy inheritance batched inferecence can be used in sequential with old online asr sequential class) 

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
        self._audio = np.append(self._audio, np.zeros(8000, dtype=np.float32)) #small silence to mark segments 

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
        self._push_lock = threading.Lock() #lock for the shared buffer
        self._get_lock = threading.Lock() #lock for the shared results

        self.last_transcribed = [] 
        self._buffer = ParallelAudioBuffer()
        self._client_events = []

        self._log = logger
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile=logfile)# cant use vad if segment times self.use_vad() 

    #also we use segment timing, if the segment time start is not zero it mean we have to normalize the words time by segment start 
    def ts_words_normalized(self, segments):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for segment in segments:
            segment_start = segment.start #normalization times if sharing the buffer segments start wont be zero it will be shifted 
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                t = (word.start - segment_start, word.end - segment_start, word.word)
                o.append(t)
        return o

    #WARNING: buffer can be a shared resource use with locks or sync
    def __reset_buffer(self):
        self._client_events = []
        self._buffer.reset()

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

    #TODO: buffer refactoring with queue?
    def append_audio(self, id, event, audio, init_prompt=""): #NOTE: for shared processing only
        """make use of the shared buffer"""
        with self._push_lock:
            self._client_events.append(event)
            self._buffer.append_token(id, audio, init_prompt)

    def transcribe_parallel(self):
        """
        inference on the shared buffer, can be shared between parallel online processors
        online processors can get results with get_last_transcribe method if pushed audio before
        cant select a specific language, will have transcribe task by default, NOTE: lang will be global
        """
        
        if self._buffer.is_empty(): #NOTE: not processing if empty  
            return 0

        # parameters setup and reset 
        # this must be done before transcribing buffer may change during the transcribe process
        timestamp = time.time()
        with self._push_lock:
            buffer_args = self._buffer.parameters()
            events = self._client_events 
            self.__reset_buffer()

        #TODO: check if info util
        segments, _ = self.model.transcribe(buffer_args.audio, batch_size=16, multilingual=True, initial_prompt=buffer_args.init_prompts,beam_size=5, word_timestamps=True, condition_on_previous_text=True, clip_timestamps=buffer_args.segment_times)
        with self._get_lock:
            self.last_transcribed.extend(list(zip(buffer_args.ids, list(segments)))) # assoc, client ids to segments
            [e.set() for e in events] # set all client event to let them get the transcription 

        transcript_time = time.time() - timestamp 
        self._log.debug(f"transcription time: {transcript_time}")
        time.sleep(0.05) # help client syncronize...

        return transcript_time
    
    def get_last_transcribed(self, id):
        """
        return the last transcribed segments associated with id (in order), if any thread is using the parallel transcribe it will wait until the transcribe is done
        collateral: will remove (and return) all the segments corresponding to id  
        """
        with self._get_lock:
            self._log.debug(f"Requesting segments for id: {id}")
            user_segments = [s for (i, s) in self.last_transcribed if i == id]
            if user_segments:
                self._log.debug(f"Current last_transcribed: {[(i, s.text) for (i, s) in self.last_transcribed]}")
                self.last_transcribed = [(i, s) for (i, s) in self.last_transcribed if i != id]
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
        except Exception as e:
            self._log.error(e)
            raise e 

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
        self._logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        self._logger.debug(f"confirmed text: {self.transcript_buffer.commited_in_buffer}")

         #WARNING: wait for psuh lock to push audio
        self.asr.append_audio(id(self), self._transcription_done, self.audio_buffer, prompt)
        self._transcription_done.wait(timeout=2) 

        #WARNING: wait for transcription to complete
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

        # there is a newly confirmed text
        # removed sentence chunking here check original OnlineASRProcessor 
 
        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it

        # cunck the buffer on on last committed work 
        if self.buffer_time_seconds > s:
            l = self.buffer_time_offset + self.buffer_time_seconds - 10
            k = len(self.commited)-1
            while k>0 and self.commited[k][1] > l:
                k -= 1
            t = self.commited[k][1] 
            logger.debug("chunking segment")
            self.chunk_at(t)

        self._logger.info(f"len of buffer now: {self.buffer_time_seconds:2.2f}")
        return self.to_flush(o)

class ParallelServerProcessor(ServerProcessor):

    def parallel_process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break #TODO : tell client i have nothing
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.parallel_process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                self.logger.info("broken pipe -- connection closed?")
                break
        self.online_asr_proc.finish()

