import sys, numpy as np, logging, os, threading
from whisper_online import FasterWhisperASR, HypothesisBuffer, logger, load_audio_chunk

from faster_whisper.vad import VadOptions
# these are set to help detect the start of a different audio source in buffer
vad_parameters = VadOptions(
    min_silence_duration_ms=500,   # minimum duration of silence to detect the end of speech (default 500 ms) 
    speech_pad_ms=30, # padding to add to the end of speech 
    max_speech_duration_s=10 # maximum duration of speech to detect the end of speech 
)

class MultiProcessingFasterWhisperASR(FasterWhisperASR):

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, workers=1):
        self.workers = workers 
        self.push_lock = threading.Lock() #lock for the shared buffer
        self.last_transcribed = None
        self.buffer = np.array([],dtype=np.float32)
        self.init_prompts = [] #shared init_prompt
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile=logfile)
        self.use_vad() 
    def reset_buffer(self):
        with self.push_lock:
            self.buffer = np.array([],dtype=np.float32)
            self.init_prompts = [] #shared init_prompt

    def warmup(self, filepath, logger=logging.getLogger(__name__)): 
        """
        original repo warmup code,
        warm up the ASR because the very first transcribe takes more time than the others. 
        Test results in https://github.com/ufal/whisper_streaming/pull/81
        """
        msg = "Whisper is not warmed up. The first chunk processing may take longer."
        if filepath: 
            if os.path.isfile(filepath):
                a = load_audio_chunk(filepath,0,1)
                try:
                    self.transcribe(a)
                except Exception as e:
                    msg = f"Error during ASR initialization {e} check the config file config.json"
                    logger.error(msg)
                    raise type(e)(msg)
                logger.info("Whisper is warmed up.")
            else:
                logger.critical("The warm up file is not available. "+msg)
        else:
            logger.warning(msg)


    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
#        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", num_workers=self.workers, download_root=cache_dir)
        batched_model = BatchedInferencePipeline(model)
        return batched_model 

    #NOTE: for shared processing
    def append_audio(self, audio, init_prompt=""):
        """make use of the shared buffer"""
        if len(audio) == 0:
            return
        with self.push_lock:
            self.init_prompts.append(init_prompt)
            #TODO: property to get the silence 
            silence = np.zeros(16000 * 5, dtype=np.float32) # one second silence 
            self.buffer = np.append(self.buffer, audio)
            self.buffer = np.append(self.buffer, silence)

    def transcribe_parallel(self):
        """inference on the shared buffer with the audio chunks pushed in the buffer"""
        #NOTE: not shared processing
        if len(self.buffer) == 0:
            return
        init_prompts = ""
        audio = self.buffer
        for prompt in self.init_prompts:
            init_prompts += prompt + " "
        #NOTE: this must be done before transcribing buffer may change during the transcribe process
        self.reset_buffer()
        #TODO: batch size must return to 64 before production and True context
        segments, info = self.model.transcribe(audio, vad_parameters=vad_parameters, batch_size=8, language=self.original_language, initial_prompt=init_prompts, beam_size=10, word_timestamps=True, condition_on_previous_text=False, **self.transcribe_kargs)
        #TODO: check if util
        #print(info)  # info contains language detection result

        self.last_transcribed = list(segments)


    def get_last_transcribed(self):
        """
        return the last transcribed segments, if any thread is using the parallel transcribe it will wait until the transcribe is done
        """
        return self.last_transcribed

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
                if len(self.buffer) > 0:
                    self.transcribe_parallel() # this will block until the transcribe is dona can take more than 1 second
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")

    #NOTE: not shared processing 
    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        #TODO: batch size must return to 64 before production and True context
        segments, info = self.model.transcribe(audio, batch_size=8, language=self.original_language, initial_prompt=init_prompt, beam_size=10, word_timestamps=True, condition_on_previous_text=False, **self.transcribe_kargs)
        #TODO: check if util
        #print(info)  # info contains language detection result

        return list(segments)

#TODO: make the real parallel online processor this is just a copy of the online processor
