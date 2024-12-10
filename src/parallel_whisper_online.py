import sys, numpy as np, logging, os
from whisper_online import FasterWhisperASR, HypothesisBuffer, logger, load_audio_chunk

class MultiProcessingFasterWhisperASR(FasterWhisperASR):

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, workers=1):
        self.workers = workers 
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile=logfile)

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

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, batch_size=64, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        #TODO: check if util
        #print(info)  # info contains language detection result

        return list(segments)

#TODO: make the real parallel online processor this is just a copy of the online processor
