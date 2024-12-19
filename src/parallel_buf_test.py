from parallel_whisper_online import MultiProcessingFasterWhisperASR
import librosa, numpy as np, logging, threading
import time

from whisper_online import FasterWhisperASR

print("test parallel whisper online")
logger = logging.getLogger().addHandler(logging.StreamHandler())
asr = MultiProcessingFasterWhisperASR("it", modelsize="large-v3-turbo", logger=logger)
print("asr initialized")
asr.warmup("../resources/sample1.wav")
print("asr warmed up")
audio_data1, file_sample_rate = librosa.load("../resources/sample1.wav", sr=16000, mono=True)
audio_data2, file_sample_rate = librosa.load("../resources/sample2.mp3", sr=16000, mono=True)
#NOTE: only for testing this will crash in exception 
print("audio loaded")

second = 16000
duration_s = 5
start = 0
end = second * duration_s 
while True:        
    words1, words2 = "", ""
    chunk1 = audio_data1[start:end].astype(np.float32)
    chunk2 = audio_data2[start:end].astype(np.float32)
    if audio_data1.size < end or audio_data2.size < end:
        break

    event = threading.Event()
    asr.append_audio(1, event, chunk1, init_prompt="")
    asr.append_audio(2, event, chunk2, init_prompt="")
    asr.append_audio(3, event, chunk1, init_prompt="")
    asr.append_audio(4, event, chunk2, init_prompt="")

    timestamp = time.time()  
    asr.transcribe_parallel()
    transcript_time = time.time() - timestamp

    event.wait()
    segments1 = asr.get_last_transcribed(1) 
    segments2 = asr.get_last_transcribed(2) 
    segments3 = asr.get_last_transcribed(3) 
    segments4 = asr.get_last_transcribed(4) 
    for segment in segments1:
        words1 += segment.text 
        [print(s.text) for s in segments1]
    for segment in segments2:
        words2 += segment.text
        [print(s.text) for s in segments2]


    print("parallel transcript time ", transcript_time)
                                      
    print("transcribed sleep 2 seconds\n")
    start = end
    end += second * duration_s 
    time.sleep(2)
    



