from parallel_whisper_online import MultiProcessingFasterWhisperASR
import librosa, numpy as np 
import time

print("test parallel whisper online")
asr = MultiProcessingFasterWhisperASR(None, modelsize="large-v3-turbo")
print("asr initialized")
asr.warmup("../resources/sample1.wav")
print("asr warmed up")
audio_data1, file_sample_rate = librosa.load("../resources/sample1.wav", sr=16000, mono=True)
audio_data2, file_sample_rate = librosa.load("../resources/sample2.mp3", sr=16000, mono=True)
#NOTE: only for testing this will crash in exception 
end = 0
start = 0
print("audio loaded")
words = ""
try:
    i = 1
    while True:        
        if i == 10:
            start = end
            i = 1
        end += 16000  
        i += 1
        chunk1 = audio_data1[start:end].astype(np.float32)
        chunk2 = audio_data2[start:end].astype(np.float32)
        if audio_data1.size < end or audio_data2.size < end:
            break

        asr.append_audio(chunk1, words)
        asr.append_audio(chunk2, words)

        timestamp = time.time()  
        asr.transcribe_parallel()
        transcript_time = time.time() - timestamp
        segments = asr.get_last_transcribed() 

        print("parallel transcribed:")
        for segment in segments:
            print(segment.text)
            words += segment.text  + " "
        print(f"transcript time for this chunks parallel: {transcript_time}")
        print("\n")

        print("sequential transcribed:")
        timestamp = time.time()
        for w in asr.transcribe(chunk1, words):
            print(w.text)
        for w in asr.transcribe(chunk2, words):
            print(w.text)
        transcript_time = time.time() - timestamp
        print(f"transcript time for sequential: {transcript_time}")
        print("\n")
                                          
        print("transcribed sleep 2 seconds\n")
        time.sleep(2)
        


except Exception as e:
    print(f"execution interrupted by esception error")
    raise e

