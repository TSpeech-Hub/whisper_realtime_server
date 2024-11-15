from whisper_online import *
import sounddevice as sd
import numpy as np

src_lan = "it"  # source language
tgt_lan = "it"  # target language  -- same as source for ASR, "en" if translate task is used
sample_rate = 16000
chunk_duration = 1
chunk_size = int(sample_rate * chunk_duration)

asr = FasterWhisperASR(src_lan, "large-v2")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
# sr.use_vad()  # set using VAD

online = OnlineASRProcessor(asr)  # create processing object with default buffer trimming option

result = "" 

def start_microphone_stream():
    print("Mic acquisition...")
    online.init() 
    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, blocksize=chunk_size):
            print("Real time streaming... Press Ctrl+C to interrupt.")
            while True:
                pass  # Mantain audio flux active
    except KeyboardInterrupt:
        print("Stop acquisition.")
    finally:
        final_output = online.finish()  # get final output
        if final_output:
            print("Stopping.")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio error: {status}", flush=True)
    audio_chunk = indata[:, 0]  # Use only one channel if stereo 
    online.insert_audio_chunk(audio_chunk) 
    output = online.process_iter() 
    if output[2] != '':
        print(output[2])  

if __name__ == "__main__":
    start_microphone_stream()

online.init()  # refresh if you're going to re-use the object for the next audio
