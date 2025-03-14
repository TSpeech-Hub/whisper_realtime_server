#!/usr/bin/env python3
import argparse
import time
import grpc
import sounddevice as sd
import librosa

from src.generated import speech_pb2_grpc, speech_pb2

# Audio configuration
class AudioConfig:
    sample_rate = 16000 
    chunk_duration = 1 
    channels = 1
    chunk_size = int(sample_rate * chunk_duration)  

# gRPC client for transcription
class TranscriptorClient:
    def __init__(self, host: str, port: int, simulate_filepath: str = None, interactive: bool = False):
        self.host = host
        self.port = port
        self.simulate_filepath = simulate_filepath
        self.interactive = interactive

    def __generate_audio_chunks_sim(self):
        # Simulation mode: load audio file (using librosa)
        audio_data, sr = librosa.load(self.simulate_filepath, sr=AudioConfig.sample_rate, mono=True)
        total_samples = len(audio_data)
        print(f"Loaded {total_samples} samples from file {self.simulate_filepath}")
        # Split into 1-second chunks
        for i in range(0, total_samples, AudioConfig.chunk_size):
            chunk = audio_data[i:i+AudioConfig.chunk_size]
            #if len(chunk) < AudioConfig.chunk_size and len(audio_data) - i+AudioConfig.chunk_size:
                # Skip incomplete last chunk
                #break
            yield speech_pb2.AudioChunk(samples=chunk.tolist())
            # Simulate real-time sending
            time.sleep(AudioConfig.chunk_duration)

    def __generate_audio_chunks_live(self):
        # Live mode: use the microphone
        print("Capturing real-time audio from the microphone...")
        with sd.InputStream(channels=AudioConfig.channels,
                            samplerate=AudioConfig.sample_rate,
                            blocksize=AudioConfig.chunk_size) as stream:
            while True:
                # Read a chunk from the microphone (returns a tuple (data, overflow_flag))
                audio_data, _ = stream.read(AudioConfig.chunk_size)
                # Flatten the array (assuming mono audio)
                chunk_samples = audio_data.flatten()
                yield speech_pb2.AudioChunk(samples=chunk_samples.tolist())
                time.sleep(AudioConfig.chunk_duration)

    def generate_audio_chunks(self):
        """
        Iteratively generates 1-second audio chunks.
        If simulate_filepath is set, the client loads the audio from a file;
        otherwise, it captures live audio from the microphone.
        Each chunk is sent as an AudioChunk message.
        """
        print("Started connection")
        if self.simulate_filepath:
            return self.__generate_audio_chunks_sim()
        else:
            return self.__generate_audio_chunks_live()

    def run(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server.
        In interactive mode, the transcript is updated on a single line.
        """
        # Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = speech_pb2_grpc.SpeechToTextStub(channel)

        # Audio chunk generator
        audio_generator = self.generate_audio_chunks()

        # Start the bidirectional call
        responses = stub.StreamingRecognize(audio_generator)
        try:
            last_resp_time = 0
            for response in responses:
                if self.interactive:
                    # Update the transcript on the same line. 
                # Clear the line and write the updated transcript
                    if response.text[-1] == "." and  response.start_time_millis - last_resp_time > 1000:
                        print(response.text)
                    else:
                        print(response.text, end="", flush=True)
                    last_resp_time = response.end_time_millis
                else:
                    print(f"Received transcription: {response.start_time_millis} {response.end_time_millis} {response.text}")
        except grpc.RpcError as e:
            print("gRPC Error:", e)
        finally:
            channel.close()

def main():
    parser = argparse.ArgumentParser(description="gRPC Transcriptor Client")
    parser.add_argument('--host', type=str, default='localhost', help='gRPC server address')
    parser.add_argument('--port', type=int, default=50051, help='gRPC server port')
    parser.add_argument('--simulate', type=str, default=None,
                        help='Path to the audio file to use in simulation mode')
    parser.add_argument('--interactive', action='store_true',
                        help='Display transcript updates interactively on a single line')
    args = parser.parse_args()

    client = TranscriptorClient(host=args.host, port=args.port, simulate_filepath=args.simulate, interactive=args.interactive)
    client.run()
