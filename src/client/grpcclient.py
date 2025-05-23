#!/usr/bin/env python3
import argparse
import time
import grpc
import sounddevice as sd
import librosa
import sys
import numpy as np

from src.generated import speech_pb2_grpc, speech_pb2


#WARNING: This code is just a test client for testing the multiclient realtime whisper server  

# Audio configuration
class AudioConfig:
    sample_rate = 16000  # Whisper wants 16 kHz
    chunk_duration = 1.0  # seconds
    channels = 1
    sample_format = np.int16  # 16-bit PCM
    max_abs_value = np.iinfo(sample_format).max  # 32767

    @classmethod
    def chunk_size(cls):
        return int(cls.chunk_duration * cls.sample_rate)

    @classmethod
    def chunk_samples(cls):
        "same as chunk_size()"
        return int(cls.chunk_duration * cls.sample_rate)

    @classmethod
    def chunk_bytes(cls):
        return cls.chunk_samples() * np.dtype(cls.sample_format).itemsize

# gRPC client for transcription
class TranscriptorClient:
    def __init__(self, host: str, port: int, with_hypothesis: bool = False, simulate_filepath: str = None, interactive: bool = False):
        self.host = host
        self.port = port
        self.with_hypothesis = with_hypothesis
        self.simulate_filepath = simulate_filepath
        self.interactive = interactive

    def __generate_audio_chunks_sim(self):
        # Simulation mode: load audio file (using librosa)
        audio_data, sr = librosa.load(self.simulate_filepath, sr=AudioConfig.sample_rate, mono=True)
        total_samples = len(audio_data)
        print(f"Loaded {total_samples} samples from file {self.simulate_filepath}")
        # Split into 1-second chunks
        for i in range(0, total_samples, AudioConfig.chunk_size()):
            chunk = audio_data[i:i+AudioConfig.chunk_size()]
            #if len(chunk) < AudioConfig.chunk_size() and len(audio_data) - i+AudioConfig.chunk_size():
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
                            blocksize=AudioConfig.chunk_size()) as stream:
            while True:
                # Read a chunk from the microphone (returns a tuple (data, overflow_flag))
                audio_data, _ = stream.read(AudioConfig.chunk_size())
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

    #TODO: refactor to reduce code duplication
    def run(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server.
        In interactive mode, the transcript is updated on a single line.
        """
        if self.with_hypothesis:
            self.run_with_hypothesis()
        else:
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

    def print_fullscreen(self, confirmed, hypothesis):
        # ANSI escape: clear screen + move cursor top-left
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        sys.stdout.write(confirmed)
        if hypothesis:
            sys.stdout.write(f"\033[90m{hypothesis}\033[0m")  # ipotesi in grigio
        sys.stdout.flush()

    def run_with_hypothesis(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server. This method also receive and handles the hypothesis of the transcription.
        """
        # Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = speech_pb2_grpc.SpeechToTextWithHypothesisStub(channel)

        # Audio chunk generator
        audio_generator = self.generate_audio_chunks()

        # Start the bidirectional call
        responses = stub.StreamingRecognize(audio_generator)
        try:
            if self.interactive:
                confirmed_text = ""
                last_confirmed = ""
                last_hypothesis = ""

                for response in responses:
                    updated = False

                    if response.confirmed.text and response.confirmed.text != last_confirmed:
                        confirmed_text += response.confirmed.text
                        last_confirmed = response.confirmed.text
                        updated = True
                        last_hypothesis = ""

                    if response.hypothesis.text != last_hypothesis:
                        last_hypothesis = response.hypothesis.text
                        updated = True

                    if updated:
                        self.print_fullscreen(confirmed_text, last_hypothesis)
            else:
                for response in responses:
                    confirmed = response.confirmed
                    hypothesis = response.hypothesis
                    print(f"Confirmed: {confirmed.start_time_millis} {confirmed.end_time_millis} \033[92m{confirmed.text}\033[00m")
                    print(f"Hypothesis: {hypothesis.start_time_millis} {hypothesis.end_time_millis} \033[91m{hypothesis.text}\033[00m\n")

        except grpc.RpcError as e:
            print("gRPC Error:", e)
        finally:
            channel.close()


def main():
    parser = argparse.ArgumentParser(description="gRPC Transcriptor Client")
    parser.add_argument('--host', type=str, default='localhost', help='gRPC server address')
    parser.add_argument('--port', type=int, default=50051, help='gRPC server port')
    parser.add_argument('--with-hypothesis', action='store_true',)
    parser.add_argument('--simulate', type=str, default=None,
                        help='Simulation mode: Path to the audio file to simulate a realtime audio stream with')
    parser.add_argument('--interactive', action='store_true',
                        help='Display transcript updates interactively on a single line')
    parser.add_argument('--chunk-duration', type=float, default=1.0)
    args = parser.parse_args()
    AudioConfig.chunk_duration = args.chunk_duration

    client = TranscriptorClient(host=args.host, port=args.port, with_hypothesis=args.with_hypothesis, simulate_filepath=args.simulate, interactive=args.interactive)
    client.run()

