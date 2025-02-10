#!/usr/bin/env python3
import argparse, time, grpc, sounddevice as sd, librosa
import speech_pb2, speech_pb2_grpc 

# Audio configuration
class AudioConfig:
    sample_rate = 16000 
    chunk_duration = 1
    channels = 1
    chunk_size = int(sample_rate * chunk_duration)  

# gRPC client for transcription
class TranscriptorClient:
    def __init__(self, host: str, port: int, simulate_filepath: str = None):
        self.host = host
        self.port = port
        self.simulate_filepath = simulate_filepath

    def generate_audio_chunks(self):
        """
        Iteratively generates 1-second audio chunks.
        If simulate_filepath is set, the client loads the audio from a file;
        otherwise, it captures live audio from the microphone.
        Each chunk is sent as an AudioChunk message.
        """
        if self.simulate_filepath:
            # Simulation mode: load audio file (using librosa)
            audio_data, sr = librosa.load(self.simulate_filepath, sr=AudioConfig.sample_rate, mono=True)
            total_samples = len(audio_data)
            print(f"Loaded {total_samples} samples from file {self.simulate_filepath}")
            # Split into 1-second chunks
            for i in range(0, total_samples, AudioConfig.chunk_size):
                chunk = audio_data[i:i+AudioConfig.chunk_size]
                if len(chunk) < AudioConfig.chunk_size:
                    # If the last chunk is incomplete, you can choose to skip it
                    break
                # Create and return the AudioChunk message
                yield speech_pb2.AudioChunk(samples=chunk.tolist())
                # Simulate real-time sending
                time.sleep(AudioConfig.chunk_duration)
        else:
            # Live mode: use the microphone
            print("Capturing real-time audio from the microphone...")
            with sd.InputStream(channels=AudioConfig.channels,
                                samplerate=AudioConfig.sample_rate,
                                blocksize=AudioConfig.chunk_size) as stream:
                while True:
                    # Read a chunk from the microphone (returns a tuple (data, overflow_flag))
                    audio_data, _ = stream.read(AudioConfig.chunk_size)
                    # audio_data is a numpy array of shape (chunk_size, channels)
                    # Assuming mono: flatten the array
                    chunk_samples = audio_data.flatten()
                    yield speech_pb2.AudioChunk(samples=chunk_samples.tolist())
                    time.sleep(AudioConfig.chunk_duration)

    def run(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server.
        """
        # Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = speech_pb2_grpc.SpeechToTextStub(channel)

        # Audio chunk generator
        audio_generator = self.generate_audio_chunks()

        # Start the bidirectional call
        responses = stub.StreamingRecognize(audio_generator)
        try:
            for response in responses:
                print("Received transcription:", response.text)
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
    args = parser.parse_args()

    client = TranscriptorClient(host=args.host, port=args.port, simulate_filepath=args.simulate)
    client.run()

if __name__ == '__main__':
    main()
