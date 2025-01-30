from dataclasses import dataclass
from typing import Text
from OpenSSL import SSL
import sounddevice as sd
import numpy as np
import socket, threading, sys, time, librosa, argparse

#NOTE: just to test the server

@dataclass
class AudioConfig:
    sample_rate = 16000
    chunk_duration = 1
    channels = 1
    chunk_size = int(sample_rate * chunk_duration)

@dataclass
class NetConfig:
    host = 'localhost'
    port = 8000

def setup_ssl_context():
    """Create an SSL context for the client."""
    context = SSL.Context(SSL.TLSv1_2_METHOD)
    context.set_verify(SSL.VERIFY_NONE, lambda *args: True)  # Skip certificate verification for testing
    return context

class TranscriptorClient:
    def __init__(self):
        self.__time_lock = threading.Lock()
        self.timestamp = 0

    mean_time, n_responses = 0, 0
    @staticmethod 
    def __error(msg):
        sys.stderr.write(f"\033[91m" + msg + "\033[0m\n")

    @staticmethod 
    def __log(msg):
        sys.stdout.write(f"\033[96m" + msg + "\033[0m\n")


    def connect_to_server(self, host, port):
        """Connect to the main server and get the Whisper server port."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = SSL.Connection(setup_ssl_context(), s)
        try:                
            #NOTE: SSL connection handshake
            conn.connect((host, port))
            conn.set_connect_state()
            conn.do_handshake()

            print(f"Connected to main server at {host}:{port}")
            
            whisper_port = conn.recv(1024)
            print(f"Whisper server port: {whisper_port}")
            whisper_port = whisper_port.decode('utf-8')
            if whisper_port.isdigit():
                print(f"Whisper server ready on port: {whisper_port}")
                return int(whisper_port)
            else:
                print(f"Error during connection invalid whisper port: {whisper_port}")
        except KeyboardInterrupt:
            print("Stopping client...")
        except SSL.Error as ssl_error:
            print(f"SSL error: {ssl_error}")
            raise ssl_error
        except Exception as e:
            print(f"Error connecting to server: {e}")
            raise e
        finally:
            conn.shutdown()
            conn.close()

    #TODO: start and simulate are duplicated 
    def start(self, port):
        """Initialize the microphone stream and handle client-server communication."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = SSL.Connection(setup_ssl_context(), s)
        try:
            #NOTE: SSL connection handshake
            conn.connect((NetConfig.host, port))
            conn.set_connect_state()
            conn.do_handshake()
            TranscriptorClient.__log(f"Streaming to server at {NetConfig.host}:{port}")
            # Initialize microphone
            with sd.InputStream(callback=None, channels=1, samplerate=AudioConfig.sample_rate, blocksize=AudioConfig.chunk_size) as stream:
                # Connect to the server
                
                sender_thread = threading.Thread(target=self.__send_audio, args=(stream, conn), daemon=True)
                receiver_thread = threading.Thread(target=self.__receive_transcriptions, args=(conn,), daemon=True)
                
                sender_thread.start()
                receiver_thread.start()
            
                while sender_thread.is_alive() and receiver_thread.is_alive(): 
                    threading.Event().wait(0.1)

        except KeyboardInterrupt : 
            TranscriptorClient.__log("Stopping client...")
        except Exception as e:
            TranscriptorClient.__error(f"Error in start{e}")
        finally: 
            s.close()

    def simulate(self, filepath, port): 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = SSL.Connection(setup_ssl_context(), s)
        try:
            #NOTE: SSL connection handshake
            conn.connect((NetConfig.host, port))
            conn.set_connect_state()
            conn.do_handshake()
            TranscriptorClient.__log(f"Streaming to server at {NetConfig.host}:{port}")
            # Initialize microphone
                # Connect to the server
                
            sender_thread = threading.Thread(target=self.__send_audio_simulation, args=(filepath, conn), daemon=True)
            receiver_thread = threading.Thread(target=self.__receive_transcriptions, args=(conn,), daemon=True)
            
            sender_thread.start()
            receiver_thread.start()
        
            while sender_thread.is_alive() and receiver_thread.is_alive(): 
                threading.Event().wait(0.1)

        except KeyboardInterrupt : 
            TranscriptorClient.__log("Stopping client...")
        except Exception as e:
            TranscriptorClient.__error(f"Error {e}")
        finally: 
            s.close()


    def __send_audio_simulation(self, filepath, sock):
        try: 
            audio_data, file_sample_rate = librosa.load(filepath, sr=16000, mono=True)

            num_samples = len(audio_data) 

            for start in range(0, num_samples, 16000):
                end = start + 16000
                chunk = audio_data[start:end]

                audio_chunk = (chunk * 32768).astype(np.int16).tobytes()

                sock.sendall(audio_chunk)

                time.sleep(1)        
        except Exception as e:
            TranscriptorClient.__error(f"Error in sending audio: {e}")

    def __send_audio(self, stream, sock):
        """Capture audio from the microphone and send it to the server."""
        try:
            while True:
                # Capture audio chunk
                now = time.time()
                audio_data = stream.read(AudioConfig.chunk_size)[0]
                audio_chunk = (audio_data[:, 0] * 32768).astype(np.int16).tobytes()

                with self.__time_lock:
                    if self.timestamp == 0:
                        self.timestamp = time.time() - now

                
                # Send audio chunk to server
                sock.sendall(audio_chunk)
        except Exception as e:
            TranscriptorClient.__error(f"Error in sending audio: {e}")

    def __receive_transcriptions(self, sock):
        """Receive transcriptions from the server."""
        
        print("DEBUG: receiving")
        count, sum = 0, 0
        try:
            while True:
                now = time.time()
                response = sock.recv(1024).decode('utf-8')
                splitted = response.split(" ")

                print("DEBUG: received")
                if response:
                    with self.__time_lock:
                        timestamp = round(time.time() - now, 2)
                        self.timestamp = 0
                    sum += timestamp
                    count += 1
                    #NOTE: staying silent for a while increase the time between chunks, this is just for debugging
                    print(f"time since last audio chunk was sent: {timestamp} seconds")
                    print(f"{response}")
        except Exception as e:
            TranscriptorClient.__error(f"Error in receiving transcription: {e}")
        finally:
            if count == 0:
                TranscriptorClient.__error("No transcriptions received")
            else:
                TranscriptorClient.__error(f"Average time: {sum/count} milliseconds")


args = argparse.ArgumentParser("optional file path for simulation")
args.add_argument('--filepath', type=Text,
                    help='audio file path')

if __name__ == "__main__":
    client = TranscriptorClient()
    port = client.connect_to_server(NetConfig.host, NetConfig.port)

    filepath = args.parse_args().filepath

    if filepath: 
        print("DEBUG: simulation")
        client.simulate(filepath, port)
    else: 
        print("DEBUG: real")
        client.start(port)
