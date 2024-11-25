#!/usr/bin/env python3
import time, logging, os, threading, socket
from whisper_online import *
from whisper_online_server import *
from datetime import datetime
from argparse import Namespace

class WhisperServer:

    def __init__(self, host, port):
        self.port = port
        self.host = host
        self.samplig_rate = 16000

        # Set up logging
        self.log_folder = "server_logs"
        os.makedirs(self.log_folder, exist_ok=True)
        log_filename = f"whisper_{port}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.log_folder, log_filename)

        self.logger = logging.getLogger(f"WhisperServer-{port}")
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.info(f"Logging initialized for WhisperServer on port {port}. Log file: {log_path}")

        # Args for the Whisper factory
        self.args = Namespace(
            warmup_file="../resources/sample1.wav",
            host="localhost",
            port=int(port),
            model="large-v3-turbo",
            backend="faster-whisper",
            language="en",
            min_chunk_size=1.0,
            model_cache_dir=None,
            model_dir=None,
            lan="en",
            task="transcribe",
            vac=False,
            vac_chunk_size=0.04,
            vad=False,
            buffer_trimming="segment",
            buffer_trimming_sec=15,
            log_level="DEBUG"
        )

    def __warmup(self):
        self.logger.info("Starting warmup process...")
        asr, online = asr_factory(self.args)

        msg = "Whisper is not warmed up. The first chunk processing may take longer."
        if self.args.warmup_file:
            if os.path.isfile(self.args.warmup_file):
                self.logger.info(f"Warmup file found: {self.args.warmup_file}")
                a = load_audio_chunk(self.args.warmup_file, 0, 1)
                asr.transcribe(a)
                self.logger.info("Whisper is warmed up.")
                return asr, online
            else:
                self.logger.critical("The warm up file is not available. " + msg)
                raise FileNotFoundError("The warm up file is not available.")
        else:
            self.logger.warning(msg)
            return asr, online

    # WARNING: blocking function
    def start(self): 
        self.logger.info("Starting WhisperServer...")
        _, online = self.__warmup()
        self.logger.info(f"{online} online, {self.args.model} model")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            self.logger.info(f"Listening on {self.host}:{self.port}")

            # NOTE: wont let client spam connections and do not consume resources for a long time
            try:
                s.settimeout(30) # 30 seconds timeout to avoid blocking resources
                attempts = 0
                while True and attempts < 10:
                    conn, addr = s.accept()
                    attempts += 1
                    self.logger.info(f"Connected to client on {addr}")
                    connection = Connection(conn)
                    proc = ServerProcessor(connection, online, self.args.min_chunk_size, self.logger)
                    proc.process()
                    conn.close()
                    self.logger.info("Connection to client closed")
            except socket.timeout:
                self.logger.info("Timeout reached. Closing server.")

### LayerServer class definition

class LayerServer:
    def __init__(self, host="0.0.0.0", port=8000, max_clients=2):
        """Initialize the LayerServer with host, port, and max client limit."""
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.active_clients = 0
        self.lock = threading.Lock()

    @staticmethod
    def find_free_port():
        """Find a free port on the system."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def handle_client(self, client_socket):
        try:
            with self.lock:
                if self.active_clients >= self.max_clients:
                    client_socket.sendall(b"too many clients")
                    print("Connection refused: too many clients.")
                    return

                self.active_clients += 1

            # Find a free port and start Whisper server
            port = self.find_free_port()
            client_socket.sendall(b"ok")  # Indicate server allocation

            whisper = WhisperServer("localhost", port)  # Assuming WhisperServer is defined
            whisper_thread = threading.Thread(target=whisper.start, daemon=True)
            whisper_thread.start()  

            # TODO: Replace this with proper readiness checking
            time.sleep(6)
            client_socket.sendall(f"{port}".encode("utf-8"))  # Send port to client
            whisper_thread.join()  # Wait for Whisper server to finish

        except Exception as e:
            print(f"Error handling client: {e}")
            client_socket.sendall(b"internal server error")
        finally:
            client_socket.close()
            with self.lock:
                print("A whisper instance just closed.")
                self.active_clients -= 1

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"LayerServer listening on {self.host}:{self.port}")

            while True:
                client_socket, addr = server_socket.accept()
                print(f"Connection from {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

if __name__ == "__main__":
    print("Starting LayerServer...")
    layer_server = LayerServer(host="0.0.0.0", port=8000, max_clients=2)
    layer_server.start()
