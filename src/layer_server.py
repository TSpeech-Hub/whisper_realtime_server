#!/usr/bin/env python3
import logging, os, threading, socket
from whisper_online import *
from whisper_online_server import *
from datetime import datetime
from argparse import Namespace

#NOTE: Connection args definition for Whisper Streaming factory
#TODO: Do this with a json given to the LayerServer class
class WhisperStreamingConfig:
    @staticmethod
    def get_default():
        return Namespace(
            warmup_file="../resources/sample1.wav",
            host="localhost",
            port=43007,
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

#NOTE: Logging Setup
def setup_logging(log_name, use_stdout=False, log_folder="server_logs"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_folder, exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_name}.log"
    log_path = os.path.join(log_folder, log_filename)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if use_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


#NOTE: Connection class definition
class WhisperServer:

    def __init__(self, host, port, ready_event, logger):
        self.port = port
        self.host = host
        self.samplig_rate = 16000
        self.ready_event = ready_event
        self.logger = logger
        self.args = WhisperStreamingConfig.get_default() #NOTE: Args for the Whisper factory

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

    #WARNING: blocking function
    def start(self): 
        self.logger.info("Starting WhisperServer...")
        _, online = self.__warmup()
        self.logger.info(f"{online} online, {self.args.model} model")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            self.logger.info(f"Listening on {self.host}:{self.port}")

            # Signal that the server is ready to accept connections
            self.ready_event.set()

            try:
                s.settimeout(30)  # 30 seconds timeout to avoid blocking resources
                conn, addr = s.accept()
                self.logger.info(f"Connected to client on {addr}")
                connection = Connection(conn)
                proc = ServerProcessor(connection, online, self.args.min_chunk_size, self.logger)
                proc.process()
                conn.close()
                self.logger.info("Connection to client closed")
            except socket.timeout:
                self.logger.info("Timeout reached. Closing server.")

#NOTE: LayerServer class definition
#TODO: Fix The terminate called without an active exception and then abort core dump issue with delayed whisper servers crash
class LayerServer:
    def __init__(self, host="0.0.0.0", port=8000, max_clients=2, logger=setup_logging("LayerServer", use_stdout=True)):
        """Initialize the LayerServer with host, port, and max client limit."""
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.logger = logger
        self.active_clients = 0
        self.lock = threading.Lock()

    @staticmethod
    def find_free_port():
        """Find a free port on the system."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    #TODO: refactoring
    def handle_client(self, client_socket):
        #NOTE: This function is called in a separate thread for each client to reduce code duplication
        def closing():
            client_socket.close()
            with self.lock:
                self.logger.info("A whisper instance just closed.")
                self.active_clients -= 1
                self.logger.info(f"current active clients: {self.active_clients}")

        #NOTE: if set to true that means the client handler already incremented connected client number in case of exception, decrement the number of active clients
        increment = False
        try:
            #NOTE: refused too many clients
            if self.active_clients >= self.max_clients:
                client_socket.sendall(b"too many clients")
                self.logger.warning("Connection refused: too many clients.")
                return

            with self.lock:
                self.active_clients += 1
                self.logger.info(f"new active client: {self.active_clients}")

            increment = True # Flag to indicate that num of active clients should be decremented in case of exception 
            port = self.find_free_port() # Find a free port and start Whisper server
            client_socket.sendall(b"ok")  # Indicate server allocation success

            # Create a threading event to signal when the Whisper server is ready
            ready_event = threading.Event()
            logger = setup_logging(f"WhisperServer_{port}")
            whisper = WhisperServer("localhost", port, ready_event, logger)

            whisper_thread = threading.Thread(target=whisper.start, daemon=True)
            whisper_thread.start()  

            # Wait until the Whisper server signals readiness 30 sec 
            ready_event.wait(timeout=30)
            if ready_event.is_set():
                client_socket.sendall(f"{port}".encode("utf-8"))  # Send port to client
                whisper_thread.join()  # Wait for Whisper server to finish           
            else:
                client_socket.sendall(b"server not ready")

            closing()

        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            client_socket.sendall(b"internal server error")
            if increment: closing()


    def start(self):
        logger.info("Starting LayerServer...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            try:
                server_socket.bind((self.host, self.port))  
                server_socket.listen(5)
                self.logger.info(f"LayerServer listening on {self.host}:{self.port}")
                
                while True:
                    client_socket, addr = server_socket.accept()
                    self.logger.info(f"Connection from {addr}")
                    threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                self.logger.error(f"Error: {e}")
            except KeyboardInterrupt:
                self.logger.info("Stopping LayerServer...")

if __name__ == "__main__":
    layer_server = LayerServer(host="0.0.0.0", port=8000, max_clients=2)
    layer_server.start()


