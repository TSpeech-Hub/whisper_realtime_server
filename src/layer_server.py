#!/usr/bin/env python3
import logging, os, threading, socket
from OpenSSL import SSL 
from whisper_online import *
from whisper_online_server import * 
from parallel_whisper_online import MultiProcessingFasterWhisperASR, ParallelOnlineASRProcessor, ParallelServerProcessor
from datetime import datetime
from argparse import Namespace

import json  
from argparse import Namespace
from datetime import datetime
"""
The functionality of both dictionaries and defaultdict is almost the same except for the fact that defaultdict never raises a KeyError. 
It provides a default value for the key that does not exist.
"""


#NOTE: LOGGING SETUP FUNCTION

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


#NOTE: CONNECTION CLASS DEFINITION

class Server(): 

    #NOTE: class variables
    modifiable = threading.Lock() 
    
    def __init__(self, host, port, logger): 
        self._host = host
        self._port = port
        self._logger = logger

    def _setup_ssl_context(self):
        """Create an SSL context using pyOpenSSL."""
        context = SSL.Context(SSL.TLSv1_2_METHOD)
        context.use_privatekey_file("key.pem")
        context.use_certificate_file("cert.pem")
        return context

#TODO: rename to parallel whisper server, refactoring code , configuration
class WhisperServer(Server):  
    __CONFIG_FILE = "config.json"

    #NOTE: may be changed in future every subclass
    def _setup_ssl_context(self):
        return super()._setup_ssl_context()

    def __init__(self, port, host, asr):    
        with open(self.__CONFIG_FILE) as file:
            config_dict = json.load(file)  
        super().__init__(host, port, setup_logging(f"WhisperServer-{port}"))

        try:
            self._client_ip = None
            self._is_available = True
            self.__ssl_context = self._setup_ssl_context()
            self.__config = Namespace(**config_dict)  # This converts dictionary to Namespace 

            #TODO: important : must use sentenct and tokenizer for parallel to work, you must know which language is our speaker speaking
            self.__online = ParallelOnlineASRProcessor(asr, None, buffer_trimming=("segment", 15), logfile=self._logger)
            self.__online.set_logger(self._logger)
        except Exception as e:
            msg = f"Error during ASR initialization {e} check the config file config.json"
            self._logger.error(msg)


    #NOTE: getters 
    @property 
    def port(self):
        return self._port

    @property
    def client_ip(self):
        return self._client_ip

    @property
    def available(self):
        return self._is_available

    #NOTE: setters  
    @client_ip.setter
    def client_ip(self, ip):
        with Server.modifiable: self._client_ip = ip
 
    #NOTE: helper function for start: socket creation and binding
    def __socket(self):
        try: 
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._logger.info(f'Starting server {self._host} {self._port}')
            s.bind((self._host, self._port))
            s.listen(1)
            s.settimeout(30) 
            self._logger.info(f'Server started {self._host} {self._port}')
            return s
        except Exception as e:
            self._logger.error(f"Error during server binding: {e}")
            raise type(e)(f"Error during server binding: {e} check the config file config.json")

    #WARNING: blocking function. server loop
    #NOTE: startup code from whisper_online_server.py original code repo
    def start_server_loop(self): 
        s = self.__socket()
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # just for starting the server
        while True:
            try:
                conn, addr = s.accept()
                conn = SSL.Connection(self.__ssl_context ,conn)
                conn.set_accept_state()
                conn.do_handshake()
                #TODO: ban ip for DOS
                if addr[0] != self.client_ip:
                    self._logger.error(f"Connection from unknown client {addr}")
                    conn.sendall(b"denied")
                    conn.close()
                    raise BrokenPipeError("Connection from unknown client") 

                self._is_available = False
                self._logger.info(f'Connected to client on {addr}')

                #NOTE: original code from whisper_online_server.py
                whisper_connection = Connection(conn)
                proc = ParallelServerProcessor(whisper_connection, self.__online, self.__config.min_chunk_size, self._logger)
                proc.parallel_process()

            except BrokenPipeError as e:
                self._logger.error(f"connection timeout {e}")
            except (SSL.Error, socket.error, ConnectionError) as e:
                self._logger.error(f"connection error: {e}")
            except Exception as e: 
                self._logger.error(f"something went wrong: {e}")
                raise e
            finally: 
                self._is_available = True
                self._logger.info(f'Connection to client closed')
                conn.close()
                

#NOTE: LAYERSERVER CLASS DEFINITION 

#TODO: Fix The terminate called without an active exception and then abort core dump issue with delayed whisper servers crash
class LayerServer(Server):

    __CONFIG_FILE = "config.json"

    #TODO: load balancing between model instances

    def create_asr(self, processors = 1) :
        model = MultiProcessingFasterWhisperASR(lan="auto", logger=setup_logging("asr"), modelsize=self._whisper_config.model, workers=processors) 
        return model
    
    #NOTE: may be changed in future every subclass
    def _setup_ssl_context(self):
        return super()._setup_ssl_context()

    def __init__(self, host="0.0.0.0", port=8000, max_servers=2, logger=setup_logging("LayerServer", use_stdout=True), port_pool=range(8001, 8100)):
        """Initialize the LayerServer with host, port, and max client limit."""
        super().__init__(host, port, logger)
        self._max_servers = max_servers
        self.__context = self._setup_ssl_context()
        self.__servers = []
        self.__port_pool = port_pool
        self.__lock = threading.Lock()
        with open(LayerServer.__CONFIG_FILE) as file:
            config_dict = json.load(file)  
        # Add host and port to the config 
        self._whisper_config = Namespace(**config_dict)

    def __create_servers(self):
        """
        create all the WhisperServer instances
        and start them in separate threads.
        """

        #TODO: manage asr instance control      
        asr = self.create_asr()
        asr.warmup("../resources/sample2.mp3")
        #TODO: the asr can crash find a way to understand if that happened
        threading.Thread(target=asr.realtime_parallel_asr_loop, daemon=True).start()
        self._logger.info("started and warmed up the parallel asr")

        for _ in range(self._max_servers):
            free_port = self.find_free_port()
            server = WhisperServer(free_port, self._host, asr)
            threading.Thread(target=server.start_server_loop, daemon=True).start()
            self.__servers.append(server)
            self._logger.info(f"WhisperServer started on port {free_port}")

    def find_free_port(self):
        """Find a free port on the system."""
        for port in self.__port_pool:
            if port not in [server.port for server in self.__servers]:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("localhost", port))
                        return port
                    except:
                        pass
        raise Exception("No free port available.") 

    def handle_client(self, connection):
        """
        Handle the client connection. 
        Assign the client to a WhisperServer instance.
        and send the port number to the client.
        """
        assigned_port = None
        try:
            if self.__lock.acquire(timeout=5):
                for server in self.__servers:
                    #TODO: ip ban for DOS
                    if server.available:
                        assigned_port = server.port
                        server.client_ip = connection.getpeername()[0]
                        break
                self.__lock.release()
            
            if assigned_port is not None:
                self._logger.info(f"Assigning client to WhisperServer on port {assigned_port}")
                connection.sendall(f"{assigned_port}".encode("utf-8"))
            else:
                self._logger.error("No available Whisper Servers found.")
                connection.sendall(b"no server available")

        except Exception as e:
            self._logger.error(f"Error handilng client: {e}")
            connection.sendall(b"internal server error")
        finally:
            connection.shutdown()
            connection.close()

    #WARNING: blocking helper function. server loop
    #NOTE: ssl proteted connection 
    def __loop(self, server_socket):
        """
        The main server loop.
        use SSL to protect the connection.
        Accept incoming connections handle them or refuse them.
        """
        while True: 
            try:
                connection, addr = server_socket.accept()
                connection = SSL.Connection(self.__context, connection)
                connection.set_accept_state()
                connection.do_handshake()

                #NOTE: checking server availability in enough if for concurrency reason another client in hanlder thread 
                #get past this check before the last one is assigned to a server, connection will be refused in hanlder thread 
                if True in [x.available for x in self.__servers]:
                    self._logger.info(f"Connection from {addr}")
                    threading.Thread(target=self.handle_client, args=(connection,), daemon=True).start() 
                else:
                    self._logger.error("No available Whisper Servers found.")
                    connection.sendall(b"no server available")
                    connection.shutdown()
                    connection.close()

            except Exception as e:
                self._logger.error(f"Error accepting connection: {e}")

    #WARNING: blocking function
    def start(self):
        """Start the LayerServer and listen for incoming connections."""
        self._logger.info("Starting LayerServer...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            #TODO: manage DOS (and DDOS) attacks 
            server_socket.bind((self._host, self._port)) 
            server_socket.listen(2) # NOTE: listen for 2 clients at a time

            self._logger.info(f"LayerServer listening on {self._host}:{self._port}") 
            self.__create_servers() #NOTE: pre-create the WhisperServer instances
            
            self.__loop(server_socket)
 
        except KeyboardInterrupt:
            self._logger.info("Stopping LayerServer...")
            raise KeyboardInterrupt("LayerServer stopped by user.")
        finally: #NOTE: ensure socket closure before raising exceptions
            server_socket.close()

#NOTE: MAIN FUNCTION WHEN THE SCRIPT IS RUN

if __name__ == "__main__":
    layer_server = LayerServer(host="0.0.0.0", port=8000, max_servers=20, port_pool=range(8001, 8100))
    layer_server.start()


