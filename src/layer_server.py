#!/usr/bin/env python3
import logging, os, threading, socket
from whisper_online import *
from whisper_online_server import * 
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

class WhisperServer: 
   
    CONFIG_FILE = "config.json"

    def __init__(self, port, host):
        with open(self.CONFIG_FILE) as file:
            config_dict = json.load(file)  
        config_dict["port"] = port
        config_dict["host"] = host 
        self.is_available = True
        self.logger = setup_logging(f"WhisperServer-{port}")
        self.client_ip = None
        try:
            self.config = Namespace(**config_dict)  # This converts dictionary to Namespace 
            self.asr, self.online = asr_factory(self.config)
        except Exception as e:
            msg = f"Error during ASR initialization {e} check the config file config.json"
            self.logger.error(msg)
            raise type(e)(msg)

    def warmup(self): 
        """
        original repo warmup code,
        warm up the ASR because the very first transcribe takes more time than the others. 
        Test results in https://github.com/ufal/whisper_streaming/pull/81
        """
        msg = "Whisper is not warmed up. The first chunk processing may take longer."
        if self.config.warmup_file:
            if os.path.isfile(self.config.warmup_file):
                a = load_audio_chunk(self.config.warmup_file,0,1)
                try:
                    self.asr.transcribe(a)
                except Exception as e:
                    msg = f"Error during ASR initialization {e} check the config file config.json"
                    self.logger.error(msg)
                    raise type(e)(msg)
                self.logger.info("Whisper is warmed up.")
            else:
                self.logger.critical("The warm up file is not available. "+msg)
                sys.exit(1)
        else:
            self.logger.warning(msg)

    #WARNING: blocking function. server loop
    #NOTE: startup code from whisper_online_server.py original code repo
    #TODO: Connect only on known clients iptables?
    def start_server_loop(self): 
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                self.logger.info(f'Starting server {self.config.host} {self.config.port}')
                s.bind((self.config.host, self.config.port))
                s.listen(1)
                self.logger.info(f'Server started {self.config.host} {self.config.port}')
                while True:
                    try:
                        conn, addr = s.accept()
                        #TODO: ban ip for DOS
                        if addr[0] != self.client_ip:
                            self.logger.error(f"Connection from unknown client {addr}")
                            conn.sendall(b"denied")
                            conn.close()
                            continue
                        self.is_available = False
                        self.logger.info(f'Connected to client on {addr}')
                        #NOTE: original code from whisper_online_server.py
                        connection = Connection(conn)
                        proc = ServerProcessor(connection, self.online, self.config.min_chunk_size, self.logger)
                        proc.process()
                        self.logger.info('Connection to client closed')
                        conn.close()
                    except (ConnectionAbortedError, ConnectionResetError) as e:
                        self.logger.error(f"Connection error: {e}")
                    finally: 
                        self.is_available = True
        except Exception as e:
            self.logger.error(f"Error during server binding: {e}")
            raise type(e)(f"Error during server binding: {e} check the config file config.json")
                

#NOTE: LAYERSERVER CLASS DEFINITION 

#TODO: Fix The terminate called without an active exception and then abort core dump issue with delayed whisper servers crash
class LayerServer:
    def __init__(self, host="0.0.0.0", port=8000, max_servers=2, logger=setup_logging("LayerServer", use_stdout=True), port_pool=range(8001, 8100)):
        """Initialize the LayerServer with host, port, and max client limit."""
        self.host = host
        self.port = port
        self.logger = logger
        self.max_servers = max_servers
        self.servers = []
        self.port_pool = port_pool
        self.lock = threading.Lock()

    def find_free_port(self):
        """Find a free port on the system."""
        for port in self.port_pool:
            if port not in [server.config.port for server in self.servers]:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("localhost", port))
                        return port
                    except:
                        pass
        raise Exception("No free port available.") #TODO: custom exception

    def create_servers(self):
        """
        create all the WhisperServer instances
        and start them in separate threads.
        """
        for _ in range(self.max_servers):
            free_port = self.find_free_port()
            server = WhisperServer(free_port, self.host)
            server.warmup() #NOTE: warmup the ASR
            threading.Thread(target=server.start_server_loop, daemon=True).start()
            self.servers.append(server)
            self.logger.info(f"WhisperServer started on port {free_port}")

    def handle_client(self, client_socket):
        """
        Handle the client connection. 
        Assign the client to a WhisperServer instance.
        and send the port number to the client.
        """
        assigned_port = None
        try:
            if self.lock.acquire(timeout=5):
                try:
                    for server in self.servers:
                        if server.is_available:
                            assigned_port = server.config.port
                            #TODO: ip ban for DOS
                            server.client_ip = client_socket.getpeername()[0]
                            break
                finally:
                    self.lock.release()
            
            if assigned_port is not None:
                self.logger.info(f"Assigning client to WhisperServer on port {assigned_port}")
                client_socket.sendall(f"{assigned_port}".encode("utf-8"))
            else:
                self.logger.error("No available Whisper Servers found.")
                client_socket.sendall(b"no server available")

        except Exception as e:
            self.logger.error(f"Error handilng client: {e}")
            client_socket.sendall(b"internal server error")
        finally:
            client_socket.close()

    #WARNING: blocking function. server loop
    def start(self):
        """Start the LayerServer and listen for incoming connections."""
        logger.info("Starting LayerServer...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            try:
                server_socket.bind((self.host, self.port)) 
                #TODO: manage DDOS change max clients
                server_socket.listen(2) # NOTE: listen for 2 clients at a time
                self.logger.info(f"LayerServer listening on {self.host}:{self.port}") 
                self.create_servers() #NOTE: pre-create the WhisperServer instances

                while True: 
                    try:
                        client_socket, addr = server_socket.accept()
                        self.logger.info(f"Connection from {addr}")
                        threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start() 
                    except (ConnectionAbortedError, ConnectionResetError) as e:
                        self.logger.error(f"Connection error with new client: {e}")
            except Exception as e:
                self.logger.error(f"Error during layer server initalization: {e}")
                raise type(e)(f"Error during layer server initalization: {e}")
            except KeyboardInterrupt:
                server_socket.close()
                self.logger.info("Stopping LayerServer...")
                raise KeyboardInterrupt("LayerServer stopped by user.")

#NOTE: MAIN FUNCTION WHEN THE SCRIPT IS RUN

if __name__ == "__main__":
    layer_server = LayerServer(host="0.0.0.0", port=8000, max_servers=2, port_pool=range(8001, 8100))
    layer_server.start()


