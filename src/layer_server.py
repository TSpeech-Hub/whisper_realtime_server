import socket, sys, logging

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def start():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        addr = ('0.0.0.0', 6000)
        s.bind(addr) # TODO change these on real server. 
        print("ooO")
        logger.info("Listening")
        s.listen(1)
        while True:
            conn, addr = s.accept()
            logger.info("client connected {}".format(addr))
            
            s.send(bytes(6001))

            conn.close()
            logger.info("client disconnected {}".format(addr))


start()
