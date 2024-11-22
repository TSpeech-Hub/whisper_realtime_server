import socket 

def start(): 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
        s.connect(('0.0.0.0', 6000))
        received = s.recv(1024)
        print(received)


start()
