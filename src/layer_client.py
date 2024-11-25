import socket 

def start(): 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
        try:
            s.connect(('0.0.0.0', 6000))
            received = s.recv(1024)
            print(received)
        except Exception as e :
            s.close()
            print(f"ded {e}")



start()
