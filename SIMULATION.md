# Simulation tutorial

after installing everything needed (we installed fastwe-whisper and all needed to run it) use the following command to run the simulation: 

```
python3 whisper_online.py resources/sample1.wav --model tiny --backend faster-whisper  --language en --min-chunk-size 1 > out.txt
```

We running tiny model using faster-whisper. 
Lot of unreadable logs will popup in the terminal that executed the command above. If you want to see everything in a much clearer way open a new terminal and execute .

```
tail -f out.txt
```

This will show the out.txt changing in real time in the terminal.

This code open a server on localhost port 12000 listening from a client mic and returning speech to text.  
```
python3 whisper_online_server.py --warmup-file resources/sample1.wav --host localhost --port 12000 --model large-v3-turbo --backend faster-whisper  --language en --min-chunk-size 1 > out.txt
```

run client_test.py with the correct host and port set 
