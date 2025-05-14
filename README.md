# Whisper\_Realtime\_Server


> [!WARNING]
> Most of the informations in README are work in progress `whisper_realtime_server` is under development.

## Installation

You can either build the server using Docker or set up a custom environment. but the nvidia developer kit is required to run the server with any configurations.

### Nvidia Developer Kit

The Nvidia Developer Kit is required for GPU support. The server has been tested with CUDA 12.X and cuDNN 9, as specified in the Dockerfile. The Whisper Streaming project has been tested with CUDA 11.7 and cuDNN 8.5.0, so it is recommended to use at least CUDA 11.7 and cuDNN 8.5.0.&#x20;

<details>
   <summary><h2>Building with Docker</h2></summary>

#### Prerequisites

Make sure Docker is installed. Follow the official [Docker Installation Guide](https://docs.docker.com/get-docker/) if needed.

Clone the repository:

```bash
git git clone https://github.com/dariopellegrino00/whisper_realtime_server.git
```

#### Before builing using docker

1. Navigate to the project root directory:

   ```bash
   cd whisper_realtime_server
   ```

- When built, you will be able to test the server with your mic or a simulation of realtime audio streaming using audio files
- there are already two audio examples in the resources folder, if you want to add new ones, then **BEFORE** the next steps, add the audio files to `whisper_realtime_server/resources`

#### Steps to Build and Run the Docker Image

1. Navigate to the project root directory:

   ```bash
   cd whisper_realtime_server
   ```

2. Build the Docker image:

   ```bash
   docker build -t whisper_realtime_server .
   ```

3. Run the Docker container with GPU support and port mapping:

   ```bash
   docker run --gpus all -p 50051-50052:50051-50052 --name whisper_server whisper_realtime_server
   ```
   - You can change the port range `50051-50052:50051-50052` if needed, these are the server's default. You can use the parameters to customize the Dockerfile server startup command, check available args in the **Running the server** section below. By default, only the `--fallback` arg is passed to the server, to enable fallback logic.
   - The server is now running and ready to accept connections. You can access it at port `50051` and `50052` using the `grpcclient.py` script.

4. To stop the Docker container:

   ```bash
   docker stop whisper_server
   ```

5. To restart the Docker container:

   ```bash
   docker start whisper_server
   ```
</details>
<details>
   <summary><h2>Custom Environment</h2></summary>

Install all dependencies: 
- I suggest the use of python enviroments: [Python Enviroments](https://docs.python.org/3/library/venv.html)
- Check Dockerfile for addictional OS packages you may miss mainly for client side microphone support. (Linux only)

1. Install the requirements.txt:
   In the project root directory execute
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the pytgon-grpc files for grpc:
   ```bash
   make proto
   ```
</details>

## gRPC client

<details>
     <summary><h3>Running the test client using docker</h3></summary>

if you followed the `Building with Docker` section and you want to run the client directly in the docker container follow these steps:

1. Ensure the container in running:

   ```bash
   docker ps 
   ```

   if you see whisper-server listed then you are good to go, otherwise start the container :

   ```bash
   docker start whisper_server
   ```

2. Open a terminal in the container
   
   ```bash
   docker exec -it whisper_server /bin/bash 
   ```

   now you should see something like:
   
   ```bash
   root@<imageid>:/app/src# 
   ```

   Now toy can run the client following the next step.
   </details>

   ### Running the client

   Tf you setup a custom enviroment navigate to the project root directory (see previus step if you want to run the client directly in docker container)
   
   The test client provided is linux only compatible, an all OS compatible client is in the works.
   Run the client using your system microphone:

      ```bash
      python3 -m src.client
      ```

   All the possible options:
   ```
   --host HOST          gRPC server address
   --port PORT          gRPC server port
   --with-hypothesis    Display hypothesis updates along with the final transcript
   --simulate SIMULATE  Simulation mode: Path to the audio file to simulate a realtime audio
                        stream with
   --interactive        Display transcript updates interactively on a single line
   --chunk-duration     Change the chunk duration (in seconds) for the audio stream
   ```

      Example with confirmed only tokens:

      Standard output:
      ```
      0 600 Hi my names
      1000 2300 is Dario, nice 
      3000 4500 to meet you.
      5000 7000 How are you?
      ``` 

      Interactive output:
      ```
      Hi my names is Dario, nice to meet you. 
      How are you? 
      ```

      To have some good visualization of the real-time transcription i suggest to use the service returning hypothesis `with-hypothesis` and `interactive`. To have more frequent responses with low a client number (1 to 5 approx) set `chunk_duration` at `0.5`.  

## gRPC whisper server: 

   ### Running the server
   ```bash
   python3 -m src.server <options>
   ```
   The server is running and ready to accept connections. You can later customize the server models, behavior and other options using the command line arguments. Check the `--help` option for more details:
   ```
   --fallback            Enable fallback logic when similarity local agreement
                        fails for a mltitude of times
   --fallback-threshold FALLBACK_THRESHOLD
                        threshold t for fallback logic after t+1 similarity local
                        agreement fails (ignored if --fallback is not set)
   --qratio-threshold QRATIO_THRESHOLD
                        Threshold for qratio to confirm and insert new words
                        using the hypothesis buffer (between 0 and 100), lower
                        values than 90 are not recommended
   --buffer-trimming-sec BUFFER_TRIMMING_SEC
                        Buffer trimming is the threshold in seconds that triggers
                        the service processor audio buffer to be trimmed. This is
                        useful to avoid memory leaks and to keep the buffer size
                        under control. Default value is 15 seconds
   --ports PORTS [PORTS ...]
                        Ports to run the server on
   --max-workers MAX_WORKERS
                        Max workers for the server
   --log-every-processor
                        Log every processor in a separate file
   --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo,turbo}
                        Name size of the Whisper model to use (default:
                        large-v2). The model is automatically downloaded from the
                        model hub if not present in model cache dir
   --model-cache-dir MODEL_CACHE_DIR
                        Directory for the whisper model caching
   --model-dir MODEL_DIR
                        Directory for a custom ct2 whisper model skipping if
                        --model provided
   --warmup-file WARMUP_FILE
                        File to warm up the model and speed up the first request
   --lan LAN             Language for the whisper model to translate to (unused at
                        the moment)
   --vad                 Use VAD for the model (unused at the moment)
   --log-level LOG_LEVEL
                        Log level for the server (DEBUG, INFO, WARNING, ERROR,
                        CRITICAL) unused at the moment
   ```

## Documentation

> [!IMPORTANT]
> TODO: Add more documentation

Before setting up your own client, it's important to understand the server architecture. The client first connects to a GRPC server on the default port (`50051`). After connecting, the GRPC server assigns a service to the client. Then the client streams audio data to this port, and receives real-time transcriptions.&#x20;


## Credits

- This project uses parts of the Whisper Streaming project. Other projects involved in whisper streaming are credited in their repo, check it out: [whisper streaming](https://github.com/ufal/whisper_streaming)
- Credits also to: [faster whisper](https://github.com/SYSTRAN/faster-whisper)

## FIXED 
- [x] Send back last confirmed token when client send silent audio for a prolonged time (or no human speech audio)
- [x] Rarely other client words can end in others buffer 
- [x] `MultiProcessingFasterWhisperASR` and the Grpc Speech to text services can get stuck with high number of streaming active concurrently (10 to 20)
- [x] ValueError(f"{id} is not a registered processor.") at end of services

## KNOWN BUGS
- [ ] Random words like `ok` or `thank you` are transcribed when client stays silent (VAD is missing)
