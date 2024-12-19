# whisper\_realtime\_server

**Most of the information in this README is still a work in progress**

## Installation

### Docker Installation

#### Important: Before Building with Docker

Before building the server with Docker, you need to create a private key and a certificate. This is required to enable HTTPS for secure communication. A key and certificate are required, even for testing purposes.

#### 1. Generate a Private Key

Use the following command to generate a private key:

```bash
openssl genrsa -out key.pem 2048
```

- `key.pem`: File containing the private key.
- `2048`: Length of the key in bits (2048 is a secure standard).

View the generated key with:

```bash
cat key.pem
```

#### 2. Create a Self-Signed Certificate

Generate a self-signed certificate using the private key (for testing purposes):

```bash
openssl req -new -x509 -key key.pem -out cert.pem -days 365
```

During execution, you will be asked to enter some information:

- **Country Name (2 letter code):** e.g., `US`
- **State or Province Name:** e.g., `California`
- **Locality Name:** e.g., `San Francisco`
- **Organization Name:** Your organization or project name
- **Common Name:** Use `localhost` for local testing

View the self-signed certificate with:

```bash
cat cert.pem
```

For production use, a valid certificate issued by a trusted Certificate Authority (CA) is required to ensure a secure connection. Certificate check is disabled on the client for testing.

### Building with Docker

#### Prerequisites

Make sure Docker is installed. Follow the official [Docker Installation Guide](https://docs.docker.com/get-docker/) if needed.

Clone the repository:

```bash
git clone https://github.com/dariopellegrino00/whisper_realtime_server.git
cd whisper_realtime_server
```

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
   docker run --gpus all -p 8000-8050:8000-8050 --name whisper_server whisper_realtime_server
   ```

   - You can change the port range `8000-8050` if needed.
   - The server is now running and ready to accept connections. You can access it at port 8000 using the `client.py` script.

4. To stop the Docker container:

   ```bash
   docker stop whisper_server
   ```

5. To restart the Docker container:

   ```bash
   docker start whisper_server
   ```

### Custom Environment

#### TODO

## Whisper Server Config File JSON Tutorial

For now, avoid modifying the `config.json` file. If you need to experiment, it is advisable to only adjust the model size parameter.

## Nvidia Developer Kit

The Nvidia Developer Kit is required for GPU support. The server has been tested with CUDA 12.X and cuDNN 9, as specified in the Dockerfile. The Whisper Streaming project has been tested with CUDA 11.7 and cuDNN 8.5.0, so it is recommended to use at least CUDA 11.7 and cuDNN 8.5.0.&#x20;

## Documentation

Before setting up your own client, it's important to understand the server architecture. The client first connects to a layer server on the default port (8000). After connecting, the layer server assigns a port number to the client. The client then connects to the same host on the assigned port, streams audio data to this port, and receives real-time transcriptions.&#x20;

## Simulations Tutorial

1. Navigate to the `src` directory:

   ```bash
   cd src
   ```

2. Run the server directly with Python:

   ```bash
   python3 layer_server.py
   ```

3. To use a microphone for audio input:

   ```bash
   python3 client.py
   ```

4. To simulate audio streaming from a file:

   ```bash
   python3 client.py <filepath>
   ```
## Credits

- This project uses parts of the Whisper Streaming project. Other projects involved in whisper streaming are credited in their repo, check it out: [whisper streaming](https://github.com/ufal/whisper_streaming)
- Credits also to: [faster whisper](https://github.com/SYSTRAN/faster-whisper)
Ecco una versione aggiornata con il riferimento allo stato embrionale del progetto e l'invito a contribuire:

## Contributing

This project is still in an early stage of development, and there may be significant bugs or issues. All contributions are welcome and greatly appreciated!
If you'd like to contribute, here's how you can help:

- **Fork** the repository.
- Create a **new branch** for your feature or bug fix.
- Submit a **pull request** with a clear description of your changes.

For major changes, please open an **issue** first to discuss what you'd like to change.
Thank you for helping improve this project and making it better for everyone!

## TODO
- [ ] Custom enviroment setup
- [ ] grpc implementation
- [ ] remove unused packages in Dockerfile and requirements 


