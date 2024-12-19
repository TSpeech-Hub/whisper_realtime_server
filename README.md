# whisper_realtime_server TODO

**Most of the information in this README is still a work in progress, just reminders for the future of the project**

## Whisper Streaming Credits
This project uses parto of the Whisper Streaming project.  
Credits to [Dominik Macháček](https://ufal.mff.cuni.cz/dominik-machacek), [Raj Dabre](https://prajdabre.github.io/), [Ondřej Bojar](https://ufal.mff.cuni.cz/ondrej-bojar), 
Check their repository for documentation: https://github.com/ufal/whisper_streaming

## Installation TODO
### Important: Before building with Docker, create key and certificate for the server.
If first you are just testing the server just create a private key and a self signed certificate 
A private key is required to uniquely identify your server.

<instruction to build docker> 

```bash
openssl genrsa -out key.pem 2048
```

- `key.pem`: File containing the private key.
- `2048`: Length of the key in bits (2048 is a secure standard).

You can view the generated key with:
```bash
cat key.pem
```

### **3. Create a Self-Signed Certificate**
To use HTTPS, you need a certificate. For local tests, you can create a self-signed certificate:

```bash
openssl req -new -x509 -key key.pem -out cert.pem -days 365
```

During execution, you will be asked for some information:
- **Country Name (2 letter code):** Country code (e.g., `US`).
- **State or Province Name:** Name of the state or province.
- **Locality Name:** Name of the city.
- **Organization Name:** Name of the organization.
- **Common Name:** Enter `localhost` for local testing.

The file `cert.pem` is the self-signed certificate. You can view it with:
```bash
cat cert.pem
```

### Whisper Server Config File JSON Tutorial
for now do not edit the config.json file,i suggest to only edit the model size 

### Nvidia Developer Kit 

required for gpu used wich is the  

## Documentation TODO
A real-time server for speech-to-text from audio streaming.

## Simulations Tutorial
get in the src dir 
run the container or run python3 layer_server.py in you set enviroment
if you want to use a microphone just run python3 client 
if you want to simulate an audio streaming via file use python3 client.py <filepath>
