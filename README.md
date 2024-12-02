# whisper_realtime_server TODO

**Most of the information in this README is still a work in progress, just reminders for the future of the project**

## Whisper Streaming Credits
This project uses Whisper Streaming.  
Credits to [Dominik Macháček](https://ufal.mff.cuni.cz/dominik-machacek), [Raj Dabre](https://prajdabre.github.io/), [Ondřej Bojar](https://ufal.mff.cuni.cz/ondrej-bojar), 2023.  
Check their repository for installation and documentation: https://github.com/ufal/whisper_streaming

## Installation TODO
### Important: Before building Docker, create certificates for the server.

### Whisper Server Config File JSON Tutorial

### Nvidia Developer Kit 

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Documentation TODO
A real-time server for speech-to-text from audio streaming.

## Simulations Tutorial

After installing everything needed (we installed faster-whisper and all dependencies to run it), use the following command to run the simulation:

```bash
python3 whisper_online.py resources/sample1.wav --model tiny --backend faster-whisper  --language en --min-chunk-size 1 > out.txt
```

Here, we are running the tiny model using faster-whisper.  
A lot of unreadable logs will pop up in the terminal where the command is executed. If you want to see everything in a much clearer way, open a new terminal and execute:

```bash
tail -f out.txt
```

This will show the content of `out.txt` changing in real-time in the terminal.

This code opens a server on localhost, port 12000, listening from a client microphone and returning speech-to-text:
```bash
python3 whisper_online_server.py --warmup-file resources/sample1.wav --host localhost --port 12000 --model large-v3-turbo --backend faster-whisper --language en --min-chunk-size 1
```

Run `client_test.py` with the correct host set to `localhost` and port (in this case 12000).

## OpenSSL:
---

### **What is OpenSSL?**
- OpenSSL is an open-source library used to implement SSL and TLS protocols.
- It provides command-line tools to create certificates, keys, and test connection security.

---

### **1. Install OpenSSL**
First, check if OpenSSL is already installed on your system:
```bash
openssl version
```

- If it is not installed, you can do so:
  - **Fedora**:
    ```bash
    sudo dnf install openssl
    ```

---

### **2. Generate a Private Key**
A private key is required to uniquely identify your server.

```bash
openssl genrsa -out key.pem 2048
```

- `key.pem`: File containing the private key.
- `2048`: Length of the key in bits (2048 is a secure standard).

You can view the generated key with:
```bash
cat key.pem
```

---

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

---

### **4. Verify the Certificate**
To check the generated certificate:
```bash
openssl x509 -in cert.pem -text -noout
```

---

### **5. Test an HTTPS Connection**
OpenSSL can be used to test HTTPS connections.

#### **5.1 Start an SSL Server**
Run the server with your certificate and key:
```bash
openssl s_server -accept 8000 -cert cert.pem -key key.pem
```

#### **5.2 Connect to the Server**
Use OpenSSL to connect to the server:
```bash
openssl s_client -connect localhost:8000
```

If everything works, you will see an output with the certificate and other connection details.

---

### **6. Create a Certificate Signing Request (CSR)**
To obtain a certificate signed by a Certificate Authority (CA), you need to generate a CSR:

```bash
openssl req -new -key key.pem -out request.csr
```

This command generates a file `request.csr`, which should be sent to the CA to obtain a signed certificate.

---

### **7. View the Generated Files**
- **Private key (`key.pem`):**
  ```bash
  openssl rsa -in key.pem -text -noout
  ```
- **Certificate (`cert.pem`):**
  ```bash
  openssl x509 -in cert.pem -text -noout
  ```
- **CSR (`request.csr`):**
  ```bash
  openssl req -in request.csr -text -noout
  ```

---

### **8. Simulate HTTPS with OpenSSL**
You can simulate an HTTPS server with OpenSSL and test it with a browser or tools like `curl`.

1. **Start the Server:**
   ```bash
   openssl s_server -accept 4433 -cert cert.pem -key key.pem
   ```

2. **Connect to the Server:**
   With a browser, go to:
   ```
   https://localhost:4433
   ```

   Or use `curl`:
   ```bash
   curl -k https://localhost:4433
   ```

   - The `-k` flag tells `curl` to ignore certificate validity (useful for self-signed certificates).

---

### **9. Create CA Certificates**
If you want to simulate a CA to sign certificates:

1. **Generate a CA Key:**
   ```bash
   openssl genrsa -out ca-key.pem 2048
   ```

2. **Create a CA Certificate:**
   ```bash
   openssl req -new -x509 -key ca-key.pem -out ca-cert.pem -days 365
   ```

3. **Sign a Certificate with the CA:**
   - Use a previously generated CSR:
     ```bash
     openssl x509 -req -in request.csr -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out signed-cert.pem -days 365
     ```

   - The file `signed-cert.pem` is the certificate signed by your CA.

---

### **10. Command Checklist**
| Action                          | Command                                                                                      |
|---------------------------------|----------------------------------------------------------------------------------------------|
| Generate a private key          | `openssl genrsa -out key.pem 2048`                                                           |
| Create a self-signed certificate| `openssl req -new -x509 -key key.pem -out cert.pem -days 365`                                |
| Generate a CSR                  | `openssl req -new -key key.pem -out request.csr`                                             |
| View certificate                | `openssl x509 -in cert.pem -text -noout`                                                    |
| Start an HTTPS server           | `openssl s_server -accept 4433 -cert cert.pem -key key.pem`                                 |
| Connect to the server           | `openssl s_client -connect localhost:4433`                                                 |

---

## DDOS Protection TODO

**Rate-limiting** at the connection level and using a **firewall**.

1. **Rate-Limiting**:
   - Limit the number of connections accepted from a single IP within a specific time frame.
   - Use libraries like `python-iptables` or external middleware like **HAProxy** or **Cloudflare**.

2. **Firewall (iptables)**:
   - Block IPs that exceed a certain number of requests with `iptables` rules like:
     ```bash
     iptables -A INPUT -p tcp --dport 8000 -m connlimit --connlimit-above 10 -j DROP
     ```

3. **SSL Connections with Validation**: Ensure the SSL certificate and client validation work effectively.
