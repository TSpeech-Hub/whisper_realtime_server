# whisper_realtime_server
## Whisper Streaming Credits
This project uses whisper streaming
Credits to [Dominik Macháček](https://ufal.mff.cuni.cz/dominik-machacek), [Raj Dabre](https://prajdabre.github.io/), [Ondřej Bojar](https://ufal.mff.cuni.cz/ondrej-bojar), 2023
go check their reposioy for isntallation and documentation: https://github.com/ufal/whisper_streaming

## Installation TODO
### Whisper Server Config file json tutorial

## Documentation TODO 
A realtime server for speech to text given audio streaming

## Simulations tutorial

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
python3 whisper_online_server.py --warmup-file resources/sample1.wav --host localhost --port 12000 --model large-v3-turbo --backend faster-whisper  --language en --min-chunk-size 1
```

run client_test.py with the correct host as localhost and port (in this case 12000) set 

## OpenSSL: 
---

### **Cos'è OpenSSL?**
- È una libreria open-source utilizzata per implementare protocolli SSL e TLS.
- Fornisce strumenti da riga di comando per creare certificati, chiavi e per testare la sicurezza delle connessioni.

---

### **1. Installare OpenSSL**
Per prima cosa, verifica se OpenSSL è già installato sul tuo sistema:
```bash
openssl version
```

- Se non è installato, puoi farlo:
  - **Fedora**:
    ```bash
    sudo dnf install openssl
    ```

---

### **2. Generare una Chiave Privata**
La chiave privata è necessaria per identificare univocamente il tuo server.

```bash
openssl genrsa -out key.pem 2048
```

- `key.pem`: File che contiene la chiave privata.
- `2048`: Lunghezza della chiave in bit (2048 è uno standard sicuro).

Puoi visualizzare la chiave generata con:
```bash
cat key.pem
```

---

### **3. Creare un Certificato Auto-Firmato**
Per usare HTTPS, hai bisogno di un certificato. Per test locali, puoi creare un certificato auto-firmato (self-signed):

```bash
openssl req -new -x509 -key key.pem -out cert.pem -days 365
```

Durante l’esecuzione del comando, ti verranno chieste alcune informazioni:
- **Country Name (2 letter code):** Codice del paese (es. `IT`).
- **State or Province Name:** Nome dello stato o provincia.
- **Locality Name:** Nome della città.
- **Organization Name:** Nome dell’organizzazione.
- **Common Name:** Inserisci `localhost` per test locali.

Il file `cert.pem` è il certificato auto-firmato. Puoi visualizzarlo con:
```bash
cat cert.pem
```

---

### **4. Verificare il Certificato**
Per controllare il certificato generato:
```bash
openssl x509 -in cert.pem -text -noout
```

---

### **5. Testare una Connessione HTTPS**
OpenSSL può essere usato per testare connessioni HTTPS.

#### **5.1 Avviare un Server SSL**
Esegui il server con il tuo certificato e chiave:
```bash
openssl s_server -accept 8000 -cert cert.pem -key key.pem
```

#### **5.2 Collegarsi al Server**
Usa OpenSSL per connetterti al server:
```bash
openssl s_client -connect localhost:8000
```

Se tutto funziona, vedrai un output con il certificato e altre informazioni di connessione.

---

### **6. Creare una Certificate Signing Request (CSR)**
Se vuoi ottenere un certificato firmato da una CA (Certificate Authority), devi generare una CSR:

```bash
openssl req -new -key key.pem -out request.csr
```

Questo comando genera un file `request.csr` che deve essere inviato alla CA per ottenere un certificato firmato.

---

### **7. Visualizzare i File Generati**
- **Chiave privata (`key.pem`):**
  ```bash
  openssl rsa -in key.pem -text -noout
  ```
- **Certificato (`cert.pem`):**
  ```bash
  openssl x509 -in cert.pem -text -noout
  ```
- **CSR (`request.csr`):**
  ```bash
  openssl req -in request.csr -text -noout
  ```

---

### **8. Simulare HTTPS con OpenSSL**
Puoi simulare un server HTTPS con OpenSSL e testare con il browser o con strumenti come `curl`.

1. **Avvia il Server:**
   ```bash
   openssl s_server -accept 4433 -cert cert.pem -key key.pem
   ```

2. **Collegati al Server:**
   Con il browser, vai su:
   ```
   https://localhost:4433
   ```

   Oppure usa `curl`:
   ```bash
   curl -k https://localhost:4433
   ```

   - Il flag `-k` dice a `curl` di ignorare la validità del certificato (utile per i certificati auto-firmati).

---

### **9. Creare Certificati CA**
Se vuoi simulare una CA per firmare certificati:

1. **Genera una Chiave CA:**
   ```bash
   openssl genrsa -out ca-key.pem 2048
   ```

2. **Crea un Certificato CA:**
   ```bash
   openssl req -new -x509 -key ca-key.pem -out ca-cert.pem -days 365
   ```

3. **Firma un Certificato con la CA:**
   - Usa un CSR generato in precedenza:
     ```bash
     openssl x509 -req -in request.csr -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out signed-cert.pem -days 365
     ```

   - Il file `signed-cert.pem` è il certificato firmato dalla tua CA.

---

### **10. Checklist dei Comandi**
| Azione                         | Comando                                                                                     |
|--------------------------------|---------------------------------------------------------------------------------------------|
| Generare chiave privata        | `openssl genrsa -out key.pem 2048`                                                          |
| Creare certificato auto-firmato| `openssl req -new -x509 -key key.pem -out cert.pem -days 365`                               |
| Generare una CSR               | `openssl req -new -key key.pem -out request.csr`                                           |
| Visualizzare certificato        | `openssl x509 -in cert.pem -text -noout`                                                   |
| Avviare server HTTPS           | `openssl s_server -accept 4433 -cert cert.pem -key key.pem`                                |
| Connettersi al server          | `openssl s_client -connect localhost:4433`                                                |

---

## DDOS Protection

Il modo migliore per limitare il **DDoS** nella tua applicazione è implementare un **rate-limiting** a livello di connessione e usare un **firewall**.

1. **Rate-Limiting**:
   - Limita il numero di connessioni accettate da un singolo IP in un determinato periodo.
   - Usa librerie come `python-iptables` o middleware esterni come **HAProxy** o **Cloudflare**.

2. **Firewall (iptables)**:
   - Blocca IP che superano un certo numero di richieste con regole di `iptables`:
     ```bash
     iptables -A INPUT -p tcp --dport 8000 -m connlimit --connlimit-above 10 -j DROP
     ```

3. **Connessioni SSL con validazione**:
   - Richiedi un certificato client valido per autorizzare solo connessioni legittime.
