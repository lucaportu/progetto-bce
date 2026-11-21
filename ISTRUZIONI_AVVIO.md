# Come avviare il progetto e la dashboard Streamlit

Per usare questo progetto e avviare la dashboard web sul tuo computer (Mac, Windows o Linux), segui questi passaggi:

---

### Requisiti
- **Python** installato, versione **3.12 o successiva** (consigliata 3.13): le versioni bloccate in
  `requirements.txt` (numpy 2.5, pandas 3.1, networkx 3.7, …) non si installano con Python 3.11 o precedenti.
- Al primo uso dell'analizzatore di discorsi l'app scarica il modello SBERT da Hugging Face (circa 470 MB).

---

### Passaggio 1: Apri il terminale nella cartella del progetto
Apri il terminale (o PowerShell/CMD su Windows) ed entra nella cartella:

```bash
cd percorso/della/cartella/progetto_BCE
```

---

### Passaggio 2: Crea l'ambiente virtuale locale
Su Mac, se la cartella arriva da uno zip e il comando qui sotto dà "Permission denied", rendetela prima scrivibile:
`chmod -R u+w .`

L'ambiente `.venv` non si copia da un computer all'altro né si sposta con la cartella: se spostate il progetto,
cancellate `.venv` e ripetete i passaggi 2–4.

```bash
python3 -m venv .venv
```
*(Su Windows: `python -m venv .venv`)*

---

### Passaggio 3: Attiva l'ambiente virtuale

- **Su Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```

- **Su Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Se PowerShell blocca gli script, usa il Prompt dei comandi classico ed esegui `.venv\Scripts\activate.bat`)*

---

### Passaggio 4: Installa le librerie
```bash
pip install -r requirements.txt
```

---

### Passaggio 5: Avvia la dashboard Streamlit
```bash
streamlit run app/app.py
```

La dashboard si aprirà automaticamente nel tuo browser all'indirizzo `http://localhost:8501`.
