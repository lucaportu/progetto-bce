# Pubblicare la dashboard online (Streamlit Community Cloud, gratis)

Risultato: un link del tipo `https://falchi-colombe-bce.streamlit.app` che funziona da qualunque browser, anche
con il computer spento.

## Cosa è già pronto nel progetto

- `.gitignore`: tiene fuori da GitHub `.venv`, la chiave in `.streamlit/secrets.toml` e i file grandi
  rigenerabili (`dati_grezzi/ecb_speeches_all.csv`, `dati_preparati/frasi_bce_tutte.csv`, `modello/cache/`).
- `app/requirements.txt`: le librerie per il server. È uguale a `requirements.txt`, ma con torch in versione solo
  CPU (~200 MB invece di diversi GB). Streamlit Cloud lo legge al posto di quello nella cartella principale.
- Il riassunto con Kimi è protetto da password: online la chiave la pagate voi, quindi solo chi conosce
  `PASSWORD_RIASSUNTO` può generarlo. In locale, se la password non è impostata, funziona come prima.

## 1. Mettere il progetto su GitHub

1. Su <https://github.com/new> create un repository (per esempio `progetto-bce`). Va bene pubblico o privato;
   **non** aggiungete README, .gitignore o licenza (ci sono già).
2. Dal terminale, nella cartella `progetto_BCE` (il repository git locale è già inizializzato):

   ```bash
   git add .
   git commit -m "Dashboard Falchi & Colombe"
   git remote add origin https://github.com/VOSTRO-UTENTE/progetto-bce.git
   git push -u origin main
   ```

   In alternativa: GitHub Desktop → *Add existing repository* → questa cartella → *Publish repository*.

Prima del push controllate che `git status` **non** elenchi `.streamlit/secrets.toml`.

## 2. Creare l'app su Streamlit Cloud

1. Entrate su <https://share.streamlit.io> con l'account GitHub (se il repository è privato, autorizzate
   l'accesso ai repository privati quando ve lo chiede).
2. *Create app* → *Deploy a public app from GitHub*:
   - **Repository**: `VOSTRO-UTENTE/progetto-bce`, **Branch**: `main`
   - **Main file path**: `app/app.py`
   - **App URL**: il nome che volete, per esempio `falchi-colombe-bce`
3. *Advanced settings*:
   - **Python version**: `3.13` (va bene anche 3.12; le versioni precedenti non funzionano)
   - **Secrets**: incollate

     ```toml
     KIMI_API_KEY = "la-vostra-chiave"
     PASSWORD_RIASSUNTO = "una-password-a-scelta"
     ```

     Se non volete il riassunto online, non mettete nessuna delle due righe: la sezione mostra solo una nota.
4. *Deploy*. La prima installazione richiede qualche minuto. Il primo uso di "Analizza un discorso" scarica
   anche il modello SBERT (~470 MB, una volta sola).

Ogni `git push` successivo aggiorna l'app da solo.

## Da sapere

- **Pausa:** se nessuno apre l'app per qualche giorno, va in pausa. Chi la apre vede un pulsante per
  risvegliarla e aspetta circa un minuto. Prima di una presentazione apritela voi qualche minuto prima.
- **Memoria:** con il modello caricato l'app usa circa 1 GB di RAM, dentro i limiti del piano gratuito.
- **YouTube:** YouTube blocca spesso le richieste di sottotitoli che arrivano dai server cloud, quindi online
  l'analisi da link YouTube può fallire. Testo incollato e link al sito della BCE funzionano.
- **Dati:** i dataset Fed e WCB hanno licenza CC BY-NC (uso non commerciale, citando la fonte): va bene per un
  progetto universitario, e le fonti sono già citate nel piè di pagina dell'app.
