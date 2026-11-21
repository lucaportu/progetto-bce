# =============================================================================
# Approccio C – Kimi K3.1 Pro (API Moonshot) classifica le frasi di test WCB
# Da incollare nel widget "Python Script" di Orange.
#
# Ingresso  Data   : il Corpus "TEST BCE" (frasi_bce_wcb_test.tab), SENZA embedding
# Uscite    Data   : le stesse frasi con in più la colonna "kimi" (etichetta dell'LLM)
#           Object : risultati della valutazione -> collegare a Confusion Matrix
# Nel riquadro "Console" del widget: F1 pesato, accuratezza, kappa, F1 per classe.
#
# Le risposte vengono salvate in risultati/llm_<modello>.csv: se rilanciate lo script,
# le frasi già classificate (con lo stesso modello e le stesse istruzioni) non vengono
# richieste di nuovo, quindi non pagate due volte.
# =============================================================================

# ----------------------------------------------------------------- DA COMPILARE
API_KEY = ""   # lasciate vuoto: la chiave viene letta dalla variabile d'ambiente MOONSHOT_API_KEY
               # oppure dal file ~/.kimi_api_key (una riga con la sola chiave, fuori dal progetto).
               # In alternativa incollatela qui tra le virgolette solo nel nodo di Orange, mai in questo file:
               # resta salvata nel .ows, quindi va cancellata prima di consegnare o condividere il workflow.

# ----------------------------------------------------------------- impostazioni
MODELLO = "kimi-k3.1-pro"         # oppure "kimi-k3.1-pro-highspeed" (più veloce, costa il doppio)
RAGIONAMENTO = "high"             # reasoning_effort: "low" (economico), "high", "max" (default di Kimi, il più caro)
LIMITE = 10                       # prima prova: solo 10 frasi. Poi mettete None per tutte
CHIAMATE_IN_PARALLELO = 2         # alzate a 4-8 se il vostro piano lo permette (altrimenti errori 429)
BASE_URL = "https://api.moonshot.ai/v1"

# Da dove chiamare il modello. Prompt, modello, ragionamento e file delle risposte restano gli stessi:
#   "moonshot"   = API di Kimi (usa API_KEY)
#   "openrouter" = stesso modello via OpenRouter, servito SOLO dai server di Moonshot AI (usa API_KEY_OPENROUTER)
FORNITORE = "openrouter"
API_KEY_OPENROUTER = ""   # chiave OpenRouter (sk-or-...). Stessa avvertenza: cancellatela prima di condividere il .ows
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SERVER = "moonshotai/mxfp4"   # endpoint di Moonshot AI su OpenRouter (non la variante highspeed)

import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from Orange.data import DiscreteVariable
from Orange.evaluation import Results

PROGETTO = Path("/Users/lucaportugalli/Desktop/AIDA/AI and Computer Vision/progetto_BCE")
GUIDA = PROGETTO / "dati_bce_wcb/originali/guida_annotazione/stance.tex"
CACHE = PROGETTO / "risultati" / f"llm_{re.sub(r'[^A-Za-z0-9.-]', '_', MODELLO)}.csv"


# ----------------------------------------------------------------- istruzioni per l'LLM
def istruzioni():
    guida = GUIDA.read_text(encoding="utf-8")
    guida = re.sub(r"\\mptext\{ecb\}\{nine\}", "The guideline groups sentences into nine categories:", guida)
    return f"""You are an expert annotator of central bank communication.

TASK
You will receive ONE sentence taken from the accounts of the European Central Bank (ECB) monetary
policy meetings (2015-2024). The sentence is shown on its own, without the surrounding text, and it is
all in lowercase. Classify the monetary policy stance it expresses or implies:

- HAWKISH: the sentence points towards tighter monetary policy: higher interest rates, fewer asset
  purchases, rates kept high for longer. This includes sentences describing economic developments that
  would push the ECB in that direction (for example inflation above target or rising, upside inflation
  risks, stronger growth, a tighter labour market, rising wages, a weaker euro, higher energy prices).
- DOVISH: the sentence points towards looser monetary policy: rate cuts, more asset purchases, rates
  kept low for longer. This includes developments that would push the ECB in that direction (for example
  inflation below target or falling, weaker growth, downside risks to growth, rising unemployment,
  a stronger euro, lower energy prices).
- NEUTRAL: no clear direction. Purely descriptive, procedural or historical statements; signals in
  opposite directions that balance out; statements that policy stays unchanged; sentences not about
  monetary policy or the economy.

If a sentence contains signals in both directions, choose the one that clearly dominates; if neither
dominates, answer NEUTRAL.

ANNOTATION GUIDELINE
These are the guidelines given to the human experts who labelled these sentences (LaTeX source).
Use the table to decide the direction of economic developments. The "Irrelevant" column does not apply
here: answer NEUTRAL for sentences that are not about monetary policy.

{guida}

RULES
- Judge only the sentence itself. Do not use any knowledge of what the ECB decided afterwards.
- Answer with exactly one word: HAWKISH, DOVISH or NEUTRAL. No explanation."""


def chiave_api():
    if FORNITORE == "openrouter":
        k = API_KEY_OPENROUTER.strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not k:
            raise RuntimeError("Manca la chiave OpenRouter: mettetela nella variabile d'ambiente OPENROUTER_API_KEY "
                               "(oppure, solo nel nodo di Orange, in API_KEY_OPENROUTER = \"...\").")
        return k
    k = API_KEY.strip() or os.environ.get("MOONSHOT_API_KEY", "").strip()
    if not k and (Path.home() / ".kimi_api_key").exists():
        k = (Path.home() / ".kimi_api_key").read_text(encoding="utf-8").strip()
    if not k:
        raise RuntimeError("Manca la chiave Kimi: mettetela nella variabile d'ambiente MOONSHOT_API_KEY o nel file "
                           "~/.kimi_api_key (oppure, solo nel nodo di Orange, in API_KEY = \"...\").")
    return k


def richiesta(percorso, corpo=None):
    req = urllib.request.Request(
        (OPENROUTER_URL if FORNITORE == "openrouter" else BASE_URL) + percorso,
        data=None if corpo is None else json.dumps(corpo).encode("utf-8"),
        headers={"Authorization": f"Bearer {CHIAVE}", "Content-Type": "application/json"},
        method="GET" if corpo is None else "POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def spiega_errore(e):
    testo = e.read().decode("utf-8", "replace")
    if e.code == 401:
        return ("Errore 401: chiave non valida. Controllate di aver copiato la chiave completa "
                "(la piattaforma la mostra intera solo quando la create).")
    if e.code in (402, 403):
        return (f"Errore {e.code}: accesso negato o credito insufficiente. K3.1 Pro si sblocca dopo una "
                f"prima ricarica (minimo 1 $). Dettagli: {testo}")
    return f"Errore API {e.code}: {testo}"


def controlla_modello():
    """Verifica chiave e nome del modello prima di spendere qualcosa."""
    if FORNITORE == "openrouter":
        try:
            richiesta("/key")   # 401 se la chiave è sbagliata
            server = [e.get("tag") for e in
                      richiesta(f"/models/moonshotai/{MODELLO}/endpoints")["data"]["endpoints"]]
        except urllib.error.HTTPError as e:
            raise RuntimeError(spiega_errore(e)) from None
        if OPENROUTER_SERVER not in server:
            raise RuntimeError(f"Su OpenRouter il server '{OPENROUTER_SERVER}' non serve '{MODELLO}'. "
                               f"Server disponibili: {server}")
        return
    try:
        disponibili = sorted(m["id"] for m in richiesta("/models")["data"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(spiega_errore(e)) from None
    if MODELLO not in disponibili:
        raise RuntimeError(f"Il modello '{MODELLO}' non è disponibile con questa chiave.\n"
                           f"Modelli disponibili: {disponibili}")


ERRORI = PROGETTO / "risultati" / "llm_errori.txt"


def registra_errore(testo):
    """Scrive l'errore anche su file (la Console di Orange lo mostra solo a fine esecuzione)."""
    ERRORI.parent.mkdir(exist_ok=True)
    with open(ERRORI, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {testo}\n")


def classifica(frase):
    """Restituisce (etichetta, risposta grezza, modello usato).
    Errori di tutto l'account (chiave, credito, limiti esauriti): ferma tutto con un messaggio chiaro.
    Errori di una sola frase (es. frase rifiutata dal filtro): la salta, la annota e continua."""
    messaggi = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": f"Sentence: {frase}"}]
    if FORNITORE == "openrouter":
        corpo = {"model": f"moonshotai/{MODELLO}", "reasoning": {"effort": RAGIONAMENTO}, "max_tokens": 32000,
                 "messages": messaggi,
                 "provider": {"only": [OPENROUTER_SERVER], "allow_fallbacks": False}}  # solo i server di Moonshot AI
    else:
        corpo = {"model": MODELLO, "reasoning_effort": RAGIONAMENTO, "max_completion_tokens": 32000,
                 "messages": messaggi}
    for tentativo in range(6):
        try:
            r = richiesta("/chat/completions", corpo)
            if "choices" not in r:   # OpenRouter a volte segnala l'errore con codice 200
                err = r.get("error") or {}
                codice, testo = int(err.get("code") or 500), json.dumps(r)[:800]
            else:
                risposta = r["choices"][0]["message"].get("content") or ""
                trovate = re.findall(r"\b(HAWKISH|DOVISH|NEUTRAL)\b", risposta.upper())
                usato = r.get("model", "") + (f" via {r['provider']}" if r.get("provider") else "")
                return (trovate[-1] if trovate else ""), risposta.strip(), usato
        except urllib.error.HTTPError as e:
            codice, testo = e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            registra_errore(f"rete (tentativo {tentativo + 1}) {e} | frase: {frase[:80]}")
            if tentativo < 5:
                time.sleep(2 ** tentativo * 3)
                continue
            raise
        registra_errore(f"HTTP {codice} (tentativo {tentativo + 1}) {testo[:500]} | frase: {frase[:80]}")
        if codice == 402 or (codice == 429 and re.search(r"quota|balance|suspend|insufficient|recharge|credit",
                                                         testo, re.I)):
            raise RuntimeError(f"Credito esaurito ({codice}): ricaricate l'account ({FORNITORE}) e premete di "
                               f"nuovo Run. Dettagli: {testo}")
        if codice in (408, 429, 500, 502, 503, 504) and tentativo < 5:
            time.sleep(2 ** tentativo * 3)
            continue
        if codice in (401, 403, 408, 429, 500, 502, 503, 504):
            raise RuntimeError(f"Errore {codice} che riguarda l'account o il servizio, non la frase. "
                               f"Dettagli: {testo}")
        return "", f"ERRORE {codice}: {testo[:400]}", ""   # es. 400: frase rifiutata -> si salta
    return "", "", ""


# ----------------------------------------------------------------- esecuzione
if in_data is None:
    raise RuntimeError("Collegate il Corpus 'TEST BCE' all'ingresso Data del widget.")

CHIAVE = chiave_api()
SISTEMA = istruzioni()
VERSIONE = hashlib.sha1(f"{MODELLO}|{RAGIONAMENTO}|{SISTEMA}".encode()).hexdigest()[:10]
classe = in_data.domain.class_var
frasi = [str(s) for s in in_data.get_column("sentence")]

# risposte già salvate con lo stesso modello e le stesse istruzioni
salvate = {}
if CACHE.exists():
    with open(CACHE, encoding="utf-8", newline="") as f:
        salvate = {r["sentence"]: r for r in csv.DictReader(f)
                   if r["etichetta"] and r.get("versione") == VERSIONE}

da_fare = [s for s in dict.fromkeys(frasi) if s not in salvate]
if LIMITE is not None:
    da_fare = da_fare[:max(0, LIMITE - sum(s in salvate for s in frasi))]
print(f"{MODELLO} (ragionamento {RAGIONAMENTO}): {len(salvate)} risposte già salvate, "
      f"{len(da_fare)} frasi da chiedere.")

if da_fare:
    controlla_modello()
    CACHE.parent.mkdir(exist_ok=True)
    nuovo_file = not CACHE.exists()
    with open(CACHE, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sentence", "etichetta", "risposta", "modello", "versione", "ora"])
        if nuovo_file:
            w.writeheader()
        with ThreadPoolExecutor(max_workers=CHIAMATE_IN_PARALLELO) as pool:
            for frase, (etichetta, grezza, usato) in zip(da_fare, pool.map(classifica, da_fare)):
                riga = {"sentence": frase, "etichetta": etichetta, "risposta": grezza[:500],
                        "modello": usato, "versione": VERSIONE, "ora": time.strftime("%Y-%m-%d %H:%M:%S")}
                w.writerow(riga)
                f.flush()
                if usato and (MODELLO not in usato or (FORNITORE == "openrouter" and "moonshot" not in usato.lower())):
                    print(f"ATTENZIONE: ha risposto '{usato}', non '{MODELLO}' di Moonshot AI.")
                if etichetta:
                    salvate[frase] = riga
                else:
                    print("Risposta non interpretabile:", grezza[:200])

# dalla parola dell'LLM all'indice della classe di Orange (es. "hawkish (falco)")
indice = {p: next(i for i, v in enumerate(classe.values) if v.lower().startswith(p.lower()))
          for p in ("HAWKISH", "DOVISH", "NEUTRAL")}
pred = np.array([indice[salvate[s]["etichetta"]] if s in salvate else np.nan for s in frasi], dtype=float)

# uscita Data: frasi + colonna "kimi"
out_data = in_data.add_column(DiscreteVariable("kimi", values=classe.values), pred, to_metas=True)

# uscita Object: risultati per Confusion Matrix (solo le frasi già classificate)
ok = ~np.isnan(pred)
if ok.sum() == 0:
    out_object = None
    print("Nessuna frase classificata.")
else:
    sub = in_data[np.flatnonzero(ok)]
    p = pred[ok].astype(int)
    prob = np.zeros((len(p), len(classe.values)))
    prob[np.arange(len(p)), p] = 1
    res = Results(data=sub, row_indices=np.arange(len(p)), actual=sub.Y.astype(float),
                  predicted=p[None, :].astype(float), probabilities=prob[None])
    res.learner_names = [f"Kimi ({MODELLO})"]
    out_object = res

    # uscita Classifier: un "modello" che restituisce le risposte salvate di Kimi.
    # Collegato a Predictions (insieme al Corpus TEST BCE) mostra F1, CA, Prec, Recall, MCC
    # nella stessa tabella usata da Test & Score. Non chiama l'API: legge solo le risposte salvate.
    from Orange.classification import Model

    class RisposteLLM(Model):
        def __init__(self, domain, risposte, nome):
            super().__init__(domain)
            self.risposte, self.name = risposte, nome

        def predict_storage(self, data):
            testi = data.get_column("sentence") if "sentence" in data.domain else data.documents
            v = np.array([self.risposte.get(str(t), np.nan) for t in testi], dtype=float)
            k = len(self.domain.class_var.values)
            prob = np.full((len(v), k), 1 / k)          # frasi senza risposta: nessuna preferenza
            ok_ = ~np.isnan(v)
            prob[ok_] = 0
            prob[np.flatnonzero(ok_), v[ok_].astype(int)] = 1
            return np.argmax(prob, axis=1).astype(float), prob

    out_classifier = RisposteLLM(in_data.domain,
                                 {s: float(indice[salvate[s]["etichetta"]]) for s in salvate},
                                 f"Kimi ({MODELLO})")

    # metriche
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    y = sub.Y.astype(int)
    print(f"\nFrasi valutate: {len(y)} su {len(frasi)}"
          + ("" if len(y) == len(frasi) else
             "   <-- PROVA PARZIALE: mettete LIMITE = None per tutte" if LIMITE is not None else
             "   <-- alcune frasi mancano: vedete risultati/llm_errori.txt e premete di nuovo Run"))
    print(f"F1 pesato   {f1_score(y, p, average='weighted'):.3f}")
    print(f"F1 macro    {f1_score(y, p, average='macro'):.3f}")
    print(f"Accuratezza {accuracy_score(y, p):.3f}")
    print(f"Kappa       {cohen_kappa_score(y, p):.3f}")
    rng = np.random.default_rng(0)
    boot = [f1_score(y[i], p[i], average="weighted")
            for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    print(f"F1 pesato, intervallo 95% (bootstrap): {np.percentile(boot, 2.5):.3f} – {np.percentile(boot, 97.5):.3f}")
    etichette = list(range(len(classe.values)))
    for v, f1 in zip(classe.values, f1_score(y, p, average=None, labels=etichette)):
        print(f"  F1 {v:<20} {f1:.3f}")
    print("Matrice di confusione (righe = esperti, colonne = Kimi):", list(classe.values))
    print(confusion_matrix(y, p, labels=etichette))
