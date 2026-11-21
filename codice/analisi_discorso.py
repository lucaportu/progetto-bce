"""Motore dell'analizzatore di discorsi (README, sezione 8, punto 3). Usato da app/app.py.

- dividi_in_frasi: stesso trattamento delle frasi dei discorsi BCE (prepara_frasi_bce.py): paragrafi, NLTK punkt,
  stessi scarti (frammenti, frasi lunghissime, note e riferimenti) e stesso filtro di politica monetaria.
- Classificatore: il modello finale (modello/rete_finale.joblib) sugli embedding SBERT calcolati in locale.
- testo_da_url_bce: testo di un discorso dal sito della BCE.
- frasi_da_youtube: sottotitoli di un video YouTube, divisi in frasi con il tempo di inizio di ciascuna.
- riassunto_llm: riassunto finale con Kimi (unico uso dell'LLM; serve la chiave API).
"""
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import joblib
import nltk
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "codice"))
from prepara_frasi_bce import (MAX_PAROLE, MIN_PAROLE, RE_INIZIO_RIFERIMENTO, RE_RIFERIMENTO,  # noqa: E402
                               normalizza, parole_chiave)

CLASSI_BREVI = ["colomba", "falco", "neutrale"]      # stesso ordine delle colonne di predict_proba
RE_PARAGRAFO_TESTO = re.compile(r"\n\s*\n|\n(?=[A-Z“\"‘'(\[])")
UA = "Mozilla/5.0 (Macintosh) progetto-universitario"


def _punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@dataclass
class Frase:
    testo: str
    politica_monetaria: bool
    motivo_esclusione: str = ""     # "" se la frase è tenuta
    inizio: float | None = None     # secondi (solo per YouTube)


def _valuta(frase: str) -> Frase:
    n = len(frase.split())
    if n < MIN_PAROLE or not re.search(r"[A-Za-z]{3}", frase):
        return Frase(frase, False, "frammento")
    if n > MAX_PAROLE:
        return Frase(frase, False, "troppo lunga")
    if RE_RIFERIMENTO.search(frase) or RE_INIZIO_RIFERIMENTO.match(frase):
        return Frase(frase, False, "riferimento")
    return Frase(frase, bool(parole_chiave(frase)))


def dividi_in_frasi(testo: str) -> list[Frase]:
    """Testo -> frasi, ciascuna marcata come di politica monetaria o no (le altre restano, in grigio)."""
    _punkt()
    out = []
    for blocco in RE_PARAGRAFO_TESTO.split(testo.replace("\r", "")):
        blocco = normalizza(blocco)
        if blocco:
            out += [_valuta(f) for f in nltk.sent_tokenize(blocco)]
    return out


class Classificatore:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        info = json.loads((BASE / "modello/rete_finale.json").read_text())
        self.sbert = SentenceTransformer(info["modello_embedding"], device="cpu")
        self.rete = joblib.load(BASE / "modello/rete_finale.joblib")
        self.info = info

    def probabilita(self, frasi: list[str]) -> np.ndarray:
        """Matrice n x 3 con P(colomba), P(falco), P(neutrale)."""
        if not frasi:
            return np.zeros((0, 3))
        X = self.sbert.encode(frasi, batch_size=64, normalize_embeddings=False)
        return self.rete.predict_proba(X.astype(np.float64))


def descrivi(p: np.ndarray) -> dict:
    """Da una riga di probabilità: classe, punteggio per l'indice, punteggio continuo, sicurezza."""
    k = int(np.argmax(p))
    return {"classe": CLASSI_BREVI[k], "punteggio": {0: -1, 1: 1, 2: 0}[k],
            "punteggio_continuo": float(p[1] - p[0]), "sicurezza": float(p.max()),
            "p_colomba": float(p[0]), "p_falco": float(p[1]), "p_neutrale": float(p[2])}


# ----------------------------------------------------------------------------- testo dal sito BCE
def testo_da_url_bce(url: str) -> tuple[str, str]:
    """(titolo, testo) di un discorso pubblicato su ecb.europa.eu. Note e riquadri finali sono esclusi."""
    from bs4 import BeautifulSoup
    host = urllib.parse.urlparse(url).hostname or ""
    if not (host == "ecb.europa.eu" or host.endswith(".ecb.europa.eu")):
        raise ValueError("Sono accettati solo link al sito della BCE (ecb.europa.eu).")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        pagina = r.read().decode("utf-8", "replace")
    soup = BeautifulSoup(pagina, "html.parser")
    titolo = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    corpo = soup.find("main") or soup
    for tag in corpo.select("footer, nav, script, style, .footnotes, .related-topics, .address-box, sup"):
        tag.decompose()
    paragrafi = [p.get_text(" ", strip=True) for p in corpo.find_all(["p", "li"])]
    paragrafi = [p for p in paragrafi if len(p.split()) >= 4]
    if not paragrafi:
        raise ValueError("Non ho trovato il testo del discorso in questa pagina.")
    return titolo, "\n\n".join(paragrafi)


# ----------------------------------------------------------------------------- YouTube
def id_video(url: str) -> str:
    u = urllib.parse.urlparse(url.strip())
    if u.hostname in ("youtu.be",):
        return u.path.lstrip("/").split("/")[0]
    if u.hostname and "youtube" in u.hostname:
        if u.path == "/watch":
            return urllib.parse.parse_qs(u.query).get("v", [""])[0]
        for pref in ("/live/", "/embed/", "/shorts/"):
            if u.path.startswith(pref):
                return u.path[len(pref):].split("/")[0]
    if re.fullmatch(r"[\w-]{11}", url.strip()):
        return url.strip()
    raise ValueError("Link YouTube non riconosciuto.")


def frasi_da_youtube(url: str) -> tuple[str, list[Frase]]:
    """Sottotitoli inglesi del video -> frasi con il tempo d'inizio.
    Se i sottotitoli sono automatici e senza punteggiatura, si spezza a blocchi di circa 30 parole."""
    from youtube_transcript_api import YouTubeTranscriptApi
    vid = id_video(url)
    trascr = YouTubeTranscriptApi().fetch(vid, languages=["en", "en-GB", "en-US"])
    pezzi = [(html.unescape(s.text).replace("\n", " ").strip(), float(s.start)) for s in trascr.snippets]
    pezzi = [(t, s) for t, s in pezzi if t and not re.fullmatch(r"\[.*\]", t)]   # "[Music]", "[Applause]"
    # testo unico con la posizione di inizio di ogni pezzo
    testo, inizi = "", []
    for t, s in pezzi:
        inizi.append((len(testo), s))
        testo += t + " "
    _punkt()
    frasi = nltk.sent_tokenize(testo)
    if frasi and np.mean([len(f.split()) for f in frasi]) > 60:      # niente punteggiatura
        parole, frasi, blocco = testo.split(), [], []
        for w in parole:
            blocco.append(w)
            if len(blocco) >= 30:
                frasi.append(" ".join(blocco))
                blocco = []
        if blocco:
            frasi.append(" ".join(blocco))
    out, cursore = [], 0
    pos = np.array([p for p, _ in inizi])
    for f in frasi:
        i = testo.find(f[:40], cursore)
        i = cursore if i < 0 else i
        cursore = i + len(f)
        k = max(int(np.searchsorted(pos, i, side="right")) - 1, 0)
        fr = _valuta(normalizza(f))
        fr.inizio = inizi[k][1]
        out.append(fr)
    return vid, out


# ----------------------------------------------------------------------------- riassunto LLM
def riassunto_llm(frasi_classificate: list[dict], punteggio: float, chiave: str,
                  modello: str = "kimi-k3.1-pro", base_url: str = "https://api.moonshot.ai/v1") -> str:
    """Riassunto finale in italiano (3-4 frasi). È l'unico punto in cui si usa l'LLM."""
    righe = [f"[{f['classe'].upper()}] {f['testo']}" for f in frasi_classificate if f["politica_monetaria"]]
    prompt = ("Sei un analista di politica monetaria. Qui sotto ci sono le frasi di politica monetaria di un "
              "discorso di un membro della BCE, ciascuna con l'etichetta data da un classificatore (FALCO = verso "
              f"tassi più alti, COLOMBA = verso tassi più bassi, NEUTRALE). Punteggio complessivo: {punteggio:+.2f} "
              "(da -1 tutto colomba a +1 tutto falco).\n\nScrivi in italiano un riassunto di 3-4 frasi: temi "
              "principali, quali passaggi spingono il tono verso falco o colomba, e se il messaggio complessivo è "
              "coerente con il punteggio. Non fare previsioni sui mercati.\n\n" + "\n".join(righe[:250]))
    corpo = {"model": modello, "reasoning_effort": "low", "max_completion_tokens": 4000,
             "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(corpo).encode(),
                                 headers={"Authorization": f"Bearer {chiave}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def chiave_kimi(secrets=None) -> str:
    try:
        if secrets is not None and "KIMI_API_KEY" in secrets:
            return str(secrets["KIMI_API_KEY"]).strip()
    except Exception:  # noqa: BLE001  (st.secrets solleva un errore se il file non esiste)
        pass
    return os.environ.get("KIMI_API_KEY", "").strip()
