"""
Prepara i discorsi BCE per Orange: una frase per riga, testo pulito,
con un indicatore delle frasi che parlano di politica monetaria.

Uso (dalla cartella progetto_BCE):
    python3 codice/prepara_frasi_bce.py

Richiede: pandas, nltk (con il modello "punkt_tab").
Produce in dati_preparati/:
    frasi_bce_politica_monetaria.csv  solo le frasi di politica monetaria (da caricare in Orange)
    frasi_bce_tutte.csv               tutte le frasi pulite, con la colonna politica_monetaria = 0/1
    report_preparazione.txt           quante frasi sono state scartate e perché
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "dati_grezzi" / "ecb_speeches_all.csv"
OUT_DIR = BASE / "dati_preparati"

MIN_PAROLE = 5      # sotto: frammenti ("2.", "Thank you.", titoletti)
MAX_PAROLE = 100    # sopra: ringraziamenti, elenchi, frasi unite per errore

MESI = ("January|February|March|April|May|June|July|August|"
        "September|October|November|December")
# "30 October 2026", "3-4 June 2020", "2 September 2014"
RE_DATA = re.compile(rf"\b\d{{1,2}}(?:\s?[-–]\s?\d{{1,2}})?\s(?:{MESI}),?\s(\d{{4}})")

# Fine paragrafo: 2+ spazi seguiti da maiuscola, cifra, "[" o virgolette.
# (Doppi spazi tra parole minuscole compaiono anche dentro le frasi: lì non si taglia.)
RE_PARAGRAFO = re.compile(r" {2,}(?=[A-Z\[\d“\"(‘'])")
RE_NOTA_INLINE = re.compile(r"\[\s*\d{1,3}\s*\]")
RE_NOTA_INIZIO = re.compile(r"^\[\s*\d{1,3}\s*\]")

# Frasi che sono note bibliografiche o riferimenti, non testo del discorso.
RE_RIFERIMENTO = re.compile(
    r"\bet al\.|\bibid\b|\bop\. cit\.|Working Paper|Discussion Paper|Occasional Paper|"
    r"Policy Brief|Economic Bulletin|Statistics Paper|Journal of|\bVol\. ?\d|\bpp?\. ?\d|"
    r"\bNo\.? ?\d{2,}|https?://|www\.|\(OJ L|ECB/\d{4}/|"
    r"Monthly Bulletin|Press Release|University Press|"
    r"\(\d{4}[a-z]?\)\s*[,:.]?\s*[“\"]|"                      # "(2018), “Titolo…”"
    r"[A-Z][a-zà-ü]+, [A-Z]\.(?:[A-Z]\.)*(?:,| and)",          # "Bena, J., Bian, B."
)
RE_INIZIO_RIFERIMENTO = re.compile(
    r"^(?:See\b|Cf\.|For (?:a|an|more|further) \w+,? see\b|Sources?:|Notes?\b|"
    r"\(\d{4}|(?:Chart|Figure|Table|Slide)s? ?\d|"
    r"[A-Z][\w.,' -]{0,60}\(\d{4}[a-z]?\)[,:]\s+[A-Z‘“\"'])")   # "Issing (2003b), Introductory…"

# Parole chiave di politica monetaria (confronto senza distinzione di maiuscole).
# Forti: bastano da sole per tenere la frase.
PAROLE_CHIAVE = [
    r"inflation\w*", r"disinflation\w*", r"deflation\w*", r"price stability", r"price pressures?",
    r"monetary policy", r"policy stance", r"monetary stance",
    r"interest rates?", r"policy rates?", r"deposit facility", r"main refinancing",
    r"rate (?:hikes?|increases?|cuts?|reductions?)", r"tightening",
    r"(?:monetary|policy) easing", r"easing cycle", r"accommodative", r"(?:monetary|policy) accommodation",
    r"sufficiently restrictive", r"restrictive (?:territory|stance|policy|monetary)",
    r"neutral rate", r"natural rate", r"r-star",
    r"asset purchases?", r"quantitative easing", r"forward guidance",
    r"meeting-by-meeting", r"data-dependen\w+",
    r"(?:monetary policy|policy) transmission", r"transmission (?:mechanism|of (?:our |monetary )?policy)",
    r"financing conditions", r"wage growth", r"wage pressures?",
]
# Deboli: compaiono anche in discorsi su pagamenti, vigilanza, geopolitica.
# Si registrano solo se nella frase c'è già una parola forte.
PAROLE_DEBOLI = [r"wages?", r"fragmentation", r"transmission", r"easing", r"restrictive",
                 r"accommodation", r"hik(?:e|es|ed|ing)", r"governing council"]
RE_DEBOLI = re.compile(r"\b(?:" + "|".join(PAROLE_DEBOLI) + r")\b", re.IGNORECASE)
# Sigle: confronto con maiuscole esatte
SIGLE = ["APP", "PEPP", "TLTRO", "LTRO", "TPI", "OMT", "HICP", "€STR"]

RE_PC = re.compile(r"\b(?:" + "|".join(PAROLE_CHIAVE) + r")\b", re.IGNORECASE)
RE_SIGLE = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(s) for s in SIGLE) + r")s?(?![\w-])")


def _regex_flessibile(s: str) -> re.Pattern:
    """Regex che trova la stringa s anche se gli spazi nel testo sono diversi."""
    return re.compile(r"\s*".join(re.escape(p) for p in s.split()))


def _taglia_data(testo: str, anno: int, entro: int) -> str:
    for m in RE_DATA.finditer(testo[:entro]):
        if abs(int(m.group(1)) - anno) <= 1:
            return testo[m.end():]
    return testo


def togli_intestazione(testo: str, titolo: str, sottotitolo: str, anno: int) -> str:
    """Toglie 'SPEECH', titolo, sottotitolo, luogo e data all'inizio del discorso.
    Il testo inizia con: [SPEECH] titolo  sottotitolo  luogo, data  corpo."""
    testo = re.sub(r"^\s*SPEECH\s+", "", testo)
    # 1) se il sottotitolo compare all'inizio, si taglia dopo di esso e poi dopo luogo e data
    if isinstance(sottotitolo, str) and len(sottotitolo.split()) >= 3:
        m = _regex_flessibile(sottotitolo).search(testo[:2500])
        if m:
            return _taglia_data(testo[m.end():], anno, 150).strip()
    # 2) altrimenti si toglie il titolo e si taglia dopo la prima data dell'anno del discorso
    if isinstance(titolo, str) and titolo.strip():
        m = _regex_flessibile(titolo).search(testo[:600])
        if m and m.start() < 50:
            testo = testo[m.end():]
    return _taglia_data(testo, anno, 900).strip()


# Inglese o no: quota di parole funzionali inglesi nelle prime 5.000 battute del discorso
PAROLE_EN = set("the of and to in is that for it as with be on are this by have not we our has "
                "which at from will".split())


def lingua(testo: str) -> str:
    w = re.findall(r"[a-zA-Z]+", testo[:5000].lower())
    return "en" if sum(x in PAROLE_EN for x in w) / max(len(w), 1) >= 0.15 else "altra"


# Righe di intestazione rimaste e titoletti senza punteggiatura finale
RE_RIGA_INTESTAZIONE = re.compile(
    r"\b(?:[Ss]peech|[Kk]eynote|[Rr]emarks|[Ll]ecture|[Ss]tatement|[Aa]ddress|[Pp]resentation)"
    r"(?: delivered| given)? by\b|^\[Updated on")
RE_FINE_FRASE = re.compile(r"[.!?:;”\"’')\]]$")


def togli_coda(testo: str) -> str:
    """Toglie la sezione 'SEE ALSO' (link a contenuti collegati) in fondo alla pagina."""
    i = testo.find("SEE ALSO")
    return testo[:i] if i >= 0 else testo


def normalizza(s: str) -> str:
    s = RE_NOTA_INLINE.sub("", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)   # spazio rimasto prima della punteggiatura dopo "[1]"
    return s


def parole_chiave(frase: str) -> list[str]:
    """Parole chiave trovate. Le parole 'deboli' contano solo se c'è anche una parola forte."""
    forti = {m.group(0).lower() for m in RE_PC.finditer(frase)}
    forti |= {m.group(0) for m in RE_SIGLE.finditer(frase)}
    if not forti:
        return []
    deboli = {m.group(0).lower() for m in RE_DEBOLI.finditer(frase)}
    deboli = {d for d in deboli if not any(d in f for f in forti)}   # niente doppioni
    return sorted(forti | deboli)


def scrivi_tab_orange(df: pd.DataFrame, path: Path) -> None:
    """File .tab con l'intestazione a 3 righe di Orange (nome / tipo / ruolo).
    Tutte le colonne sono meta: così ID, anno e lunghezza non finiscono tra le variabili
    del modello quando si aggiungono Bag of Words o Document Embedding.
    'include=True' dice al widget Corpus che il testo da analizzare è la colonna 'frase'."""
    tipi = {"id_frase": "c", "id_discorso": "c", "posizione": "c", "data": "time", "anno": "c",
            "trimestre": "d", "oratore": "d", "titolo": "s", "n_parole": "c",
            "parole_chiave": "s", "frase": "s"}
    ruoli = {c: "meta" for c in df.columns}
    ruoli["frase"] = "meta include=True"
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Orange legge il .tab come CSV con separatore tab: i campi con virgolette vanno racchiusi
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(df.columns)
        w.writerow([tipi[c] for c in df.columns])
        w.writerow([ruoli[c] for c in df.columns])
        w.writerows(df.itertuples(index=False))


def main() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    csv.field_size_limit(sys.maxsize)
    # QUOTE_NONE: nel file le virgolette non delimitano i campi (il codice del README fallisce senza)
    df = pd.read_csv(INPUT, sep="|", engine="python", quoting=csv.QUOTE_NONE, dtype=str)
    n_discorsi_tot = len(df)
    df = df[df.contents.notna() & (df.contents.str.strip() != "")].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["id_discorso"] = range(1, len(df) + 1)

    scarti = Counter()
    righe = []
    for d in df.itertuples(index=False):
        lng = lingua(d.contents)
        corpo = togli_coda(togli_intestazione(d.contents, d.title, d.subtitle, d.date.year))
        pos = 0
        visti = set()
        for blocco in RE_PARAGRAFO.split(corpo):
            blocco = blocco.strip()
            if not blocco:
                continue
            if RE_NOTA_INIZIO.match(blocco):           # nota a piè di pagina "[12] ..."
                scarti["nota a piè di pagina"] += len(nltk.sent_tokenize(blocco))
                continue
            for frase in nltk.sent_tokenize(normalizza(blocco)):
                n = len(frase.split())
                if n < MIN_PAROLE or not re.search(r"[A-Za-z]{3}", frase):
                    scarti[f"meno di {MIN_PAROLE} parole"] += 1
                    continue
                if n > MAX_PAROLE:
                    scarti[f"più di {MAX_PAROLE} parole"] += 1
                    continue
                if RE_RIFERIMENTO.search(frase) or RE_INIZIO_RIFERIMENTO.match(frase):
                    scarti["riferimento bibliografico"] += 1
                    continue
                if pos < 2 and RE_RIGA_INTESTAZIONE.search(frase):
                    scarti["resto dell'intestazione"] += 1
                    continue
                if n <= 15 and not RE_FINE_FRASE.search(frase):
                    scarti["titoletto senza punteggiatura finale"] += 1
                    continue
                if frase in visti:
                    scarti["duplicata nello stesso discorso"] += 1
                    continue
                visti.add(frase)
                pos += 1
                pc = parole_chiave(frase) if lng == "en" else []
                righe.append({
                    "lingua": lng,
                    "id_discorso": d.id_discorso,
                    "posizione": pos,
                    "data": d.date.strftime("%Y-%m-%d"),
                    "anno": d.date.year,
                    "trimestre": f"{d.date.year}-T{(d.date.month - 1) // 3 + 1}",
                    "oratore": d.speakers,
                    "titolo": normalizza(d.title or ""),
                    "n_parole": n,
                    "politica_monetaria": int(bool(pc)),
                    "parole_chiave": "; ".join(pc),
                    "frase": frase,
                })

    out = pd.DataFrame(righe)
    out.insert(0, "id_frase", range(1, len(out) + 1))
    # Il file per Orange contiene solo frasi inglesi di politica monetaria
    pm = out[out.politica_monetaria == 1].drop(columns=["politica_monetaria", "lingua"])
    n_non_en = out.loc[out.lingua != "en", "id_discorso"].nunique()

    OUT_DIR.mkdir(exist_ok=True)
    kw = dict(index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    pm.to_csv(OUT_DIR / "frasi_bce_politica_monetaria.csv", **kw)
    out.to_csv(OUT_DIR / "frasi_bce_tutte.csv", **kw)
    scrivi_tab_orange(pm, OUT_DIR / "frasi_bce_politica_monetaria.tab")

    # Report
    per_trim = pm[pm.anno >= 2018].groupby("trimestre").size()
    oratori = pm.groupby("oratore").size().sort_values(ascending=False)
    righe_rep = [
        f"Discorsi nel file: {n_discorsi_tot}; con testo: {len(df)}",
        f"Discorsi non in inglese (esclusi dal file di politica monetaria): {n_non_en}",
        f"Frasi tenute: {len(out)}; di politica monetaria: {len(pm)} ({len(pm) / len(out):.1%})",
        "",
        "Frasi scartate:",
        *[f"  {k}: {v}" for k, v in scarti.most_common()],
        "",
        f"Frasi di politica monetaria per trimestre dal 2018: min {per_trim.min()}, "
        f"mediana {per_trim.median():.0f}, max {per_trim.max()} ({len(per_trim)} trimestri)",
        "",
        "Frasi di politica monetaria per oratore:",
        *[f"  {k}: {v}" for k, v in oratori.items()],
        "",
        "Parole chiave più frequenti:",
        *[f"  {k}: {v}" for k, v in Counter(
            p for s in pm.parole_chiave for p in s.split("; ")).most_common(25)],
    ]
    (OUT_DIR / "report_preparazione.txt").write_text("\n".join(righe_rep) + "\n", encoding="utf-8")
    print("\n".join(righe_rep[:12]))


if __name__ == "__main__":
    main()
