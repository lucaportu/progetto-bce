"""
Accordo tra annotatori (kappa di Cohen) ed etichette finali del campione BCE.

Uso (dalla cartella progetto_BCE), quando i due file sono compilati:
    python3 codice/calcola_accordo.py

1a esecuzione: controlla che i file siano completi, calcola il kappa sul test e scrive
   annotazione/disaccordi_da_risolvere.xlsx  → le frasi del test su cui gli annotatori non sono d'accordo.
   Discutetele insieme e scrivete la decisione nella colonna "etichetta_finale".
2a esecuzione (con i disaccordi risolti): scrive le etichette finali
   annotazione/etichette_test_bce.csv   test set per valutare A, B, C: stesse colonne del file Fed
                                        (sentence, year, fonte, label, label_nome) + strato, periodo, peso, ...
                                        Le frasi "irrilevante" sono tolte, come nelle frasi WCB di addestramento.
   annotazione/report_accordo.txt
Il kappa è calcolato sulle 4 etichette (falco, colomba, neutrale, irrilevante).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "annotazione"   # cartella alternativa: per prove
CODICI = {"colomba": 0, "falco": 1, "neutrale": 2}     # come nel dataset Fed
VALIDE = list(CODICI) + ["irrilevante"]
NOMI_FED = {0: "dovish (colomba)", 1: "hawkish (falco)", 2: "neutral (neutrale)"}
ORDINE = ["falco", "colomba", "neutrale", "irrilevante"]


def leggi_annotatore(a: str) -> pd.DataFrame:
    df = pd.read_excel(DIR / f"annotatore_{a}.xlsx", sheet_name="Frasi", dtype=str)
    df["etichetta"] = df["etichetta"].str.strip().str.lower()
    for c in ("incerto", "da_scartare"):
        df[c] = df[c].fillna("").str.strip() != ""
    errate = df[df.etichetta.notna() & ~df.etichetta.isin(VALIDE)]
    if len(errate):
        sys.exit(f"Annotatore {a}: etichette non valide in {list(errate.id_campione)}")
    vuote = df[df.etichetta.isna() & ~df.da_scartare]
    if len(vuote):
        sys.exit(f"Annotatore {a}: {len(vuote)} frasi senza etichetta (es. {list(vuote.id_campione[:5])})")
    return df.set_index("id_campione")[["etichetta", "incerto", "da_scartare", "note"]]


def kappa_ic(y1, y2, n_boot=2000, seed=0):
    """Kappa di Cohen con intervallo di confidenza bootstrap al 95%."""
    y1, y2 = np.asarray(y1), np.asarray(y2)
    k = cohen_kappa_score(y1, y2)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y1), len(y1))
        if len(set(y1[i]) | set(y2[i])) > 1:
            boot.append(cohen_kappa_score(y1[i], y2[i]))
    return k, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def formato_fed(df: pd.DataFrame) -> pd.DataFrame:
    """Stesse colonne di dati_fed/frasi_fed_etichettate.csv, così i due file si possono unire."""
    out = pd.DataFrame({
        "sentence": df.frase,
        "year": pd.to_datetime(df.data).dt.year,
        "fonte": "discorso BCE",
        "label": df.label,
        "label_nome": df.label.map(NOMI_FED),
    })
    return out


def main() -> None:
    chiave = pd.read_csv(DIR / "chiave_campione.csv").set_index("id_campione")
    a1, a2 = leggi_annotatore("1"), leggi_annotatore("2")
    rep = []

    # --- Test: entrambi gli annotatori ---------------------------------------------------------
    test = chiave.join(a1.add_suffix("_1")).join(a2.add_suffix("_2"))
    scartate = test.da_scartare_1 | test.da_scartare_2
    t = test[~scartate]
    k, lo, hi = kappa_ic(t.etichetta_1, t.etichetta_2)
    accordo = (t.etichetta_1 == t.etichetta_2).mean()
    cm = pd.DataFrame(confusion_matrix(t.etichetta_1, t.etichetta_2, labels=ORDINE),
                      index=[f"ann.1 {x}" for x in ORDINE], columns=[f"ann.2 {x}" for x in ORDINE])
    rep += [
        f"TEST: {len(test)} frasi, scartate da almeno un annotatore: {int(scartate.sum())}, usate: {len(t)}",
        f"Accordo osservato: {accordo:.1%}",
        f"Kappa di Cohen: {k:.3f}  (IC 95% bootstrap: {lo:.3f} – {hi:.3f})",
        "  riferimento (Landis e Koch): 0,41–0,60 moderato, 0,61–0,80 buono, >0,80 molto buono",
        "",
        "Matrice di confusione tra annotatori:",
        cm.to_string(),
        "",
        "Kappa per periodo:",
    ]
    for p, g in t.groupby("periodo"):
        if g.etichetta_1.nunique() > 1 or g.etichetta_2.nunique() > 1:
            rep.append(f"  {p}: {cohen_kappa_score(g.etichetta_1, g.etichetta_2):.3f} (n={len(g)})")

    # --- Disaccordi da risolvere -----------------------------------------------------------------
    file_dis = DIR / "disaccordi_da_risolvere.xlsx"
    dis = t[t.etichetta_1 != t.etichetta_2]
    if not file_dis.exists():
        out = dis[["frase", "etichetta_1", "note_1", "etichetta_2", "note_2"]].assign(etichetta_finale="")
        out.to_excel(file_dis)
        rep += ["", f"Disaccordi: {len(dis)} → scritti in {file_dis.name}.",
                "Scrivete la decisione comune in 'etichetta_finale' e rilanciate lo script."]
        fine(rep)
        return
    risolti = pd.read_excel(file_dis, index_col=0, dtype=str)
    risolti["etichetta_finale"] = risolti.etichetta_finale.str.strip().str.lower()
    mancanti = [i for i in dis.index if i not in risolti.index or risolti.etichetta_finale.get(i) not in VALIDE]
    if mancanti:
        rep += ["", f"Disaccordi ancora da risolvere in {file_dis.name}: {len(mancanti)} (es. {mancanti[:5]})"]
        fine(rep)
        return

    # --- Etichette finali --------------------------------------------------------------------------
    t = t.copy()
    t["etichetta"] = t.etichetta_1
    t.loc[dis.index, "etichetta"] = risolti.loc[dis.index, "etichetta_finale"]
    irr = t[t.etichetta == "irrilevante"]
    t = t[t.etichetta != "irrilevante"].copy()
    t["label"] = t.etichetta.map(CODICI)
    test_out = formato_fed(t).assign(
        id_campione=t.index, strato=t.strato, periodo=t.periodo, peso=t.peso,
        oratore=t.oratore, data=t.data, id_frase=t.id_frase)
    test_out.to_csv(DIR / "etichette_test_bce.csv", index=False, encoding="utf-8")

    def distr(s):
        return ", ".join(f"{x} {int((s == x).sum())}" for x in ORDINE)
    rep += ["", f"Etichette finali test ({len(t)} frasi, tolte {len(irr)} irrilevanti): {distr(t.etichetta)}",
            "Scritto: etichette_test_bce.csv"]
    fine(rep)


def fine(rep: list[str]) -> None:
    testo = "\n".join(rep) + "\n"
    (DIR / "report_accordo.txt").write_text(testo, encoding="utf-8")
    print(testo)


if __name__ == "__main__":
    main()
