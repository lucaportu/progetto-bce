"""File .tab pronti per i workflow Orange (cartella orange/).

Uso (dalla cartella progetto_BCE, dopo prepara_wcb_bce.py):
    python3 codice/prepara_file_orange.py

1) Frasi etichettate Fed e WCB in formato .tab, con l'intestazione di Orange già impostata:
   testo = sentence (meta, include=True), variabile da prevedere = label_nome, tutto il resto meta.
   Così il widget Corpus le carica pronte, senza Select Columns.
     dati_fed/frasi_fed_etichettate.tab
     dati_bce_wcb/frasi_bce_wcb_addestramento.tab   (2015–2022)
     dati_bce_wcb/frasi_bce_wcb_test.tab            (2023–2024)
     dati_bce_wcb/frasi_bce_wcb_etichettate.tab     (tutte, per il modello finale)

2) Tassi BCE per trimestre, per la verifica dell'indice:
     dati_preparati/tassi_bce_trimestrali.tab
   Una riga per trimestre dal 2018 (stesso formato di 'trimestre' nelle frasi, es. 2022-T3):
     tasso_DFR_fine           tasso sui depositi a fine trimestre
     variazione_t             variazione del tasso decisa nel trimestre (punti percentuali)
     variazione_t+1/+2/+3     variazione decisa 1, 2, 3 trimestri dopo
     decisione_t+1            rialzo / taglio / fermo nel trimestre successivo
     regola_banale_t+1        previsione "come l'ultima decisione presa" (rialzo/taglio se l'ultima variazione
                              fino a quel trimestre era un rialzo/taglio, altrimenti fermo)
   La data della decisione è stimata come entrata in vigore − 6 giorni (riunione del giovedì, tasso in vigore
   il mercoledì successivo). L'ultimo trimestre osservato per intero è il 2026-T3: le variazioni successive
   sono lasciate vuote.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ULTIMO_TRIMESTRE_COMPLETO = pd.Period("2026Q3")


def scrivi_tab(df: pd.DataFrame, path: Path, tipi: dict, ruoli: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(df.columns)
        w.writerow([tipi[c] for c in df.columns])
        w.writerow([ruoli.get(c, "") for c in df.columns])
        w.writerows(df.itertuples(index=False))


def frasi_etichettate() -> None:
    tipi = {"sentence": "s", "year": "c", "fonte": "d", "label": "c", "label_nome": "d"}
    ruoli = {"sentence": "meta include=True", "year": "meta", "fonte": "meta", "label": "meta",
             "label_nome": "class"}
    for csv_path in [BASE / "dati_fed/frasi_fed_etichettate.csv",
                     BASE / "dati_bce_wcb/frasi_bce_wcb_addestramento.csv",
                     BASE / "dati_bce_wcb/frasi_bce_wcb_test.csv",
                     BASE / "dati_bce_wcb/frasi_bce_wcb_etichettate.csv"]:
        df = pd.read_csv(csv_path)[["sentence", "year", "fonte", "label", "label_nome"]]
        df["sentence"] = df.sentence.str.replace(r"\s+", " ", regex=True).str.strip()
        scrivi_tab(df, csv_path.with_suffix(".tab"), tipi, ruoli)
        print(f"{csv_path.with_suffix('.tab').relative_to(BASE)}: {len(df)} frasi")


def tassi_trimestrali() -> None:
    t = pd.read_csv(BASE / "dati_grezzi/tassi_bce.csv", parse_dates=["data_entrata_in_vigore"])
    t["decisione"] = t.data_entrata_in_vigore - pd.Timedelta(days=6)
    t["trim"] = t.decisione.dt.to_period("Q")
    trimestri = pd.period_range("2017Q1", "2026Q4", freq="Q")
    var = t.dropna(subset=["variazione_DFR"]).groupby("trim").variazione_DFR.sum().reindex(trimestri, fill_value=0.0)
    var[var.index > ULTIMO_TRIMESTRE_COMPLETO] = np.nan
    livello = []
    for q in trimestri:
        fine = q.end_time
        prima = t[t.data_entrata_in_vigore <= fine]
        livello.append(prima.tasso_depositi_DFR.iloc[-1] if q <= ULTIMO_TRIMESTRE_COMPLETO else np.nan)
    df = pd.DataFrame({"q": trimestri, "tasso_DFR_fine": livello, "variazione_t": var.values})
    for k in (1, 2, 3):
        df[f"variazione_t+{k}"] = df.variazione_t.shift(-k)

    def segno(x):
        return np.nan if pd.isna(x) else ("rialzo" if x > 0 else "taglio" if x < 0 else "fermo")
    df["decisione_t+1"] = df["variazione_t+1"].map(segno)
    # regola banale: la prossima decisione sarà come l'ultima variazione avvenuta (fino al trimestre t)
    ultima, regola = None, []
    for v in df.variazione_t:
        if pd.notna(v) and v != 0:
            ultima = "rialzo" if v > 0 else "taglio"
        regola.append(ultima if ultima else "fermo")
    # "come l'ultima" ha senso solo se il ciclo è in corso: dopo più di 2 trimestri fermi si prevede fermo
    fermi = 0
    for i, v in enumerate(df.variazione_t):
        fermi = fermi + 1 if v == 0 else 0
        if fermi > 2:
            regola[i] = "fermo"
    df["regola_banale_t+1"] = regola
    df["trimestre"] = df.q.map(lambda q: f"{q.year}-T{q.quarter}")
    df["anno_decimale"] = df.q.map(lambda q: q.year + (q.quarter - 1) / 4 + 0.125)
    df = df[df.q >= pd.Period("2018Q1")]
    df = df[["trimestre", "anno_decimale", "tasso_DFR_fine", "variazione_t", "variazione_t+1", "variazione_t+2",
             "variazione_t+3", "decisione_t+1", "regola_banale_t+1"]]
    tipi = {c: "c" for c in df.columns}
    tipi.update({"trimestre": "d", "decisione_t+1": "d", "regola_banale_t+1": "d"})
    ruoli = {"trimestre": "meta", "decisione_t+1": "meta", "regola_banale_t+1": "meta"}
    out = BASE / "dati_preparati/tassi_bce_trimestrali.tab"
    scrivi_tab(df.astype(object).where(df.notna(), ""), out, tipi, ruoli)
    print(f"{out.relative_to(BASE)}: {len(df)} trimestri")


if __name__ == "__main__":
    frasi_etichettate()
    tassi_trimestrali()
