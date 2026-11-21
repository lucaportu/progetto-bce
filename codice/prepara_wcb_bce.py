"""Frasi BCE etichettate del dataset World Central Banks (WCB) → formato del file Fed.

Fonte: Shah et al. (2025), "Words That Unite The World", https://arxiv.org/abs/2505.17048
Dati:  https://huggingface.co/datasets/gtfintechlab/european_central_bank  (CC BY-NC-SA 4.0)

Legge i parquet in dati_bce_wcb/originali/ (3 semi, ognuno con train/val/test delle STESSE 1.000 frasi)
e scrive:
  dati_bce_wcb/frasi_bce_wcb_addestramento.csv  frasi 2015–2022 → addestramento del modello B (con la Fed)
  dati_bce_wcb/frasi_bce_wcb_test.csv           frasi 2023–2024 → test comune per A, B e C
  dati_bce_wcb/frasi_bce_wcb_etichettate.csv    tutte (2015–2024): per riaddestrare il modello scelto alla fine
  I tre file hanno le stesse colonne di dati_fed/frasi_fed_etichettate.csv e sono senza le frasi "irrelevant".
  dati_bce_wcb/frasi_bce_wcb_completo.csv       tutte le 1.000 frasi con le etichette originali, le altre due
                                                etichette (temporale, incertezza) e lo split di ogni seme
Uso: python3 codice/prepara_wcb_bce.py

La divisione è per ANNO e non quella casuale degli autori: con la divisione casuale frasi dello stesso resoconto
finiscono sia in addestramento sia in test e il risultato esce ottimistico. Così il test riguarda anni che il
modello non ha visto (2023–2024: picco dei tassi e inizio dei tagli).
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ORIG = BASE / "dati_bce_wcb" / "originali"
OUT = BASE / "dati_bce_wcb"
SEMI = ["5768", "78516", "944601"]
ULTIMO_ANNO_ADDESTRAMENTO = 2022
CODICI = {"dovish": (0, "dovish (colomba)"), "hawkish": (1, "hawkish (falco)"),
          "neutral": (2, "neutral (neutrale)")}

tabelle = {}
for seme in SEMI:
    parti = [pd.read_parquet(ORIG / seme / f"{sp}.parquet").assign(split=sp) for sp in ("train", "val", "test")]
    tabelle[seme] = pd.concat(parti, ignore_index=True)

base = tabelle[SEMI[0]].drop(columns=["split", "__index_level_0__"])
for seme in SEMI:
    t = tabelle[seme]
    assert len(t) == 1000 and t.sentences.is_unique, seme
    assert set(t.sentences) == set(base.sentences), f"il seme {seme} ha frasi diverse"
    # le etichette devono essere identiche tra i semi
    m = base.merge(t[["sentences", "stance_label"]], on="sentences", suffixes=("", "_x"))
    assert (m.stance_label == m.stance_label_x).all(), seme
    base = base.merge(t[["sentences", "split"]].rename(columns={"split": f"split_seme_{seme}"}), on="sentences")

completo = base.rename(columns={"sentences": "frase", "stance_label": "orientamento",
                                "time_label": "temporale", "certain_label": "incertezza", "year": "anno"})
completo = completo.sort_values(["anno", "frase"]).reset_index(drop=True)
completo.to_csv(OUT / "frasi_bce_wcb_completo.csv", index=False)

fed_formato = completo[completo.orientamento != "irrelevant"].copy()
fed_formato["label"] = fed_formato.orientamento.map(lambda x: CODICI[x][0])
fed_formato["label_nome"] = fed_formato.orientamento.map(lambda x: CODICI[x][1])
fed_formato["fonte"] = "resoconto riunione BCE"
fed_formato = fed_formato.rename(columns={"frase": "sentence", "anno": "year"})
fed_formato = fed_formato[["sentence", "year", "fonte", "label", "label_nome"]]
fed_formato.to_csv(OUT / "frasi_bce_wcb_etichettate.csv", index=False)
add = fed_formato[fed_formato.year <= ULTIMO_ANNO_ADDESTRAMENTO]
test = fed_formato[fed_formato.year > ULTIMO_ANNO_ADDESTRAMENTO]
add.to_csv(OUT / "frasi_bce_wcb_addestramento.csv", index=False)
test.to_csv(OUT / "frasi_bce_wcb_test.csv", index=False)

print(f"completo: {len(completo)} frasi; formato Fed: {len(fed_formato)} (tolte "
      f"{(completo.orientamento == 'irrelevant').sum()} irrilevanti)")
print(fed_formato.label_nome.value_counts().to_string())
print("anni:", fed_formato.year.min(), "–", fed_formato.year.max())
for nome, d in (("addestramento", add), ("test", test)):
    print(f"{nome} {d.year.min()}–{d.year.max()}: {len(d)} frasi,",
          ", ".join(f"{k} {v}" for k, v in d.label_nome.value_counts().items()))
