"""Punto 1 del piano (README, sezione 8): il modello finale di Orange ricostruito in Python, sul computer.

Uso (dalla cartella progetto_BCE, con l'ambiente del progetto):
    .venv/bin/python codice/modello_locale.py

Cosa fa
1. Embedding: lo stesso SBERT che usa Orange (paraphrase-multilingual-MiniLM-L12-v2, 384 numeri per frase),
   calcolato in locale con sentence-transformers. Verificato: identico a quello del server di Orange
   (coseno 1,0000 sulle 244 frasi di test).
2. Rete: come in Orange. Standardizzazione (media 0, deviazione standard 1: è il preprocessing che Orange applica
   da solo alla rete neurale) + MLP con 100 neuroni, ReLU, Adam, alpha = 1, 500 iterazioni, seme 1.
3. Verifica sul test WCB 2023-24 di A, B e B2 (stessi numeri di Orange?). Si prosegue solo se B è vicino a 0,633.
4. Modello finale: Fed + tutte le 976 frasi WCB  ->  modello/rete_finale.joblib
5. Classifica le frasi dei discorsi dal 2018 (con le probabilità) e ricalcola l'indice per trimestre e per discorso:
       risultati/frasi_discorsi_classificate.csv
       risultati/indice_trimestre_locale.csv      (confrontato con l'indice di Orange)
       risultati/indice_discorsi.csv              (serve per la classe relativa di un nuovo discorso)

Gli embedding vengono salvati in modello/cache/: rilanciando lo script non si ricalcolano.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
MODELLO_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSI = ["dovish (colomba)", "hawkish (falco)", "neutral (neutrale)"]   # stesso ordine di Orange
DAL = "2018-01-01"
F1_ORANGE = {"A": 0.592, "B": 0.633, "B2": 0.611}                       # README, sezione 7
CACHE = BASE / "modello" / "cache"

_sbert = None


def embedding(testi, nome):
    """Embedding SBERT, con cache su disco (il nome del file dipende dal contenuto)."""
    global _sbert
    chiave = hashlib.md5(("\n".join(testi) + MODELLO_SBERT).encode()).hexdigest()[:12]
    f = CACHE / f"{nome}_{chiave}.npy"
    if f.exists():
        return np.load(f)
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(MODELLO_SBERT, device="cpu")
    t0 = time.time()
    X = _sbert.encode(list(testi), batch_size=64, normalize_embeddings=False, show_progress_bar=len(testi) > 2000)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, X)
    print(f"  embedding {nome}: {len(testi)} frasi in {time.time() - t0:.0f} s")
    return X


def rete():
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=1.0,
                                       max_iter=500, random_state=1))


def leggi(percorso):
    d = pd.read_csv(BASE / percorso)
    return d.sentence.astype(str).tolist(), d.label_nome.map(CLASSI.index).to_numpy()


def main():
    fed_t, fed_y = leggi("dati_fed/frasi_fed_etichettate.csv")
    tr_t, tr_y = leggi("dati_bce_wcb/frasi_bce_wcb_addestramento.csv")
    te_t, te_y = leggi("dati_bce_wcb/frasi_bce_wcb_test.csv")
    tutte_t, tutte_y = leggi("dati_bce_wcb/frasi_bce_wcb_etichettate.csv")
    X_fed, X_tr = embedding(fed_t, "fed"), embedding(tr_t, "wcb_addestramento")
    X_te, X_tutte = embedding(te_t, "wcb_test"), embedding(tutte_t, "wcb_tutte")

    # ------------------------------------------------------------------ 3. verifica sul test
    # La rete parte da pesi casuali: cambiando il seme (o anche solo float32/float64) l'F1 sul test cambia di
    # qualche centesimo. Per questo il confronto con Orange (un solo addestramento) si fa contro la variabilità
    # su 10 semi, non contro un singolo numero.
    print("\nVerifica sul test WCB 2023-24 (244 frasi), rete alpha=1 (seme 1 e variabilità su 10 semi):")
    approcci = {"A": (X_fed, fed_y), "B": (np.vstack([X_fed, X_tr]), np.r_[fed_y, tr_y]), "B2": (X_tr, tr_y)}
    verifica = {}
    for nome, (X, y) in approcci.items():
        p = rete().fit(X, y).predict(X_te)
        f1 = f1_score(te_y, p, average="weighted")
        semi = []
        for s in range(1, 11):
            r = rete()
            r.set_params(mlpclassifier__random_state=s)
            semi.append(f1_score(te_y, r.fit(X, y).predict(X_te), average="weighted"))
        verifica[nome] = {"seme_1": round(f1, 3), "media_10_semi": round(float(np.mean(semi)), 3),
                          "dev_st_10_semi": round(float(np.std(semi)), 3), "orange": F1_ORANGE[nome]}
        print(f"  {nome:<3} F1 pesato seme 1 {f1:.3f} | 10 semi {np.mean(semi):.3f} ± {np.std(semi):.3f} "
              f"(min {min(semi):.3f}, max {max(semi):.3f}) | Orange {F1_ORANGE[nome]:.3f} | "
              f"accuratezza {accuracy_score(te_y, p):.3f} | kappa {cohen_kappa_score(te_y, p):.3f}")
        if nome == "B":
            print("      matrice seme 1 (righe esperti, colonne modello; colomba/falco/neutrale):",
                  confusion_matrix(te_y, p).tolist())
            verifica[nome].update({
                "matrice_seme_1": confusion_matrix(te_y, p).tolist(),
                "accuratezza_seme_1": round(accuracy_score(te_y, p), 3),
                "kappa_seme_1": round(cohen_kappa_score(te_y, p), 3),
                "f1_per_classe_seme_1": [round(v, 3) for v in f1_score(te_y, p, average=None)]})
    vb = verifica["B"]
    if abs(vb["orange"] - vb["media_10_semi"]) > 2.5 * max(vb["dev_st_10_semi"], 0.005):
        sys.exit("STOP: il risultato di Orange è fuori dalla variabilità del modello locale. Non salvo nulla.")
    print("  OK: il valore di Orange rientra nella variabilità del modello locale.")

    # ------------------------------------------------------------------ 4. modello finale
    finale = rete().fit(np.vstack([X_fed, X_tutte]), np.r_[fed_y, tutte_y])
    (BASE / "modello").mkdir(exist_ok=True)
    joblib.dump(finale, BASE / "modello/rete_finale.joblib")
    info = {"modello_embedding": MODELLO_SBERT, "classi": CLASSI, "rete": "StandardScaler + MLPClassifier(100, relu, "
            "adam, alpha=1, max_iter=500, random_state=1)", "addestramento": "Fed 2.339 + WCB 976 frasi",
            "verifica_test_F1_pesato": verifica, "creato": time.strftime("%Y-%m-%d %H:%M")}
    (BASE / "modello/rete_finale.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("\nModello finale salvato in modello/rete_finale.joblib (+ rete_finale.json)")

    # ------------------------------------------------------------------ 5. frasi dei discorsi
    d = pd.read_csv(BASE / "dati_preparati/frasi_bce_politica_monetaria.csv")
    d = d[d.data >= DAL].reset_index(drop=True)
    print(f"\nFrasi dei discorsi dal {DAL}: {len(d)}")
    P = finale.predict_proba(embedding(d.frase.astype(str).tolist(), "discorsi_dal_2018"))
    d["p_colomba"], d["p_falco"], d["p_neutrale"] = P[:, 0], P[:, 1], P[:, 2]
    d["classe"] = np.array(["colomba", "falco", "neutrale"])[P.argmax(1)]
    d["punteggio"] = np.select([d.classe == "falco", d.classe == "colomba"], [1, -1], 0)   # quello dell'indice
    d["punteggio_continuo"] = d.p_falco - d.p_colomba          # per ordinare le frasi (README, sezione 8)
    d["sicurezza"] = P.max(1)
    cols = ["id_frase", "id_discorso", "posizione", "data", "trimestre", "oratore", "titolo", "frase", "classe",
            "p_falco", "p_colomba", "p_neutrale", "punteggio", "punteggio_continuo", "sicurezza"]
    d[cols].round(4).to_csv(BASE / "risultati/frasi_discorsi_classificate.csv", index=False)
    print("  classi:", d.classe.value_counts(normalize=True).round(3).to_dict())

    q = d.groupby("trimestre").agg(indice=("punteggio", "mean"), n_frasi=("punteggio", "size")).reset_index()
    q.round(4).to_csv(BASE / "risultati/indice_trimestre_locale.csv", index=False)
    orange = pd.read_csv(BASE / "risultati/indice_trimestre_e_tassi.csv")[["trimestre", "indice"]]
    c = q.merge(orange, on="trimestre", suffixes=("_locale", "_orange"))
    print(f"  indice per trimestre, locale vs Orange: correlazione {c.indice_locale.corr(c.indice_orange):.4f}, "
          f"differenza massima {np.abs(c.indice_locale - c.indice_orange).max():.4f} su {len(c)} trimestri")

    disc = d.groupby("id_discorso").agg(data=("data", "first"), oratore=("oratore", "first"),
                                        titolo=("titolo", "first"), indice=("punteggio", "mean"),
                                        n_frasi=("punteggio", "size")).reset_index()
    disc.round(4).to_csv(BASE / "risultati/indice_discorsi.csv", index=False)
    print(f"  discorsi: {len(disc)} (indice per discorso in risultati/indice_discorsi.csv)")


if __name__ == "__main__":
    main()
