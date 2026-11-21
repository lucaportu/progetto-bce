"""
Estrae il campione di TEST di frasi BCE da far etichettare a mano e prepara i file per gli annotatori.

Uso (dalla cartella progetto_BCE, dopo prepara_frasi_bce.py e prepara_wcb_bce.py):
    python3 codice/campione_annotazione.py

Disegno (novembre 2026, dopo l'aggiunta delle frasi WCB)
- L'addestramento del modello B usa le frasi BCE già etichettate di WCB (resoconti delle riunioni, 2015–2024):
  non servono più frasi di addestramento etichettate a mano.
- Il test viene dai DISCORSI del 2025–2026, cioè dopo la fine di WCB: periodo nuovo e genere diverso,
  nessuna frase in comune con l'addestramento.
- Strati: 3 periodi x 3 classi previste = 9 strati, con lo stesso numero di frasi per strato. Così falco e
  colomba non spariscono sotto le frasi neutrali, che sono la maggioranza.
- La classe prevista viene da un modello semplice (TF-IDF + regressione logistica su frasi Fed + WCB).
  Serve solo a stratificare: agli annotatori non viene mostrata.
- 200 frasi, etichettate da ENTRAMBI gli annotatori (ordine diverso per ciascuno) → kappa e valutazione di A, B, C.
- Colonna `peso` = frasi dello strato nella popolazione / frasi dello strato nel campione.
  Con i pesi le metriche si riportano alla distribuzione reale (dove le frasi neutrali sono la maggioranza).
- Le istruzioni seguono la guida di etichettatura BCE degli autori di WCB (dati_bce_wcb/originali/
  guida_annotazione/stance.tex), così le etichette del test hanno lo stesso criterio di quelle di addestramento.
- File di prova: 30 frasi WCB già etichettate, per allenarsi e controllare di usare lo stesso criterio.

Produce in annotazione/:
    annotatore_1.xlsx, annotatore_2.xlsx   da consegnare (foglio Istruzioni + foglio Frasi)
    prova_linee_guida.xlsx                 da consegnare prima: 30 frasi di prova (stesso formato)
    prova_linee_guida_soluzioni.csv        NON consegnare prima di aver fatto la prova: etichette WCB
    chiave_campione.csv                    NON consegnare: strato, classe prevista, peso, oratore, data
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEED = 20261113
N_TEST = 200
N_PROVA = {"falco": 9, "colomba": 9, "neutrale": 9, "irrilevante": 3}
DAL = "2025-01-01"             # dopo l'ultima frase WCB (2024)
MAX_PER_DISCORSO = 3           # frasi massime dallo stesso discorso

BASE = Path(__file__).resolve().parent.parent
FRASI = BASE / "dati_preparati" / "frasi_bce_politica_monetaria.csv"
FED = BASE / "dati_fed" / "frasi_fed_etichettate.csv"
WCB = BASE / "dati_bce_wcb" / "frasi_bce_wcb_etichettate.csv"
WCB_COMPLETO = BASE / "dati_bce_wcb" / "frasi_bce_wcb_completo.csv"
OUT = BASE / "annotazione"

# Periodi: cicli dei tassi BCE. Il confine è la data della decisione, circa 6 giorni prima dell'entrata
# in vigore (tassi_bce.csv): ultimo taglio in vigore l'11/06/2025, primo rialzo in vigore il 17/06/2026.
PERIODI = [
    ("2025-01-01", "2025-07-01", "gen–giu 2025 ultimi tagli"),
    ("2025-07-01", "2026-06-11", "lug 2025–giu 2026 pausa"),
    ("2026-06-11", "2027-01-01", "giu–ott 2026 nuovi rialzi"),
]
CLASSI = {0: "colomba", 1: "falco", 2: "neutrale"}   # stessi codici del dataset Fed

ISTRUZIONI = """ETICHETTATURA DELLE FRASI BCE: FALCO / COLOMBA / NEUTRALE / IRRILEVANTE

Lavorate da soli: non confrontate le etichette con l'altro annotatore prima di aver finito.
Per ogni frase scegliete un'etichetta nella colonna "etichetta" (menu a tendina).
Giudicate la frase da sola: non cercate il discorso da cui viene, la data o cosa ha deciso dopo la BCE.
La domanda è: questa frase, letta oggi, spinge verso tassi più alti o più bassi?

Queste regole sono la traduzione della guida usata dagli autori del dataset WCB per la BCE, cioè per le frasi
con cui viene addestrato il modello B. Seguitele anche quando non siete d'accordo: se test e addestramento
usano criteri diversi, il confronto tra modelli non è valido.

FALCO (spinge verso una politica più restrittiva: tassi più alti, meno acquisti di titoli):
  - inflazione HICP in aumento o sopra il 2%; rischi per l'inflazione al rialzo;
  - PIL o crescita dell'area euro in aumento; inflazione o crescita più alte del previsto;
  - euro che si deprezza; prezzi dell'energia (petrolio, gas) o delle case in aumento;
  - disoccupazione in calo, salari e costi del lavoro in aumento;
  - aumenti dei tassi BCE, vendita o riduzione dei titoli acquistati con APP/PEPP;
  - aumento del disavanzo commerciale dell'UE; offerta di moneta elevata, banche ben capitalizzate,
    domanda di prestiti bassa;
  - accento su "price stability" e "sustained growth".

COLOMBA (spinge verso una politica più accomodante: tassi più bassi, più acquisti):
  - inflazione HICP in calo o sotto il 2%;
  - PIL o crescita in calo; rischi per la crescita al ribasso; inflazione o crescita più basse del previsto;
  - euro che si apprezza; prezzi dell'energia o delle case in calo;
  - disoccupazione in aumento, salari in calo; calo della domanda estera di beni europei;
  - tagli dei tassi BCE, nuovi acquisti o proseguimento dei programmi di acquisto;
  - offerta di moneta bassa, banche poco capitalizzate, domanda di prestiti in aumento;
  - orientamento "accommodative".

NEUTRALE (riguarda la politica monetaria ma non indica una direzione):
  - inflazione stabile intorno al 2%; crescita invariata; inflazione e crescita in linea con le attese;
  - tassi BCE invariati, nessun cambiamento nei programmi di acquisto;
    es. "Current interest rates are at levels that, maintained for a sufficiently long duration, will make a
    substantial contribution to this goal." = neutrale (nessuna variazione dei tassi);
  - paesi fuori dall'area euro, senza effetti indicati sull'area euro;
  - segnali opposti che si compensano; parole come "mixed", "moderate", "reaffirmed".

IRRILEVANTE (non riguarda la politica monetaria):
  - es. "Against this background, it was recalled that the ECB also had secondary objectives.";
  - vigilanza bancaria, pagamenti, euro digitale, organizzazione interna, storia o teoria senza riferimento
    alla situazione attuale.

ALTRI ESEMPI DELLA GUIDA
  "Labour costs had also confirmed continued moderate domestic price pressures." = falco
  "Data on new export orders pointed to a sharp fall of global trade in the second quarter of the year." = colomba
  "In addition, options markets had, by and large, priced out low inflation outcomes over the following five
   years." = falco (aspettative di inflazione più alte)

COLONNE DA COMPILARE
  etichetta   falco / colomba / neutrale / irrilevante (obbligatoria)
  incerto     scrivete "x" se siete in dubbio tra due etichette
  da_scartare scrivete "x" se la frase è rotta (troncata, titolo attaccato, non in inglese)
  note        facoltative: perché siete incerti, quale altra etichetta avreste dato

PRIMA DI INIZIARE: fate da soli il file prova_linee_guida.xlsx (30 frasi), poi confrontate con le soluzioni
e discutete gli errori. Potete aggiungere qui sotto chiarimenti, ma NON cambiate le regole dopo aver iniziato.
"""


def classe_prevista(pop: pd.DataFrame) -> pd.Series:
    fed = pd.concat([pd.read_csv(FED), pd.read_csv(WCB)])
    m = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(fed.sentence, fed.label)
    return pd.Series(m.predict(pop.frase), index=pop.index)


def estrai(pop: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n frasi distribuite in parti uguali sugli strati (se uno strato è piccolo, il resto va agli altri)."""
    strati = pop.groupby("strato")
    disponibili = strati.size().to_dict()
    quota = {s: 0 for s in disponibili}
    rimasti = n
    while rimasti > 0:
        aperti = [s for s in disponibili if quota[s] < disponibili[s]]
        if not aperti:
            raise ValueError(f"Popolazione troppo piccola per {n} frasi")
        for s in sorted(aperti, key=lambda s: rng.random())[:rimasti]:
            quota[s] += 1
            rimasti -= 1
    # Estrazione casuale nello strato, con al massimo MAX_PER_DISCORSO frasi dallo stesso discorso
    # (altrimenti pochi discorsi lunghi pesano troppo). Il peso resta frasi nello strato / frasi estratte.
    # Si parte dagli strati più piccoli: le frasi che mancano lì passano agli strati successivi.
    per_discorso: dict[int, int] = {}
    parti = []
    ordine = sorted(disponibili, key=lambda s: (disponibili[s], rng.random()))
    mancanti = 0
    for i, s in enumerate(ordine):
        extra = -(-mancanti // (len(ordine) - i))          # arrotondato per eccesso
        obiettivo, mancanti = quota[s] + extra, mancanti - extra
        g = strati.get_group(s).sample(frac=1, random_state=int(rng.integers(1e9)))
        scelte = []
        for idx, d in zip(g.index, g.id_discorso):
            if len(scelte) == obiettivo:
                break
            if per_discorso.get(d, 0) < MAX_PER_DISCORSO:
                scelte.append(idx)
                per_discorso[d] = per_discorso.get(d, 0) + 1
        mancanti += obiettivo - len(scelte)
        parti.append(g.loc[scelte].assign(peso=len(g) / len(scelte)))
    if mancanti:
        print(f"Attenzione: estratte {n - mancanti} frasi invece di {n}")
    return pd.concat(parti)


def scrivi_excel(df: pd.DataFrame, path: Path, titolo: str) -> None:
    wb = Workbook()
    ist = wb.active
    ist.title = "Istruzioni"
    ist.column_dimensions["A"].width = 110
    for i, riga in enumerate(ISTRUZIONI.splitlines(), start=1):
        c = ist.cell(row=i, column=1, value=riga)
        if riga.isupper() or riga.endswith(":") and riga.split(":")[0].isupper():
            c.font = Font(bold=True)

    ws = wb.create_sheet("Frasi")
    intest = ["n", "id_campione", "frase", "etichetta", "incerto", "da_scartare", "note"]
    ws.append(intest)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E79")
    for i, r in enumerate(df.itertuples(index=False), start=1):
        ws.append([i, r.id_campione, r.frase, None, None, None, None])
    for col, w in zip("ABCDEFG", [5, 12, 100, 12, 8, 11, 40]):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"falco,colomba,neutrale,irrilevante"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Etichetta non valida",
                        error="Scegliete falco, colomba, neutrale o irrilevante")
    ws.add_data_validation(dv)
    dv.add(f"D2:D{len(df) + 1}")
    ws.freeze_panes = "D2"
    wb.properties.title = titolo
    wb.save(path)


def prova_linee_guida(rng: np.random.Generator) -> pd.DataFrame:
    """Frasi WCB già etichettate, per allenarsi sulle regole prima del campione vero."""
    w = pd.read_csv(WCB_COMPLETO)
    w["etichetta_wcb"] = w.orientamento.map({"hawkish": "falco", "dovish": "colomba", "neutral": "neutrale",
                                             "irrelevant": "irrilevante"})
    parti = [w[w.etichetta_wcb == e].sample(n, random_state=int(rng.integers(1e9))) for e, n in N_PROVA.items()]
    prova = pd.concat(parti).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    prova.insert(0, "id_campione", [f"P{i:02d}" for i in range(1, len(prova) + 1)])
    return prova


def main() -> None:
    import sys
    if (OUT / "annotatore_1.xlsx").exists() and "--sovrascrivi" not in sys.argv:
        sys.exit("I file degli annotatori esistono già e potrebbero essere compilati.\n"
                 "Per rigenerarli comunque: python3 codice/campione_annotazione.py --sovrascrivi")
    rng = np.random.default_rng(SEED)
    pop = pd.read_csv(FRASI)
    pop = pop[pop.data >= DAL].copy()
    date = pd.to_datetime(pop.data)
    pop["periodo"] = None
    for inizio, fine, nome in PERIODI:
        pop.loc[(date >= inizio) & (date < fine), "periodo"] = nome
    pop["classe_prevista"] = classe_prevista(pop).map(CLASSI)
    pop["strato"] = pop.periodo + " | " + pop.classe_prevista

    test = estrai(pop, N_TEST, rng).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test.insert(0, "id_campione", [f"T{i:03d}" for i in range(1, len(test) + 1)])

    OUT.mkdir(exist_ok=True)
    for vecchio in ("disaccordi_da_risolvere.xlsx", "report_accordo.txt", "etichette_test_bce.csv",
                    "etichette_addestramento_bce.csv"):
        (OUT / vecchio).unlink(missing_ok=True)     # risultati del campione precedente, non più validi
    cols = ["id_campione", "strato", "periodo", "classe_prevista", "peso",
            "id_frase", "id_discorso", "data", "oratore", "parole_chiave", "frase"]
    test[cols].to_csv(OUT / "chiave_campione.csv", index=False, encoding="utf-8")
    # stesso campione, ordine diverso per ciascun annotatore (l'ordine non influenza il confronto)
    scrivi_excel(test, OUT / "annotatore_1.xlsx", "Annotatore 1")
    scrivi_excel(test.sample(frac=1, random_state=SEED + 2), OUT / "annotatore_2.xlsx", "Annotatore 2")

    prova = prova_linee_guida(rng)
    scrivi_excel(prova, OUT / "prova_linee_guida.xlsx", "Prova linee guida")
    prova[["id_campione", "etichetta_wcb", "anno", "frase"]].to_csv(
        OUT / "prova_linee_guida_soluzioni.csv", index=False, encoding="utf-8")

    print(f"Popolazione: {len(pop)} frasi da {pop.id_discorso.nunique()} discorsi (dal {DAL})")
    print(f"Test: {len(test)} frasi da {test.id_discorso.nunique()} discorsi, "
          f"max {test.id_discorso.value_counts().max()} per discorso")
    print("\nPopolazione e campione per strato:")
    print(pd.DataFrame({"popolazione": pop.strato.value_counts(), "campione": test.strato.value_counts(),
                        "peso": test.groupby("strato").peso.first().round(1)}).to_string())
    print("\nProva linee guida:", prova.etichetta_wcb.value_counts().to_dict())


if __name__ == "__main__":
    main()
