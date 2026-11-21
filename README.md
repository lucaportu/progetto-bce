# Progetto BCE – indice falco/colomba dai discorsi

Dati scaricati il 12/11/2026.

## Struttura

```
progetto_BCE/
├── orange_1_valutazione_modelli.ows   workflow Orange pronto: A e B, Bag of Words ed embedding (sezione 6)
├── orange_2_indice_e_verifica.ows     workflow Orange pronto: indice e verifica sui tassi (sezione 6)
├── risultati/                    tabelle salvate dai workflow e risposte dell'LLM (llm_kimi-k3.1-pro.csv)
├── app/                          "Falchi & Colombe": dashboard e analizzatore di discorsi (Streamlit)
│   ├── app.py                    l'app: avvio nella sezione 8 ("Punti 2 e 3") o in ISTRUZIONI_AVVIO.md
│   ├── live.html                 pannello in tempo reale dell'analizzatore (video YouTube e testo), sezione 9
│   └── assets/                   logo e simbolo (anello di stelle con €); grafica descritta nella sezione 9
├── .streamlit/                   tema dell'app (config.toml) e modello per la chiave dell'LLM (secrets.toml.esempio)
├── .claude/skills/               skill di design per Claude Code: frontend-design, ui-ux-pro-max (sezione 9)
├── modello/                      modello finale in Python (rete_finale.joblib) e cache degli embedding (sezione 8)
├── .venv/                        ambiente Python del progetto (Python ≥ 3.12, librerie di requirements.txt), circa 1,2 GB
├── requirements.txt              versioni esatte delle librerie (pip install -r requirements.txt)
├── ISTRUZIONI_AVVIO.md           come creare l'ambiente e avviare l'app passo per passo (Mac, Linux, Windows)
├── dati_grezzi/
│   ├── ecb_speeches_all.csv      discorsi BCE (fonte: sito BCE)
│   └── tassi_bce.csv             tassi ufficiali BCE 1999–2026 (fonte: ECB Data Portal)
├── codice/
│   ├── prepara_frasi_bce.py      discorsi → una frase per riga, pulite e filtrate
│   ├── prepara_wcb_bce.py        frasi BCE etichettate WCB → addestramento e test nel formato Fed
│   ├── prepara_file_orange.py    file .tab etichettati (target già impostato) e tassi per trimestre
│   ├── costruisci_workflow_orange.py  costruisce i due workflow .ows (Python di Orange)
│   ├── llm_kimi_orange.py        approccio C: nodo Python Script che fa classificare il test a Kimi (sezione 7)
│   ├── grafico_indice_tassi.py   grafici indice vs tasso sui depositi e correlazione per ritardo (sezione 7)
│   ├── modello_locale.py         modello finale ricostruito in Python (SBERT locale + rete), discorsi classificati (sezione 8)
│   ├── analisi_discorso.py       motore dell'analizzatore di discorsi usato dall'app (sezione 8)
│   ├── campione_annotazione.py   (facoltativo) campione da etichettare a mano
│   └── calcola_accordo.py        (facoltativo) kappa di Cohen ed etichette finali
├── dati_preparati/               frasi pronte (sezione 1b)  ← da caricare in Orange
├── dati_bce_wcb/                 frasi BCE già etichettate (sezione 3b)  ← addestramento di B e TEST
├── annotazione/                  (facoltativo, non usato) etichettatura a mano dei discorsi (sezione 1c)
├── prova_etichettatura_llm/      un LLM etichetta alla cieca il test WCB: accordo con gli esperti (report.txt)
└── dati_fed/
    ├── frasi_fed_etichettate.csv 2.339 frasi Fed etichettate falco/colomba/neutrale  ← per addestrare
    ├── split_ufficiali/          train/test degli autori (3 semi casuali), per confrontarsi con il paper
    ├── indice_fed_e_mercato/     indice falco/colomba della Fed calcolato dagli autori + dati di mercato USA
    └── LICENSE_fed_dataset.md
```

## 1. Discorsi BCE – `dati_grezzi/ecb_speeches_all.csv`

- Fonte: https://www.ecb.europa.eu/press/key/html/downloads.en.html ("Download all speeches")
- **Separatore `|`** (non la virgola). Colonne: `date | speakers | title | subtitle | contents`
- 3.086 discorsi dal 07/02/1997 al 30/10/2026; 172 non hanno testo (pubblicati solo come slide) → da scartare.
  Ne restano 2.914, di cui 157 non in inglese (soprattutto prima del 2018).
- Sono i discorsi del Comitato esecutivo (Lagarde, Lane, Schnabel, de Guindos, …), non dei governatori nazionali.
- Circa 100–115 discorsi l'anno dal 2018 in poi.

Lettura in Python:
```python
import pandas as pd, csv, sys
csv.field_size_limit(sys.maxsize)
df = pd.read_csv("dati_grezzi/ecb_speeches_all.csv", sep="|", engine="python",
                 quoting=csv.QUOTE_NONE)   # senza questo pandas si ferma con un errore sulle virgolette
df = df[df.contents.notna()]
```

**Non aprite questo file con il widget File di Orange**: non dà errore, ma riconosce male il separatore
e carica una tabella sbagliata (una sola colonna, testo troncato). Usate i file in `dati_preparati/`.

Attenzione: molti discorsi non parlano di politica monetaria (pagamenti, euro digitale, vigilanza, clima).
Prima di classificare le frasi conviene filtrarle: lo fa lo script descritto qui sotto.

## 1b. Frasi pronte per Orange – `dati_preparati/`

Si rigenerano con `python3 codice/prepara_frasi_bce.py` (circa 20 secondi; servono pandas e nltk).

| File | Contenuto |
|---|---|
| `frasi_bce_politica_monetaria.tab` | **Da aprire in Orange** (widget Corpus o File). 60.764 frasi inglesi di politica monetaria, una per riga. Tutte le colonne sono meta e il testo è già impostato sulla colonna `frase`: Bag of Words / Document Embedding usano solo il testo |
| `frasi_bce_politica_monetaria.csv` | Le stesse frasi in CSV normale, per pandas, Excel o per gli annotatori |
| `frasi_bce_tutte.csv` | Tutte le 346.464 frasi pulite, con `politica_monetaria` (0/1) e `lingua` (en/altra), per chi vuole cambiare filtro |
| `report_preparazione.txt` | Frasi scartate per motivo, frasi per trimestre, per oratore, parole chiave più frequenti |

Colonne: `id_frase, id_discorso, posizione` (n° della frase nel discorso), `data, anno, trimestre` (es. 2022-T3),
`oratore, titolo, n_parole, parole_chiave` (quelle che hanno fatto tenere la frase), `frase`.

Cosa fa lo script:
1. toglie l'intestazione di ogni discorso ("SPEECH", titolo, sottotitolo, luogo e data) e la sezione "SEE ALSO";
2. divide in paragrafi e poi in frasi con NLTK punkt, toglie i richiami alle note "[1]";
3. scarta: frammenti sotto 5 parole, frasi sopra 100 parole, note a piè di pagina, riferimenti bibliografici,
   didascalie di grafici, titoletti senza punteggiatura finale, frasi ripetute nello stesso discorso;
4. tiene come "politica monetaria" le frasi con almeno una parola chiave forte (inflation, price stability,
   interest rates, monetary policy, tightening, accommodative, asset purchases, forward guidance, APP, PEPP,
   TLTRO, TPI, …). Parole come *wages*, *fragmentation*, *transmission*, *Governing Council* da sole non
   bastano, perché compaiono anche nei discorsi su pagamenti, vigilanza e geopolitica.

Limiti noti: i titoletti di paragrafo restano a volte attaccati alla frase che segue
(es. "Monetary Policy Focusing first on…"); il filtro per parole chiave tiene anche frasi teoriche o storiche
che non esprimono un orientamento (saranno in gran parte "neutrali"). Restano circa 500 frasi per trimestre
dal 2018 (minimo 105, massimo 952).

## 1c. (Facoltativo) Etichettatura a mano dei discorsi – `annotazione/`

**Nel disegno attuale non serve**: il test è fatto con le frasi WCB 2023–2024 (sezione 3b), già etichettate.
La cartella resta per chi volesse aggiungere un controllo sui discorsi (l'accuratezza misurata su WCB vale per i
resoconti delle riunioni, non per i discorsi). Contiene 200 frasi dai discorsi 2025–2026 (stratificate per
periodo e classe prevista, colonna `peso` per riportare le metriche alla distribuzione reale), istruzioni tradotte
dalla guida BCE di WCB e lo script `codice/calcola_accordo.py` per il kappa.

Attenzione: 11 delle 30 frasi di `prova_linee_guida.xlsx` sono del 2023–2024, cioè del test WCB. Se fate la prova,
non usate quelle frasi per scegliere il modello.

## 2. Tassi BCE – `dati_grezzi/tassi_bce.csv`

- Fonte: ECB Data Portal, serie `FM.B.U2.EUR.4F.KR.DFR.LEV`, `...MRR_FR.LEV`, `...MLFR.LEV`
- Una riga per ogni data in cui un tasso è cambiato. `variazione_DFR` = variazione del tasso sui depositi
  (+ = rialzo, − = taglio). Il tasso sui depositi (DFR) è quello che guida oggi la politica monetaria.
- **La data è quella di entrata in vigore**: la decisione viene annunciata circa una settimana prima
  (riunione del Consiglio direttivo). Per la verifica dell'indice usate la data della riunione o tenetene conto.
- Cicli utili per la verifica: rialzi da luglio 2022 a settembre 2023 (da −0,50% a 4,00%), tagli da giugno 2024
  a giugno 2025 (fino al 2,00%), nuovi rialzi a giugno e settembre 2026 (fino al 2,50%).

## 3. Frasi Fed etichettate – `dati_fed/frasi_fed_etichettate.csv`

- Fonte: Shah, Paturi, Chava (2023), "Trillion Dollar Words", ACL 2023 – https://github.com/gtfintechlab/fomc-hawkish-dovish
- **Licenza CC BY-NC 4.0**: uso non commerciale, citare il paper.
- Colonne: `sentence, year, fonte (discorso / verbale riunione / conferenza stampa), label, label_nome`
- Etichette: **0 = dovish (colomba), 1 = hawkish (falco), 2 = neutral (neutrale)**
- 2.339 frasi uniche (1996–2022): 1.134 neutrali, 619 colomba, 586 falco.
- `split_ufficiali/`: i file `lab-manual-combine-train/test-<seme>.xlsx` sono gli split usati nel paper
  (80/20, semi 5768, 78516, 944601). Usateli se volete confrontare i vostri risultati con quelli pubblicati.
- Il modello migliore del paper è già pubblico su Hugging Face: `gtfintechlab/FOMC-RoBERTa`
  (in uscita LABEL_0 = colomba, LABEL_1 = falco, LABEL_2 = neutrale). Può fare da baseline "A".

## 3b. Frasi BCE etichettate (WCB) – `dati_bce_wcb/`

- Fonte: Shah et al. (2025), "Words That Unite The World" (dataset World Central Banks, stesso gruppo del
  dataset Fed) – https://huggingface.co/datasets/gtfintechlab/european_central_bank
- **Licenza CC BY-NC-SA 4.0** (vedi `LICENSE_wcb.md`): uso non commerciale, citare il paper.
- 1.000 frasi dai **resoconti delle riunioni di politica monetaria** (non dai discorsi), 2015–2024,
  etichettate da due annotatori + revisione. Nessuna frase in comune con il file Fed o con i discorsi BCE.
- Etichette originali: hawkish 240, dovish 338, neutral 398, **irrelevant 24**.
- Nel paper: RoBERTa-large F1 pesato 0,755 sulla BCE, GPT-4.1 senza addestramento 0,66 (split casuali, sui resoconti).

**Divisione per anno** (non quella casuale degli autori, in cui frasi dello stesso resoconto finiscono sia in
addestramento sia in test e il risultato esce ottimistico):

| File | Contenuto |
|---|---|
| `frasi_bce_wcb_addestramento.csv` | **Addestramento di B.** 732 frasi 2015–2022: 161 falco, 270 colomba, 301 neutrali |
| `frasi_bce_wcb_test.csv` | **Test comune per A, B e C.** 244 frasi 2023–2024 (picco dei tassi e inizio dei tagli): 79 falco, 68 colomba, 97 neutrali |
| `frasi_bce_wcb_etichettate.csv` | Tutte le 976 frasi. Solo alla fine, per riaddestrare il modello scelto prima di costruire l'indice |
| `frasi_bce_wcb_completo.csv` | Tutte le 1.000 frasi con le etichette originali (anche "irrelevant"), le etichette temporale e di incertezza, e lo split dei 3 semi degli autori |
| `originali/` | Parquet scaricati da Hugging Face e guida di etichettatura BCE degli autori (`guida_annotazione/stance.tex`), non modificati. `python3 codice/prepara_wcb_bce.py` rigenera i CSV |

Tutti i CSV in formato Fed hanno le **stesse colonne di `frasi_fed_etichettate.csv`** (`sentence, year, fonte,
label, label_nome`, codifica 0 colomba / 1 falco / 2 neutrale, `fonte` = "resoconto riunione BCE") e sono senza le
24 frasi irrilevanti.

In Orange:
- **A**: Corpus (`frasi_fed_etichettate.csv`) → preprocessing / Bag of Words → Logistic Regression → **Test & Score**
  con "Test on test data", dando come dati di test `frasi_bce_wcb_test.csv` (con lo stesso preprocessing).
- **B**: come A, ma l'addestramento è **Concatenate** di `frasi_fed_etichettate.csv` e `frasi_bce_wcb_addestramento.csv`.
  **B2**: solo `frasi_bce_wcb_addestramento.csv`.
- **C**: l'LLM classifica le 244 frasi di test (Python Script), con le stesse istruzioni (guida BCE di WCB).
- Come variabile da prevedere (target) usate **`label_nome`**; `label` (numero) e `year` vanno ignorate o messe tra i meta.
- Il test non va mai usato per scegliere impostazioni del modello: per quello usate la cross-validation
  sull'addestramento. Le 244 frasi si guardano una volta sola, alla fine.

Controllo indicativo (TF-IDF + regressione logistica in Python, non è il risultato del progetto): sul test 2023–2024
A ≈ 0,54, B ≈ 0,53, B2 ≈ 0,49 di F1 pesato. Con un modello così semplice aggiungere le frasi WCB non aiuta: da
verificare in Orange e con modelli migliori (embedding, RoBERTa) prima di trarre conclusioni.

**Approccio C, prova già fatta** (`prova_etichettatura_llm/`): un LLM (Claude) ha etichettato alla cieca le
244 frasi di test con la guida WCB. Su 243 frasi (una esclusa perché la sua etichetta era già stata vista):
accordo 76%, **F1 pesato 0,76** (IC 95% 0,71–0,81), kappa 0,64. Limiti: possibile contaminazione (WCB è pubblico dal
2025) ed etichette date in conversazione, non con un programma ripetibile. Per lo stesso motivo (una etichetta su
quattro diversa dagli esperti) un LLM non va usato per etichettare il test.

**Se C resta il migliore**, per l'indice ci sono due strade: LLM solo sulle frasi dal 2018 (circa 19.000; API,
qualche ora, spesa contenuta, risposte salvate in un file), oppure B per l'indice e C come riferimento. Una via di
mezzo: calcolare l'indice con entrambi sul 2022–2026 e controllare che le curve si muovano insieme.

Miglioramenti facoltativi:
- ripetere A e B con **Document Embedding (SBERT)** al posto di Bag of Words, prima di concludere che le frasi
  BCE non aiutano B;
- per confrontarsi con lo 0,755 del paper, valutare anche sugli **split casuali degli autori** (colonne
  `split_seme_…` di `frasi_bce_wcb_completo.csv`: un seme alla volta, train → test, media dei 3). È un confronto
  ottimistico, non il risultato principale.

Attenzione:
- **Le frasi sono tutte in minuscolo** (le Fed e i discorsi no). Per Bag of Words in Orange non cambia nulla (il
  preprocessing mette già in minuscolo); per modelli tipo RoBERTa sì, è da tenere presente.
- Le linee guida degli autori considerano "falco" anche le frasi su un'economia forte (es. "real GDP had
  increased by 0.3%"), come per la Fed. La guida originale BCE è in `originali/guida_annotazione/`; usatela
  anche come istruzioni per l'LLM dell'approccio C.
- **Limite da dichiarare**: test e addestramento sono resoconti delle riunioni; l'indice si costruisce sui
  discorsi. L'accuratezza misurata vale per i resoconti; per i discorsi il controllo è indiretto (verifica
  dell'indice sui tassi).

## 4. Indice Fed e mercato – `dati_fed/indice_fed_e_mercato/`

Indice falco/colomba della Fed calcolato dagli autori per discorsi, verbali e conferenze stampa,
più rendimenti dei Treasury, CPI, PPI e QQQ. Descrizione delle colonne in `data_deatails.md`.
Utile come esempio di come costruire e validare l'indice.

## 5. Verifica: l'indice anticipa i tassi?

L'indice (frasi falco − frasi colomba) / frasi totali misura il **tono**. La verifica controlla se il tono si muove
**prima** delle decisioni sui tassi (sale prima dei rialzi, scende prima dei tagli). Tre livelli, dal minimo al più
rigoroso:

1. **Grafico** (minimo): indice della BCE per trimestre e tasso sui depositi (`dati_grezzi/tassi_bce.csv`) sullo
   stesso asse temporale. Mostra a occhio chi si muove prima, ma è solo un'evidenza descrittiva: non dice di quanto
   l'indice anticipa, né se l'anticipo è casuale, e chi guarda sapendo già quando sono arrivati i rialzi tende a
   notare solo le salite dell'indice che li hanno preceduti.
2. **Due numeri accanto al grafico** (consigliato):
   - **correlazione con ritardo**: correlazione tra l'indice del trimestre t e la variazione del tasso sui depositi
     nei trimestri t, t+1, t+2, t+3. Se è più alta con 1–2 trimestri di ritardo che con 0, l'indice anticipa, e il
     ritardo con la correlazione più alta dice di quanto. In Orange: Feature Constructor (colonna "variazione del
     tasso nel trimestre successivo") → Correlations; in Python con `shift()` di pandas;
   - **confronto con la regola banale** "la prossima decisione sarà come l'ultima" (dopo un rialzo un altro
     rialzo, altrimenti fermo). Le decisioni BCE sono molto prevedibili da sole: l'indice aggiunge informazione
     solo se indovina più decisioni di questa regola, soprattutto ai **punti di svolta** (inizio rialzi 2022, fine
     rialzi 2023, inizio tagli 2024, ripresa rialzi 2026).
3. **Test previsivo sulle decisioni reali** (facoltativo, il più convincente). Non servono etichette: la risposta
   giusta è la decisione che la BCE ha preso davvero.
   - Unità: la **riunione** di politica monetaria (circa 70 dal 2018, contro circa 35 trimestri). Per ogni riunione,
     indice sui discorsi tra la riunione precedente e l'inizio del periodo di silenzio (la settimana prima della
     riunione, in cui i membri non parlano di politica monetaria). Mai discorsi successivi alla riunione.
   - Da prevedere: rialzo / fermo / taglio. Regola semplice con due soglie (sopra la prima → rialzo, sotto la
     seconda → taglio, altrimenti fermo).
   - Soglie scelte sul **2018–2024**, verifica **una sola volta** sul **2025–2026** (ultimi tagli, pausa, rialzi di
     giugno e settembre 2026). Confronto con la regola banale del punto 2.
   - Il punto chiave è la ripresa dei rialzi nel 2026: l'indice era salito prima di giugno 2026?
   - Dati da aggiungere: il **calendario di tutte le riunioni** (sito BCE, "Monetary policy meetings"), comprese
     quelle senza variazione dei tassi. `tassi_bce.csv` contiene solo le date in cui il tasso cambia, ed è la data
     di entrata in vigore: la decisione è di circa una settimana prima.

Limiti da dichiarare: i punti di svolta sono pochi (3–4 dal 2018), quindi anche il test previsivo resta
un'evidenza, non una prova statistica forte. In più l'indice misura il tono dei membri del Comitato esecutivo, non
dei governatori nazionali, che votano anche loro.

## 6. Workflow Orange pronti (da usare come confronto)

Due workflow già montati e configurati, da aprire con Orange (File → Open). Servono come confronto con quelli
che costruite voi: stessi dati, stesse scelte descritte qui sopra.

Si rigenerano con il Python di Orange (prima `python3 codice/prepara_file_orange.py`):
```
/Applications/Orange.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3 codice/costruisci_workflow_orange.py
```
(con `--prova` li esegue anche senza embedding e stampa i risultati).

**File di supporto creati per i workflow** (`codice/prepara_file_orange.py`):
- `dati_fed/frasi_fed_etichettate.tab`, `dati_bce_wcb/frasi_bce_wcb_{addestramento,test,etichettate}.tab`: stesse
  frasi dei CSV, ma con l'intestazione di Orange: testo = `sentence`, target = `label_nome`, il resto meta.
  Il widget Corpus le carica pronte, senza Select Columns.
- `dati_preparati/tassi_bce_trimestrali.tab`: un trimestre per riga dal 2018 (stesso formato `2022-T3` delle frasi):
  tasso sui depositi a fine trimestre, variazione decisa nel trimestre e 1, 2, 3 trimestri dopo, decisione del
  trimestre successivo e previsione della **regola banale**. Data della decisione stimata come entrata in vigore − 6
  giorni; dopo il 2026-T3 le variazioni sono vuote (trimestre non ancora concluso).
- `dati_preparati/stopword_inglesi_senza_negazioni.txt`: stopword inglesi di NLTK **senza** negazioni e parole di
  direzione (no, not, nor, further, more, most, up, down, above, below, over, under, against, again, very, too, few,
  only), che per il tono contano.

**`orange_1_valutazione_modelli.ows`** (22 widget): Corpus Fed, WCB addestramento, WCB test → Concatenate (Fed +
WCB = addestramento di B) → due rami:
- Bag of Words: Preprocess Text (minuscolo, parole, stopword dal file qui sopra) → Bag of Words (TF-IDF);
- embedding: Document Embedding (SBERT, inglese); il test passa per il suo Document Embedding.

Modelli: Logistic Regression, Neural Network (replicabile), Constant (classe più frequente = termine di paragone
minimo). Quattro Test & Score ("Test on test data" sulle 244 frasi WCB 2023-24): A e B × Bag of Words / embedding,
ognuno con la sua Confusion Matrix. Leggete l'F1 con "Average over classes" (è l'F1 pesato).

Risultati del ramo Bag of Words, ottenuti eseguendo il workflow (F1 pesato sul test):

| | Logistic Regression | Neural Network | Constant |
|---|---|---|---|
| A (solo Fed), Bag of Words | 0,497 | 0,516 | 0,226 |
| B (Fed + WCB), Bag of Words | 0,479 | 0,439 | 0,226 |

I numeri differiscono un po' dal controllo in Python (0,54 / 0,53) perché il preprocessing è diverso (solo parole
singole, altra lista di stopword, regolarizzazione di Orange). Il ramo embedding **non è stato eseguito** qui: manda
il testo al server di Orange e lo fa la prima volta che aprite il workflow (alcuni minuti; poi Orange lo tiene in
memoria).

**`orange_2_indice_e_verifica.ows`** (26 widget):
1. modello finale: Corpus Fed + WCB tutte (976) → Concatenate → Document Embedding → Logistic Regression (nome
   `modello`). Se nel workflow 1 vince un'altra combinazione, cambiate questi widget;
2. frasi dei discorsi: Corpus `frasi_bce_politica_monetaria.tab` → Select Rows (anno ≥ 2018: 19.038 frasi) →
   Document Embedding → Predictions. **Questo Document Embedding non parte da solo** (circa 19.000 frasi, lento):
   apritelo e premete Apply;
3. Python Script (toglie le colonne numeriche della rappresentazione, tiene testo, metadati e previsione) → Feature
   Constructor `punteggio` = `1 if modello == "hawkish (falco)" else (-1 if modello == "dovish (colomba)" else 0)`;
4. tre Group By (media e conteggio del punteggio = indice): per trimestre, per oratore e trimestre, per discorso
   (con data, oratore, titolo); le ultime due con Data Table e Save Data in `risultati/` (premete Save);
5. verifica: Merge Data (indice per trimestre + `tassi_bce_trimestrali.tab` sul trimestre) → Correlations (filtrate
   sull'indice: guardate le righe con `variazione_t+1`, `+2`, `+3`), Scatter Plot indice nel tempo (colore = tasso),
   Scatter Plot indice vs variazione del trimestre successivo, Data Table e Save Data.

Il workflow 2 è stato provato eseguendolo con Bag of Words al posto dell'embedding: 19.038 frasi classificate,
indice per 36 trimestri, 209 coppie oratore-trimestre, 656 discorsi, unione con i tassi corretta.

Note pratiche:
- i widget Corpus ricordano il **percorso completo** dei file: se spostate la cartella del progetto, in ogni Corpus
  premete Browse e riselezionate il file (oppure rigenerate i workflow con lo script);
- il confronto con la regola banale non è un widget: in `tassi_bce_trimestrali.tab` ci sono `decisione_t+1` e
  `regola_banale_t+1`; per confrontarli con l'indice serve una soglia (sezione 5), si fa in Excel o Python;
- per il grafico a linee nel tempo installate l'add-on Timeseries (Line Chart); qui si usa lo Scatter Plot.

## 7. Risultati ottenuti – workflow `Workflow BCE AI&CV.ows` (20/11/2026)

Il workflow con cui è stato fatto il lavoro è `../Workflow BCE AI&CV.ows` (nella cartella sopra `progetto_BCE`).
Orange non salva i punteggi nel file: i numeri qui sotto sono stati ricalcolati con le stesse impostazioni del
workflow, gli stessi file e gli stessi embedding SBERT (già in memoria, nessun nuovo invio al server). La
cross-validation riprodotta coincide con quella mostrata da Test & Score; le correlazioni sono quelle del widget
Correlations.

### Impostazioni usate

- **Rappresentazione del testo**: solo Document Embedding (SBERT multilingue, server di Orange). Niente Bag of Words.
  In ogni Corpus la *Used text feature* è solo `sentence` (frasi etichettate) o `frase` (discorsi).
- **Modelli** (gli stessi nodi collegati a tutti e tre i Test & Score): regressione logistica L2 con C = 1 e C = 10;
  rete neurale con 100 neuroni, ReLU, Adam, 500 iterazioni, addestramento replicabile, α = 0,01 e α = 1.
- **Scelta dei parametri**: cross-validation a 10 fold stratificata sull'addestramento di ciascun approccio,
  F1 pesato, differenze sotto 0,02 considerate trascurabili.
- **Valutazione finale**: "Test on test data", una volta, sulle 244 frasi WCB 2023–2024.
- **Approccio C**: Kimi K3.1 Pro (Moonshot, ragionamento "high") tramite il nodo Python Script `LLM`
  (codice in `codice/llm_kimi_orange.py`, risposte salvate in `risultati/llm_kimi-k3.1-pro.csv`), con le istruzioni
  della guida BCE di WCB. Tutte le 244 frasi sono state classificate dallo stesso modello e con le stesse istruzioni.

### Confronto A, B, B2, C sul test WCB 2023–2024 (244 frasi)

F1 pesato. "CV" = cross-validation sull'addestramento (serve solo a scegliere il modello: ogni approccio è misurato
su frasi diverse, quindi i valori CV non si confrontano tra approcci).

| Modello | A – solo Fed (2.339) CV / **test** | B – Fed + WCB 2015-22 (3.071) CV / **test** | B2 – solo WCB 2015-22 (732) CV / **test** |
|---|---|---|---|
| logistica C=10 | 0,577 / **0,564** | 0,585 / **0,596** | 0,583 / **0,574** |
| logistica C=1 | 0,598 / **0,559** | 0,608 / **0,606** | 0,627 / **0,595** |
| rete α=0,01 | 0,608 / **0,569** | 0,616 / **0,611** | 0,623 / **0,596** |
| **rete α=1** | 0,612 / **0,592** | 0,634 / **0,633** | 0,643 / **0,611** |

Il modello migliore in cross-validation è la **rete con α=1** in tutti e tre gli approcci (la regolarizzazione più
forte funziona meglio); è anche il migliore sul test.

Nota: questi sono i numeri di **un solo addestramento** (seme 1 di Orange). Ripetendo la rete con 10 semi diversi
(sezione 8, punto 1) l'F1 sul test varia di ±0,01–0,02 e le medie sono un po' più basse: A ≈ 0,57, B ≈ 0,62,
B2 ≈ 0,60. Le conclusioni qui sotto non cambiano.

| Approccio (modello migliore) | F1 pesato test | IC 95% (bootstrap) | Accuratezza | Kappa con gli esperti |
|---|---|---|---|---|
| A – rete α=1, solo Fed | 0,592 | 0,53 – 0,65 | 0,594 | 0,38 |
| B – rete α=1, Fed + WCB | **0,633** | 0,57 – 0,69 | 0,631 | 0,44 |
| B2 – rete α=1, solo WCB | 0,611 | 0,55 – 0,67 | 0,607 | 0,41 |
| C – Kimi K3.1 Pro, senza addestramento | **0,763** | 0,71 – 0,82 | 0,766 | 0,65 |
| C – Claude (prova precedente, 243 frasi) | 0,759 | 0,71 – 0,81 | 0,761 | 0,64 |

Differenze appaiate sulle stesse frasi (bootstrap, 2.000 ricampionamenti):
- **B − A = +0,04** (IC 95% −0,02 / +0,10; B migliore nel 91% dei ricampionamenti): aggiungere le 732 frasi BCE
  aiuta, ma la differenza è dentro l'incertezza del test;
- **B − B2 = +0,02** (−0,04 / +0,08): con le sole 732 frasi BCE si ottiene quasi lo stesso risultato; i dati Fed
  aggiungono poco;
- **C (Kimi) − B = +0,13** (+0,06 / +0,21): l'unica differenza netta. L'LLM senza addestramento batte tutti i modelli
  addestrati.
- Kimi e Claude concordano sull'85% delle frasi (kappa 0,77): due LLM diversi danno lo stesso risultato.

Matrici di confusione sul test (righe = esperti, colonne = modello; ordine colomba / falco / neutrale):

| | B – rete α=1 | Kimi K3.1 Pro |
|---|---|---|
| colomba (68) | 41 / 16 / 11 | 56 / 10 / 2 |
| falco (79) | 14 / 51 / 14 | 5 / 73 / 1 |
| neutrale (97) | 17 / 18 / 62 | 16 / 23 / 58 |

Gli errori della rete sono distribuiti su tutte le classi. Kimi riconosce quasi tutte le frasi da falco (recall 0,92)
e da colomba (0,82), ma solo il 60% delle neutrali: 39 frasi che gli esperti considerano descrittive vengono lette come
falco (23) o colomba (16). Quando Kimi dice "neutrale" ha ragione nel 95% dei casi. È lo stesso errore visto con
Claude.

**Conclusione sul confronto**: il modello addestrato sulla Fed perde molto sulla BCE (0,59 contro circa 0,61 in
cross-validation sulla Fed); qualche centinaio di frasi BCE lo migliora di poco (0,63), mentre un LLM con la guida
degli annotatori arriva a 0,76. **Per l'indice è stato usato B (rete α=1)**, perché classificare con l'LLM le 19.000
frasi dei discorsi sarebbe costato troppo; C resta come riferimento di quanto si potrebbe guadagnare.

### Indice falco/colomba

- Modello finale: rete α=1 riaddestrata su Fed + tutte le 976 frasi WCB (`frasi_bce_wcb_etichettate.tab`).
- Frasi classificate: 19.038 frasi di politica monetaria dei discorsi dal 1/1/2018 (Select Rows su `data`):
  49% neutrali, 26% falco, 24% colomba.
- Punteggio +1 falco, −1 colomba, 0 neutrale (Formula); indice = media per trimestre (Group By, 36 trimestri);
  unito ai tassi trimestrali con Merge Data sul trimestre; grafico con Form Timeseries (passo di 3 mesi da gennaio
  2018) e Line Chart.

| Trimestre | Indice | Frasi | Tasso depositi | Decisione nel trimestre successivo |
|---|---|---|---|---|
| 2018-T1 … 2019-T1 | da −0,03 a −0,13 | 449–763 | −0,40 | fermo |
| 2019-T2 | −0,16 | 323 | −0,40 | taglio (−0,10) |
| 2019-T3 … 2020-T1 | da −0,15 a −0,19 | 438–638 | −0,50 | fermo |
| 2020-T2 | **−0,34** (minimo: pandemia, PEPP) | 437 | −0,50 | fermo |
| 2020-T3 … 2021-T1 | da −0,18 a −0,29 | 441–627 | −0,50 | fermo |
| 2021-T2 | −0,03 | 285 | −0,50 | fermo |
| 2021-T3 | −0,01 | 548 | −0,50 | fermo |
| 2021-T4 | +0,06 | 725 | −0,50 | fermo |
| 2022-T1 | +0,16 | 785 | −0,50 | fermo |
| 2022-T2 | +0,18 | 638 | −0,50 | **rialzo** (+1,25) |
| 2022-T3 | **+0,28** (massimo del ciclo) | 573 | 0,75 | rialzo |
| 2022-T4 | +0,25 | 712 | 2,00 | rialzo |
| 2023-T1 | +0,17 | 828 | 3,00 | rialzo |
| 2023-T2 | +0,14 | 578 | 3,50 | rialzo |
| 2023-T3 | +0,15 | 696 | 4,00 | fermo |
| 2023-T4 | +0,14 | 408 | 4,00 | fermo |
| 2024-T1 | +0,06 | 521 | 4,00 | **taglio** (−0,25) |
| 2024-T2 | +0,06 | 938 | 3,75 | taglio |
| 2024-T3 | +0,14 | 429 | 3,50 | taglio |
| 2024-T4 | +0,11 | 952 | 3,00 | taglio |
| 2025-T1 | +0,10 | 477 | 2,50 | taglio |
| 2025-T2 | +0,05 | 425 | 2,00 | fermo |
| 2025-T3 | +0,04 | 105 | 2,00 | fermo |
| 2025-T4 | −0,05 | 447 | 2,00 | fermo |
| 2026-T1 | **+0,17** | 313 | 2,00 | **rialzo** (+0,25) |
| 2026-T2 | +0,19 | 602 | 2,25 | rialzo |
| 2026-T3 | +0,31 | 197 | 2,50 | (non ancora noto) |
| 2026-T4 | +0,04 | 142 (trimestre incompleto) | – | – |

### Verifica sui tassi

**Grafici** (fuori da Orange): `python3 codice/grafico_indice_tassi.py` legge `risultati/indice_trimestre_e_tassi.csv`
(l'indice per trimestre di questa sezione) e `dati_grezzi/tassi_bce.csv`, e salva in `risultati/`:
- `grafico_indice_tassi.png/.pdf`: in alto indice e tasso sui depositi **sovrapposti su un solo asse**, entrambi
  standardizzati sul 2018–2026 (0 = media del periodo, 1 = una deviazione standard), così si confrontano forma e
  tempi senza un secondo asse y che altererebbe incroci e distanze; sotto, i valori reali di indice e tasso. Linee
  tratteggiate = decisioni di svolta; fasce azzurre = periodi in cui l'indice si era già mosso e il tasso no;
- `grafico_correlazione_ritardi.png/.pdf`: correlazione tra l'indice e la variazione del tasso nello stesso
  trimestre e 1, 2, 3 trimestri dopo.

**Correlazione** (widget Correlations) tra l'indice del trimestre t e la variazione del tasso sui depositi:

| Variazione del tasso | nello stesso trimestre (t) | t+1 | t+2 | t+3 |
|---|---|---|---|---|
| Correlazione con l'indice | +0,40 | +0,38 | +0,33 | +0,24 |

Calcolate solo sui trimestri che hanno il dato (escludendo gli ultimi, ancora senza variazione) diventano +0,40 /
+0,40 / +0,36 / +0,26: la conclusione non cambia.

Lettura:
- il legame è positivo e moderato: discorsi più da falco vanno con rialzi, più da colomba con tagli;
- la correlazione con il trimestre **successivo** (+0,38/0,40) è alta quanto quella con lo stesso trimestre e usa
  solo discorsi fatti **prima** della decisione: c'è un contenuto anticipatorio, che si riduce con l'orizzonte;
- con circa 35 trimestri e serie che cambiano lentamente, è un'evidenza descrittiva, non una prova statistica.

#### Le correlazioni spiegate in modo semplice

**Cos'è una correlazione.** Un numero tra −1 e +1 che dice quanto due cose si muovono insieme: +1 = sempre insieme
(quando una sale, sale anche l'altra), 0 = nessun legame, −1 = sempre in senso opposto. Qui le due cose sono
l'**indice** del trimestre (quanto i discorsi erano da falco o da colomba) e **di quanto è cambiato il tasso**
(positivo se è salito, negativo se è sceso, zero se è rimasto fermo).

**Le quattro barre** fanno la stessa domanda guardando il tasso in momenti diversi:

| Barra | Domanda | Risposta |
|---|---|---|
| stesso trimestre | Quando i discorsi sono da falco, i tassi salgono in quello stesso trimestre? | +0,40 |
| 1 trimestre dopo | …salgono nei 3 mesi successivi? | +0,40 (Orange: +0,38) |
| 2 trimestri dopo | …salgono tra 3 e 6 mesi? | +0,36 (Orange: +0,33) |
| 3 trimestri dopo | …salgono tra 6 e 9 mesi? | +0,26 (Orange: +0,24) |

Esempio: nel 2022-T2 l'indice era già alto (+0,18) ma il tasso era ancora fermo a −0,50%; nel trimestre dopo la BCE
ha alzato di 1,25 punti. Questa coppia "indice alto oggi, tasso che sale domani" è uno dei casi che fanno salire la
barra "1 trimestre dopo".

**Come si leggono.**
1. **Tutte positive**: quando i membri della BCE parlano da falchi i tassi tendono a salire, quando parlano da colombe
   a scendere. L'indice misura qualcosa di reale.
2. **0,40 è un legame moderato, non forte**: l'indice "spiega" circa il 16% delle variazioni dei tassi
   (0,40 × 0,40 = 0,16). Il resto dipende da altro: dati sull'inflazione, eventi imprevisti, decisioni prese
   all'ultimo.
3. **La barra più importante è "1 trimestre dopo"**. Quella dello stesso trimestre è un po' ingannevole: molti
   discorsi arrivano *dopo* la decisione e la commentano (dopo un rialzo si parla di inflazione alta e di rialzi),
   quindi l'indice *reagisce* invece di prevedere. La barra "1 trimestre dopo" usa solo discorsi fatti *prima* della
   decisione: che sia alta quanto l'altra vuol dire che **le parole anticipano le decisioni**.
4. **Il legame cala con il tempo** (0,40 → 0,36 → 0,26): i discorsi di oggi dicono qualcosa sui prossimi 3–6 mesi,
   poco su quello che succederà tra 9 mesi. È quello che ci si aspetta.

**Due cautele da dire sempre.**
- **Pochi dati**: circa 35 trimestri e 3–4 cambi di direzione dei tassi. I numeri sono un'indicazione, non una prova.
- **I tassi BCE sono prevedibili anche da soli**: dopo un rialzo di solito ne arriva un altro (la correlazione tra la
  variazione di un trimestre e quella del trimestre dopo è 0,76, più del nostro 0,40). Nei periodi "normali" basta
  guardare l'ultima mossa; l'indice diventa utile soprattutto **ai punti di svolta**, dove quella regola sbaglia
  (inizio dei rialzi 2022, ripresa dei rialzi 2026), e che l'indice aveva anticipato.

**In una frase**: il tono dei discorsi è legato alle decisioni sui tassi (correlazione di circa 0,4); il legame vale
anche con il trimestre successivo, cioè le parole anticipano le decisioni, e si indebolisce allungando l'orizzonte.
È un legame moderato, utile soprattutto nei momenti di svolta.

**Confronto con la regola banale** ("la prossima decisione sarà come l'ultima"): le variazioni del tasso sono molto
persistenti (correlazione tra variazione in t e in t+1 = **+0,76**, contro +0,38/0,40 dell'indice). La regola banale
indovina 22 decisioni su 34 (65%), ma sbaglia proprio ai punti di svolta. Lì l'indice dà i segnali più utili:
- **inizio dei rialzi (luglio 2022)**: l'indice sale da −0,18 (2021-T1) a +0,06 (2021-T4), +0,16 (2022-T1) e
  +0,18 (2022-T2), cioè **2–3 trimestri prima** del primo rialzo; la regola banale prevedeva "fermo";
- **fine dei rialzi (settembre 2023)**: l'indice scende dal massimo di +0,28 (2022-T3) a circa +0,14 già durante
  il 2023, prima della pausa;
- **inizio dei tagli (giugno 2024)**: l'indice scende a +0,06 nel 2024-T1, il trimestre prima del primo taglio, ma
  resta positivo e risale a +0,14 durante i tagli: l'indice segue il tono sull'inflazione più che la direzione dei
  tassi;
- **ripresa dei rialzi (giugno 2026)**: l'indice passa da −0,05 (2025-T4) a **+0,17 nel 2026-T1**, il trimestre
  prima del primo rialzo; la regola banale prevedeva "taglio".

**Per oratore** (minimo 100 frasi nel periodo; indice = media del punteggio):

| Periodo | dal più colomba al più falco |
|---|---|
| 2022–2023 | Lane +0,07 · Elderson +0,16 · Schnabel +0,18 · de Guindos +0,19 · Panetta +0,22 · Lagarde +0,28 |
| 2024–2026 | Lane +0,06 · de Guindos +0,07 · Schnabel +0,09 · Cipollone +0,10 · Elderson +0,11 · Lagarde +0,17 |

Lane è il più colomba in entrambi i periodi. Il confronto Schnabel–Panetta del 2022–23 non va nella direzione
attesa (Panetta risulta più "falco"): l'indice misura il contenuto delle frasi (per la guida WCB descrivere
un'inflazione alta è "falco"), non la preferenza dell'oratore sui tassi. Da presentare con questa cautela.

### Limiti e note sul workflow

- L'accuratezza è misurata sui **resoconti delle riunioni**; l'indice è costruito sui **discorsi**. Per i discorsi il
  controllo è solo indiretto (la verifica sui tassi).
- Gli embedding SBERT sono calcolati sul server di Orange; l'approccio C manda le 244 frasi di test a Moonshot. Sono
  testi pubblici, ma va scritto nella relazione. I dati WCB sono pubblici dal 2025: l'LLM potrebbe averli visti.
- Trimestri con poche frasi, da leggere con cautela: 2025-T3 (105), 2026-T3 (197), 2026-T4 (142, incompleto).
- ⚠️ **Il file `.ows` contiene la chiave API di Kimi** nel nodo `LLM` (riga `API_KEY = ...`). Prima di consegnare o
  condividere il workflow aprite il nodo, cancellate la chiave (`API_KEY = ""`) e salvate.
  Il file `codice/llm_kimi_orange.py` invece non contiene più la chiave (tolta il 20/11/2026): la legge dalla variabile
  d'ambiente `MOONSHOT_API_KEY` o dal file `~/.kimi_api_key` (una riga con la sola chiave, fuori dal progetto). L'unica
  chiave rimasta nel progetto è quella dell'app in `.streamlit/secrets.toml`, che non va consegnato né caricato online.
- Il collegamento Predictions → `script` usa l'uscita *Selected Predictions*: funziona perché in Predictions non è
  selezionata nessuna riga (in quel caso passano tutte). Se selezionate delle righe, passano solo quelle; più sicuro
  usare l'uscita *Predictions*.
- Non fatto: il test previsivo con soglie sulle riunioni (sezione 5, punto 3). La dashboard è stata fatta dopo, in
  Python (sezioni 8 e 9).

## 8. Piano: dashboard e analisi di un discorso (deciso il 20/11/2026)

Obiettivo: trasformare l'analisi in uno strumento utilizzabile, con due parti: una **dashboard** interattiva e un
**analizzatore di discorsi** che colora le frasi una alla volta e alla fine dà punteggio e classe. Tutto in Python
(Streamlit), fuori da Orange. Le frasi sono classificate **solo dalla rete neurale**; l'LLM serve solo per il
riassunto finale di un discorso.

### Punto 1 – Riaddestrare il modello in locale, in Python

Il modello finale di Orange (rete α=1) usa gli embedding calcolati dal server di Orange e non si può usare fuori da
Orange. Lo si ricostruisce in locale, con le stesse scelte:
- embedding SBERT calcolati sul computer (libreria `sentence-transformers`, nessun server esterno);
- stessa rete: 100 neuroni, ReLU, Adam, α = 1, 500 iterazioni, seme fisso (scikit-learn `MLPClassifier`);
- **controllo**: addestramento su Fed + WCB 2015–2022 e valutazione sulle 244 frasi di test. Si va avanti solo se
  l'F1 pesato è vicino a quello di Orange (0,63). Poi riaddestramento su Fed + tutte le 976 frasi WCB (modello
  finale) e salvataggio su file;
- con lo stesso modello si ricalcolano le previsioni sulle 19.038 frasi dei discorsi dal 2018, con le probabilità
  di ogni classe, e si salvano in un file usato dalla dashboard. Va controllato che l'indice per trimestre sia
  coerente con quello di Orange (sezione 7).

**Fatto (20/11/2026)** – `codice/modello_locale.py`, da lanciare con l'ambiente del progetto:
```
.venv/bin/python codice/modello_locale.py
```
(circa 1 minuto e mezzo; gli embedding vengono salvati in `modello/cache/` e non si ricalcolano). L'ambiente `.venv/`
(Python 3.13, sentence-transformers, scikit-learn, pandas) si ricrea con
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (su Windows `.venv\Scripts\pip`; dettagli in
`ISTRUZIONI_AVVIO.md`).

- **Embedding**: Orange usa `paraphrase-multilingual-MiniLM-L12-v2` (384 numeri per frase). Calcolato in locale dà
  vettori **identici** a quelli del server di Orange (coseno 1,0000 sulle 244 frasi di test).
- **Rete**: standardizzazione (Orange la applica da sola alla rete neurale) + MLP 100 neuroni, ReLU, Adam, α = 1,
  500 iterazioni, seme 1.
- **Verifica sul test** (F1 pesato). La rete parte da pesi casuali: con semi diversi l'F1 cambia di qualche
  centesimo anche a parità di dati. Per questo il confronto con Orange (un solo addestramento) è fatto contro la
  variabilità su 10 semi:

  | Approccio | locale, seme 1 | locale, media di 10 semi (± dev. st.) | min – max | Orange |
  |---|---|---|---|---|
  | A – solo Fed | 0,564 | 0,566 ± 0,021 | 0,531 – 0,596 | 0,592 |
  | B – Fed + WCB | 0,612 | 0,615 ± 0,012 | 0,599 – 0,638 | 0,633 |
  | B2 – solo WCB | 0,603 | 0,603 ± 0,010 | 0,589 – 0,623 | 0,611 |

  I valori di Orange rientrano nella variabilità, ma stanno nella parte alta: il suo singolo addestramento è stato
  un po' fortunato. Stime più prudenti per la relazione: **A ≈ 0,57, B ≈ 0,62, B2 ≈ 0,60**. La conclusione non
  cambia (B un po' meglio di A, B e B2 vicini, tutti molto sotto l'LLM a 0,76).
- **Modello finale** (Fed + tutte le 976 frasi WCB): `modello/rete_finale.joblib` (con `rete_finale.json`:
  impostazioni e verifica). Si carica con `joblib.load` e prende in ingresso gli embedding SBERT delle frasi.
- **Frasi dei discorsi dal 2018** (19.038): `risultati/frasi_discorsi_classificate.csv`, una riga per frase con
  classe, probabilità (`p_falco`, `p_colomba`, `p_neutrale`), `punteggio` (+1 / −1 / 0, per l'indice),
  `punteggio_continuo` (P(falco) − P(colomba), per ordinare le frasi) e `sicurezza` (probabilità più alta).
  Classi: 49% neutrali, 27% falco, 25% colomba (Orange: 49 / 26 / 24).
- **Indice per trimestre** (`risultati/indice_trimestre_locale.csv`): correlazione 0,993 con quello di Orange,
  differenza massima 0,06. Correlazione con la variazione del tasso: t +0,41, t+1 +0,41, t+2 +0,36, t+3 +0,25
  (Orange, stessi trimestri: +0,40 / +0,40 / +0,36 / +0,26).
- **Indice per discorso** (`risultati/indice_discorsi.csv`): 656 discorsi dal 2018 (mediana 19 frasi di politica
  monetaria). Serve per la classe relativa di un nuovo discorso ("più da falco dell'X% dei discorsi").

### Punto 2 – Dashboard

Pagine:
1. **Panoramica**: grafico interattivo indice + tasso sui depositi (stessa impostazione di
   `codice/grafico_indice_tassi.py`: serie standardizzate su un solo asse, valori reali sotto; passando il mouse si
   leggono valori e decisioni). Schede con indice dell'ultimo trimestre, variazione rispetto al trimestre precedente
   e ultima decisione BCE.
2. **Esplora un trimestre**: scelto un trimestre, le frasi più da falco e più da colomba, colorate, con la
   distribuzione delle classi e i discorsi che hanno pesato di più (vedi sotto come si ordinano le frasi).
3. **Il modello**: solo il modello finale (rete α=1): risultati sul test (F1 0,63, accuratezza, matrice di
   confusione) e cosa sa fare e cosa no, per dire quanto fidarsi.
4. **Insight automatici**, scritti con regole semplici sui dati, per esempio: "il tono è salito di 0,22 rispetto
   al trimestre precedente; nel 2022 e nel 2026 salite simili hanno preceduto un rialzo".

La pagina per oratore **non** si fa.

#### Come si decide che una frase è "più da falco" di un'altra

Per ogni frase la rete dà tre probabilità che sommano a 1: P(falco), P(colomba), P(neutrale). Da queste:
- **classe** della frase = la più probabile (è quella che entra nell'indice: +1, −1, 0);
- **punteggio continuo** della frase = **P(falco) − P(colomba)**, da −1 a +1. Una frase con P(falco) = 0,90 e
  P(colomba) = 0,02 vale +0,88; una con P(falco) = 0,50 e P(colomba) = 0,20 vale +0,30. La prima è "più da falco"
  della seconda nel senso che **il modello è più convinto** che spinga verso tassi più alti;
- **sicurezza** del modello = la probabilità più alta (0,90 nel primo esempio, 0,50 nel secondo).

Nella pagina "Esplora un trimestre" le frasi sono ordinate per questo punteggio continuo (colonna di sinistra: le più
chiaramente da colomba; di destra: le più chiaramente da falco) ed è la posizione orizzontale di ogni frase nello
sciame. Attenzione a cosa **non** misura: non è l'intensità
economica della frase (un rialzo di 75 punti base non vale "più" di uno di 25), ma la convinzione del modello. E le
probabilità di una rete neurale non sono perfettamente tarate: vanno usate per ordinare le frasi, non lette come
"90% di probabilità che sia vero". L'indice ufficiale resta la media di +1 / −1 / 0, come nella sezione 7.

### Punto 3 – Analisi di un discorso "in tempo reale"

Due modi di dare il discorso:
1. **testo incollato o link a un discorso sul sito BCE** (la BCE pubblica i testi lo stesso giorno);
2. **link YouTube**: si usano i sottotitoli del video con i loro tempi (libreria `youtube-transcript-api`); se il
   video non li ha, trascrizione dell'audio con Whisper. Il video è incorporato nella pagina e la frase che si sta
   ascoltando si colora mentre il video va avanti. Meglio usare sottotitoli o video della BCE: scaricare l'audio da
   YouTube può violare i termini d'uso.

Il microfono dal vivo non si fa.

Flusso: testo → divisione in frasi → filtro di politica monetaria (stesse parole chiave di
`codice/prepara_frasi_bce.py`) → rete neurale → frase colorata → indice del discorso che si aggiorna.

Durante il discorso:
- frasi di politica monetaria colorate per classe (falco / colomba / neutrale), con **intensità del colore secondo la
  sicurezza** del modello; le altre in grigio come "non rilevanti per la politica monetaria", **escluse dal
  punteggio**;
- indice del discorso aggiornato frase per frase e grafico del suo andamento;
- insight in tempo reale, per esempio "ultime 10 frasi più da falco della media del discorso".

A fine discorso:
- **punteggio finale** = (frasi falco − frasi colomba) / frasi di politica monetaria, come l'indice;
- **classe relativa**, non con soglie fisse: posizione rispetto a tutti i discorsi BCE dal 2018 (per esempio "più
  da falco dell'85% dei discorsi") e rispetto alla media dell'oratore;
- **peso al cambiamento**: confronto con i discorsi recenti e con l'indice dell'ultimo trimestre ("tono in salita
  rispetto agli ultimi 3 mesi"), perché è il cambiamento del tono, più del livello, che ha anticipato le svolte
  (sezione 7);
- **riassunto finale con un LLM** (3–4 frasi: temi principali e frasi che hanno spinto il punteggio). È l'unico uso
  dell'LLM; la chiave API va nei *secrets* di Streamlit, mai nel codice.

### Limiti da dichiarare nella demo

- Il modello è addestrato su resoconti scritti: su discorsi parlati e trascritti sarà meno preciso. Da provare su
  3–4 discorsi veri, confrontando l'analisi del video con quella del testo ufficiale.
- Non è un segnale di trading: il legame con i tassi è moderato (correlazione circa 0,4) e i punti di svolta sono
  pochi.
- Dati di addestramento solo per uso non commerciale (licenze Fed e WCB).

### Punti 2 e 3 – Fatto (20/11/2026): l'app Streamlit

Avvio, dalla cartella `progetto_BCE` (poi si apre http://localhost:8501):
```
.venv/bin/streamlit run app/app.py            # Mac / Linux
.venv\Scripts\streamlit.exe run app/app.py    # Windows
```
L'ambiente si crea con **Python 3.12 o successivo** (consigliato 3.13: le versioni bloccate in `requirements.txt`, per
esempio numpy 2.5, pandas 3.1 e networkx 3.7, non si installano con Python 3.11) e
`pip install -r requirements.txt`; passaggi completi in `ISTRUZIONI_AVVIO.md`. Al primo uso dell'analizzatore l'app
scarica da Hugging Face il modello SBERT (`paraphrase-multilingual-MiniLM-L12-v2`, circa 470 MB, poi resta in cache).
L'app legge `modello/` e `risultati/`, già presenti: `codice/modello_locale.py` (punto 1) serve solo a rigenerarli.

File:
- `app/app.py`: l'app (quattro pagine, menu a sinistra);
- `app/live.html`: il pannello in tempo reale dell'analizzatore (HTML + JavaScript), riempito da `live_html` in
  `app/app.py` con le frasi classificate e il verdetto;
- `codice/analisi_discorso.py`: il motore dell'analizzatore (divisione in frasi e filtro di politica monetaria
  presi da `prepara_frasi_bce.py`, classificazione con il modello finale, testo dal sito BCE, sottotitoli YouTube,
  riassunto con l'LLM);
- `.streamlit/config.toml`: tema dell'app (colori e caratteri); `.streamlit/secrets.toml.esempio`: modello per la
  chiave dell'LLM.
- `app/assets/logo.svg`, `app/assets/emblema.svg`: logo dell'app (anello di 12 stelle con il simbolo €).

**Aspetto grafico**: tema "Eurosistema" (blu UE, oro delle stelle, quadrante a stelle nella Panoramica). Cosa è stato fatto,
con quali strumenti e come: sezione 9.

Pagine:
1. **Panoramica**: testata con il **quadrante a stelle** (indice dell'ultimo trimestre concluso, posizione rispetto ai
   trimestri dal 2018 e variazione sul trimestre precedente); quattro schede (tasso sui depositi con l'ultima
   decisione, indice provvisorio del trimestre in corso, correlazione con il trimestre successivo, discorsi
   analizzati); grafico interattivo indice + tasso con due viste, "Stessa scala" (serie standardizzate) e "Valori
   reali", con le fasce dei cicli di rialzi e di tagli; **"Cosa dicono i dati"** (insight automatici, calcolati sui
   dati a ogni avvio): livello dell'indice rispetto ai trimestri dal 2018, cambiamento rispetto al trimestre
   precedente, precedenti storici (quante volte un movimento simile su due trimestri è stato seguito da una decisione
   nella stessa direzione), confronto con la regola banale, correlazione con il trimestre successivo, avviso sul
   trimestre in corso.
2. **Esplora un trimestre**:
   - *scelta*: grafico con una barra per trimestre (colore = indice, da azzurro a rosso) e il tasso sui depositi a
     fine trimestre a gradini in oro; **clic su una barra** per aprire quel trimestre, oppure cursore e frecce ‹ ›;
   - *testata del trimestre*: periodo (es. "luglio – settembre 2026"), indice, posizione rispetto ai trimestri
     conclusi dal 2018, variazione sul trimestre precedente, quadrante a stelle, e tre fatti: frasi e discorsi, tasso
     a fine trimestre con la variazione, decisione del trimestre successivo (▲ rialzo, ▼ taglio, = fermo);
   - *tutte le frasi del trimestre*: barra con le quote falco / neutrale / colomba e i conteggi, poi lo **sciame**
     di tutte le frasi (come si legge: sotto, "Come si legge lo sciame delle frasi");
   - *le frasi agli estremi*: le più da colomba a sinistra e le più da falco a destra (5, 10 o 20 per lato), ordinate
     per P(falco) − P(colomba), con un misuratore divergente del punteggio, oratore, data e sicurezza. Un riquadro
     spiega come si ordinano le frasi;
   - *i discorsi del trimestre* (almeno 5 frasi): barre divergenti dal più da falco al più da colomba, con titolo e
     conteggi al passaggio del mouse; la tabella completa è in un riquadro apribile.
3. **Il modello**: solo il modello finale. F1 sul test 0,61 (accuratezza 61%, kappa 0,41; variabilità su 10
   addestramenti 0,62 ± 0,01), matrice di confusione, F1 per classe, cosa sa fare e cosa no.
4. **Analizza un discorso**: oratore facoltativo (per il confronto con i suoi discorsi), poi due schede.
   - *Testo o link al sito BCE*: link `ecb.europa.eu` (il testo viene letto dalla pagina) oppure testo incollato.
     Con "Scorrimento in tempo reale" le frasi compaiono una alla volta nel pannello in tempo reale (sotto); la
     velocità (1–10 frasi al secondo), la pausa e "Ricomincia" si comandano dal pannello, senza ricaricare la
     pagina. Senza scorrimento si vedono subito il testo colorato e il risultato.
   - *Video YouTube*: si scaricano i sottotitoli inglesi con i loro tempi; il video è incorporato nel pannello e,
     mentre va avanti, ogni frase compare nel momento in cui viene pronunciata. Clic sul sismografo o su una riga
     della trascrizione per saltare a quel punto del video; "Mostra subito tutto" per non aspettare la fine.

   Pannello in tempo reale (`app/live.html`, uguale per testo e video): quadrante a 12 stelle con la lancetta che
   segue l'indice del discorso finora (scala da −0,6 a +0,6; il 90% dei discorsi BCE dal 2018 sta tra −0,46 e
   +0,35); numero dell'indice; conteggi falco / neutrali / colomba; andamento dell'indice frase dopo frase (area
   rossa sopra lo zero, azzurra sotto); confronto delle ultime 10 frasi con la media del discorso; avviso di
   **svolta** quando l'indice passa dal lato colomba al lato falco o viceversa (almeno 5 frasi, soglia ±0,05);
   didascalia della frase in ascolto con classe, probabilità (falco / neutrale / colomba) e sicurezza;
   **sismografo del tono**: una tacca per ogni frase di politica monetaria lungo tutto il discorso, in su se falco e
   in giù se colomba, alta quanto |P(falco) − P(colomba)|, che si colora quando la testina d'oro la supera;
   trascrizione che scorre; "finora agli estremi" (le 2 frasi più da falco e le 2 più da colomba ascoltate fin
   lì); a fine discorso il verdetto. Il verdetto completo e il riassunto con l'LLM sono anche nel riquadro apribile
   sotto il pannello.

   Colori: falco vermiglio, colomba azzurro cielo, neutrale grigio; intensità secondo la sicurezza del modello; frasi non di
   politica monetaria in grigio chiaro ed escluse dal punteggio. Risultato finale: punteggio; classe relativa
   rispetto ai 420 discorsi BCE dal 2018 con almeno 10 frasi (colomba / tendenza colomba / nella media / tendenza
   falco / falco, a fasce di 20 percentili); cambiamento rispetto all'ultimo trimestre concluso, alla media degli
   ultimi 10 discorsi e, se indicato, alla media dell'oratore; le 3 frasi più da falco e più da colomba; riassunto
   con l'LLM (pulsante; serve `KIMI_API_KEY` in `.streamlit/secrets.toml` o come variabile d'ambiente: copiate
   `secrets.toml.esempio` in `secrets.toml` e incollate la chiave; il file non va condiviso).

#### Come si legge lo sciame delle frasi ("Esplora un trimestre")

Ogni punto è una frase di politica monetaria del trimestre (funzione `grafico_sciame` in `app/app.py`).
- **Asse orizzontale = punteggio continuo** della frase, P(falco) − P(colomba), da −1 a +1 (lo stesso usato per
  ordinare le frasi, sezione 8). A destra le frasi di cui il modello è convinto che spingano verso tassi più alti, a
  sinistra verso tassi più bassi, vicino a 0 quelle in cui falco e colomba si equivalgono.
- **Asse verticale: nessun significato.** I punti sono sparpagliati in su e in giù solo per non sovrapporsi. Lo
  sparpagliamento è più largo dove le frasi sono più fitte: l'asse x è diviso in 40 fasce e l'ampiezza verticale di
  ogni fascia è proporzionale al numero di frasi che contiene (da 0,12 a 1 rispetto alla fascia più piena). Così lo
  sciame si "gonfia" dove le frasi si accumulano, come un violin plot. Lo sparpagliamento usa un seme fisso: il
  disegno non cambia ricaricando la pagina.
- **Colore = classe** della frase (la probabilità più alta: è quella che entra nell'indice con +1, −1, 0);
  **intensità = sicurezza** del modello (punto pieno = sicuro, sbiadito = incerto).
- **Linea tratteggiata d'oro = indice del trimestre**, (frasi falco − frasi colomba) / frasi.

Colore e posizione vengono dalle stesse tre probabilità, quindi le classi stanno in zone precise:
- **falco** solo a destra dello zero (se vince falco, P(falco) > P(colomba));
- **colomba** solo a sinistra dello zero;
- **neutrale** al centro e mai oltre ±0,5 (se vince neutrale, P(falco) e P(colomba) restano entrambe sotto 0,5).

Esempio, 2026-T3 (197 frasi): 97 neutrali (da −0,41 a +0,39, colonna grigia fitta sullo zero, sicurezza media
0,87); 78 falco (da +0,06 a +1,00; 50 oltre +0,8, gruppo rosso pieno sul bordo destro, sicurezza media 0,85);
22 colomba (da −0,97 a −0,07, sparse, più sbiadite, sicurezza media 0,68). Indice (78 − 22) / 197 = **+0,28**.

Attenzione: la linea d'oro **non è il baricentro dei punti**. L'indice si calcola dalle classi (+1, −1, 0), non dalle
posizioni sull'asse x: nel 2026-T3 la media dei punteggi continui è +0,26, vicina ma non uguale. La linea serve a
vedere dove cade il trimestre sulla stessa scala delle frasi.

Provato il 20/11/2026:
- link al discorso di Schnabel del 5/11/2026 ("Navigating stormy waters"): 181 frasi, 69 di politica monetaria,
  punteggio +0,10, "tendenza falco" (più da falco del 65% dei discorsi), più da colomba dell'ultimo trimestre
  (−0,18), in linea con gli ultimi 10 discorsi;
- video della conferenza stampa del 29/10/2026 (`youtube.com/watch?v=8sCMllo0zSg`): 302 frasi dai sottotitoli,
  78 di politica monetaria, punteggio +0,38, classe "falco" (più da falco del 95% dei discorsi); sincronizzazione
  con il video verificata.
- riassunto con l'LLM (Kimi K3.1 Pro, ragionamento "low") sul discorso di Schnabel: funziona, circa un minuto di
  attesa. La chiave è in `.streamlit/secrets.toml` (da non consegnare).

Limiti dell'app:
- **YouTube**: funziona solo con video che hanno sottotitoli (anche automatici). La trascrizione dell'audio con
  Whisper prevista nel piano **non** è stata fatta: per un video senza sottotitoli si incolla il testo. Con
  sottotitoli automatici senza punteggiatura il testo viene spezzato a blocchi di circa 30 parole, meno precisi
  delle frasi vere. I tempi delle frasi sono quelli del pezzo di sottotitolo in cui la frase comincia.
- Una conferenza stampa non è un discorso: domande dei giornalisti e saluti entrano nel testo (le frasi non di
  politica monetaria restano comunque escluse dal punteggio).
- Il filtro di politica monetaria è a parole chiave: frasi come "growth has weakened" senza parole chiave restano
  grigie anche se rilevanti, come nel dataset su cui è calcolato l'indice.
- Il link funziona solo per pagine di `ecb.europa.eu`; se la BCE cambia l'impaginazione del sito va adattata la
  lettura del testo (`testo_da_url_bce`).

## 9. Grafica dell'app: cosa è stato fatto e con quali strumenti (20/11/2026)

Obiettivo: rendere la dashboard moderna e d'impatto restando nel tema dell'area dell'euro (blu UE, oro, 12 stelle),
senza l'aspetto da "modello generico". Punto di partenza: l'elenco di strumenti in
`possibili skill per migliroare design.txt`.

### Strumenti usati e come

| Strumento | Cos'è | Da dove / versione | Come è stato installato | Come è stato usato |
|---|---|---|---|---|
| **frontend-design** | Skill per Claude Code (istruzioni in `SKILL.md`, nessun codice) | Skill ufficiale di Anthropic, repository `anthropics/skills`, commit `e4bb5d9` (16/11/2026), licenza Apache 2.0 | `git clone --depth 1` del repository in una cartella temporanea, lettura del file, copia della cartella `skills/frontend-design` in `.claude/skills/frontend-design` | Metodo di lavoro: piano con pochi colori e caratteri scelti per il tema; confronto con i difetti tipici delle interfacce generate (fondo avorio con titoli serif, schede tutte uguali con la stessa ombra, etichette in maiuscoletto, separatori "·"); un solo elemento forte (il quadrante) e il resto sobrio; un solo momento animato nella prima passata (nella seconda il movimento è stato esteso su richiesta, con un solo vocabolario comune); controllo finale con screenshot su desktop e telefono |
| **ui-ux-pro-max** | Skill per Claude Code con un piccolo database locale (CSV) di palette, coppie di caratteri, regole UX e per i grafici, più uno script di ricerca | Repository `nextlevelbuilder/ui-ux-pro-max-skill`, versione 2.15.0, commit `2de1134` (19/11/2026), licenza MIT | `git clone --depth 1` in una cartella temporanea; controllo degli script (`scripts/*.py`: solo libreria standard di Python, nessun accesso alla rete, nessun comando di sistema); copia di `.claude/skills/ui-ux-pro-max` senza la cartella `tests`, con il file `LICENSE` | Ricerche con lo script (vedi sotto). Presi: IBM Plex Sans per il testo (coppia "Financial Trust"), oro scuro come accento sul blu, regole per i grafici (nomi delle serie scritti sulle linee, serie distinte non solo dal colore), lista di controllo finale (contrasto, focus visibile, movimento ridotto, schermi piccoli). Scartati: tema scuro OLED con accento verde e caratteri Fira (generici, fuori tema BCE) |
| **Streamlit** 1.67 | Il framework dell'app | già nell'ambiente `.venv` | nessuna installazione nuova | Tema in `.streamlit/config.toml` (colori, caratteri, barra laterale); CSS e HTML con `st.html`; `st.container(key="card…")` per dare stile ai contenitori; `st.logo` per il logo |
| **Plotly** 7.1 | Grafici interattivi | già nell'ambiente `.venv` | nessuna installazione nuova | Fasce dei cicli dei tassi (`add_vrect`), nomi delle serie a fine linea (`add_annotation`) al posto della legenda, etichette al passaggio del mouse su fondo blu abisso |
| **Google Fonts** | Caratteri gratuiti (licenza OFL) | *Bricolage Grotesque* (titoli e numeri), *IBM Plex Sans* (testo) | nessuna: il browser li scarica da Google Fonts | Dichiarati in `config.toml` (`font`, `headingFont`) e nel pannello in tempo reale (`app/live.html`) |
| **CSS + SVG** | Grafica disegnata nel codice | `app/app.py` | — | Stelle, quadrante e rosetta generati in Python e inseriti come immagini di sfondo (`svg_uri`), perché Streamlit toglie gli SVG scritti direttamente nell'HTML. Animazioni in CSS (`@keyframes` in `STILE`) |
| **HTML + JavaScript** | Pannello in tempo reale dell'analizzatore | `app/app.py` (`live_html`) + `app/live.html` | — | Pagina inserita con `st.components.v1.html`: quadrante, sismografo, trascrizione e verdetto disegnati in SVG e aggiornati nel browser (nessun rerun di Streamlit durante lo scorrimento); API IFrame di YouTube per leggere il tempo del video e saltare a un punto; l'iframe adatta da solo la sua altezza (`ResizeObserver`) |

**Nessun pacchetto Python o npm in più.** Le librerie dell'app sono quelle di `requirements.txt` (le stesse della
sezione 8).

Ricerche fatte con `ui-ux-pro-max`, dalla cartella del progetto (`-I` esegue Python isolato, senza caricare file della
cartella corrente):
```
python3 -I .claude/skills/ui-ux-pro-max/scripts/search.py "central bank monetary policy analytics dashboard institutional trust" --design-system -p "Falchi e Colombe"
python3 -I .claude/skills/ui-ux-pro-max/scripts/search.py "financial dashboard banking trust" --domain color
python3 -I .claude/skills/ui-ux-pro-max/scripts/search.py "financial editorial data dashboard" --domain typography
python3 -I .claude/skills/ui-ux-pro-max/scripts/search.py "time series comparison two series" --domain chart
```

Le due skill restano in `.claude/skills/`: Claude Code le usa da solo quando si lavora sulla grafica, oppure si
chiamano per nome (`/frontend-design`, `/ui-ux-pro-max`). Non servono per far funzionare l'app: se la cartella
`.claude/` non viene consegnata, l'app non cambia.

### Strumenti valutati e non usati

| Strumento | Perché no |
|---|---|
| shadcn/ui, Magic UI, Motion Primitives, Tremor, BoardUI | Componenti React + Tailwind: per usarli bisognerebbe riscrivere l'app in React e servire il modello Python a parte. Le idee utili (pannelli a griglia, un solo momento animato, accenti luminosi) sono state rifatte in CSS |
| shadcn-admin, next-shadcn-admin-dashboard | Modelli di dashboard React/Next.js: stesso motivo |
| taste-skill (`Leonxlnx/taste-skill`) | Pensata per siti vetrina con fotografie e sezioni a tutto schermo, non per una dashboard di dati; installa un controllo automatico (hook) che esegue script Node a ogni risposta di Claude Code |
| superdesign-skill, better-design | Non necessarie: colori e caratteri sono già scritti in `config.toml` e in testa a `app/app.py` |

### Cosa è cambiato

Tema "Eurosistema", ispirato all'area dell'euro, senza loghi ufficiali:
- **colori**: blu della bandiera europea `#003399`, blu abisso `#001233` (barra laterale e testate), oro delle stelle
  `#ffcc00` solo per gli accenti (stelle, voce di menu attiva, lancetta, bordo di focus), fondo carta fredda
  `#f3f5fa` al posto dell'avorio; falco vermiglio `#d2402f`, colomba azzurro cielo `#1e95d0` (distinto dal blu UE,
  che resta il colore dell'indice), neutrale grigio `#8a94aa`; nei grafici indice in blu e tasso in oro scuro
  `#a87800`;
- **caratteri**: *Bricolage Grotesque* per titoli e numeri, *IBM Plex Sans* per il testo (al posto di Source Serif 4
  e Inter), cifre tabellari perché i numeri restino allineati;
- **quadrante a stelle** nella Panoramica: le 12 stelle europee a semicerchio da colomba (−0,4) a falco (+0,4); la
  scala copre tutti i trimestri dal 2018 (minimo −0,34, massimo +0,31). Una lancetta d'oro indica l'indice
  dell'ultimo trimestre concluso e la stella più vicina si accende. All'apertura le stelle compaiono una alla volta
  e la lancetta ruota (disattivato se il sistema chiede meno movimento, `prefers-reduced-motion`). Funzione
  `quadrante_stelle` in `app/app.py`; dalla seconda passata (sotto) è anche nella pagina del trimestre e, in versione
  JavaScript, nel pannello in tempo reale;
- **testate** blu abisso con una rosetta a linee sottili come i motivi di sicurezza delle banconote (*guilloche*,
  funzione `guilloche`): grande nella Panoramica, compatta con l'anello di stelle nelle altre pagine;
- **schede numeriche e insight** in pannelli unici divisi in celle (come un registro), non in schede separate; ogni
  insight ha un pallino col colore della direzione (falco, colomba, informativo); titoli di sezione con una piccola
  stella d'oro;
- **grafico indice + tasso**: fasce colorate per i cicli di rialzi e di tagli (calcolate da `tassi_bce.csv`,
  funzione `cicli_tassi`) al posto delle linee tratteggiate; nomi delle serie a fine linea al posto della legenda;
  indice con punti e tasso a gradini, così le serie non si distinguono solo dal colore;
- **analizzatore**: stella come segnaposto sulla scala colomba–falco del verdetto, pallino rosso pulsante "In tempo
  reale", stesso stile anche nel lettore YouTube;
- **testi**: etichette in minuscolo, niente maiuscoletto né separatori "·", nessun sopratitolo sopra i titoli;
- **logo** (`app/assets/logo.svg`): scritta in carattere senza grazie, "il tono della BCE" in oro;
- in fondo a ogni pagina resta la nota: progetto universitario, **non affiliato né approvato dalla BCE**, fonti e
  licenze, non è un consiglio di investimento.

File modificati: `app/app.py` (palette, simboli, stile, Panoramica, grafico, pannello in tempo reale, lettore video),
`.streamlit/config.toml` (tema), `app/assets/logo.svg`. Calcoli, modello e dati non sono cambiati.

Provato il 20/11/2026 nel browser, a 1400 px e a 375 px (telefono): tutte e quattro le pagine e l'analisi di un testo
incollato (verdetto, frasi, pannello in tempo reale).

Note pratiche:
- **dopo aver cambiato `config.toml` va riavviata l'app** (Streamlit legge il tema solo all'avvio); le modifiche a
  `app/app.py` si vedono ricaricando la pagina;
- i caratteri arrivano da Google Fonts: senza connessione l'app usa quelli di sistema;
- per cambiare un colore va modificato in due posti: le costanti in testa a `app/app.py` (sezione "palette") e
  `.streamlit/config.toml`.

### Movimento e pagine più dinamiche (20/11/2026, seconda passata)

Stesso tema "Eurosistema", con un solo vocabolario di movimento (tutto spento con `prefers-reduced-motion`):
- **ovunque**: la rosetta delle testate ruota lentissima (filigrana); le celle del registro entrano in sequenza e un
  filo d'oro le "timbra"; i numeri emergono dal basso; le barre delle classi si svelano da sinistra; gli insight
  entrano uno dopo l'altro; nella pagina del modello le barre dell'F1 crescono.
- **Esplora un trimestre**, rifatta: in alto un grafico con una barra per trimestre (colore = indice) e il tasso a
  gradini: **si clicca una barra** per aprire il trimestre (oppure cursore e frecce ‹ ›). Sotto, una testata scura con
  periodo, indice, variazione sul trimestre precedente, il quadrante a stelle e tre fatti (frasi e discorsi, tasso a
  fine trimestre, decisione successiva con freccia colorata). Poi lo **sciame di tutte le frasi** del trimestre da −1
  (colomba) a +1 (falco), con la frase al passaggio del mouse; le frasi agli estremi in due colonne (colomba a
  sinistra, falco a destra, come sulla scala) con un misuratore divergente del punteggio e la scelta 5/10/20 per lato;
  i discorsi come **barre divergenti** (tabella completa in un riquadro apribile).
- **Analizza un discorso**, pannello in tempo reale riscritto in `app/live.html` (HTML + JavaScript, caricato da
  `live_html` in `app/app.py`), lo stesso per video YouTube e testo: console scura con il video (o la frase in grande),
  **quadrante a 12 stelle con la lancetta che segue l'indice del discorso** (scala ±0,6) e un'onda colorata a ogni
  frase, numero che scorre, conteggi falco/neutrali/colomba, andamento con area rossa/azzurra sopra/sotto lo zero,
  avviso di **svolta** quando il tono cambia lato; didascalia della frase in ascolto che compare parola per parola con
  le tre probabilità; **sismografo del tono** lungo tutto il discorso (tacche in su falco, in giù colomba, che si
  colorano al passaggio della testina; clic per saltare a quel punto); trascrizione che scorre (clic su una riga per
  saltare) e "finora agli estremi"; a fine discorso il verdetto con il punteggio che conta e la stella che scorre
  sulla scala. Per il testo lo scorrimento avviene nel browser: pausa, velocità da 1 a 10 frasi al secondo,
  ricomincia. L'iframe si adatta da solo all'altezza del contenuto.

Strumenti: le stesse due skill (`frontend-design` per il metodo, `ui-ux-pro-max` per le regole su movimento ridotto e
schermi piccoli), Plotly per i nuovi grafici (selezione di una barra con `on_select`, sciame, barre divergenti),
HTML + JavaScript per il pannello in tempo reale. Nessuna libreria nuova.

File modificati o aggiunti: `app/app.py` (stile e movimento, pagina "Esplora un trimestre" riscritta, `live_html` al
posto di `player_html` e del vecchio scorrimento in Python), `app/live.html` (nuovo), `.claude/launch.json` (avvio
dell'anteprima su Windows), `ISTRUZIONI_AVVIO.md` (versione di Python richiesta). Calcoli, modello e dati non sono
cambiati: con il video `8sCMllo0zSg` il verdetto è lo stesso di prima (+0,38, 78 frasi di politica monetaria su 302,
"falco", più da falco del 95% dei discorsi).

Ambiente ricreato su Windows il 20/11/2026: Python 3.13 installato per il solo utente
(`winget install Python.Python.3.13 --scope user`), `.venv` ricreato e `pip install -r requirements.txt` senza errori.

Ambiente ricreato anche su Mac il 20/11/2026: Python 3.13.5 (Anaconda,
`/opt/anaconda3/bin/python3.13 -m venv .venv`), `pip install -r requirements.txt` senza errori; il modello SBERT era
già nella cache di Hugging Face del computer. Le cartelle arrivate dallo zip (`progetto_BCE_portabile`) erano in sola
lettura: prima di creare `.venv` è servito `chmod -R u+w .`. La cartella è stata poi spostata al posto della copia
precedente (`AIDA/AI and Computer Vision/progetto_BCE`) e `.venv` è stato ricreato lì. La copia precedente, senza le
modifiche di questa seconda passata, è in `../progetto_BCE_vecchio`. Provato nel browser: Panoramica, Esplora un trimestre, analisi di un testo
incollato con il pannello in tempo reale e il verdetto. La chiave dell'LLM è presente in `.streamlit/secrets.toml`
ma il riassunto non è stato provato.

Provato nel browser a 1400 px e 375 px: tutte le pagine, clic sulle barre dei trimestri e frecce, testo incollato con
scorrimento (pausa, velocità, ricomincia), video `8sCMllo0zSg` (sincronizzazione, salto dal sismografo, verdetto).
Non provato: il riassunto con l'LLM (manca la chiave in `.streamlit/secrets.toml`).

Note pratiche:
- le modifiche a `app/live.html` si vedono rilanciando l'analisi (pulsante "Analizza il discorso" o "Prepara il
  video"): il file viene riletto a ogni analisi;
- il video YouTube deve permettere l'incorporamento in altre pagine; in caso contrario il pannello resta fermo e si
  usa "Mostra subito tutto";
- su Windows, al primo download del modello Hugging Face avvisa che la cache non usa i collegamenti simbolici: è
  solo un avviso, il modello funziona.

## Cosa manca e va fatto da voi

- Nessuna etichettatura a mano: addestramento e test BCE vengono da WCB (sezione 3b).
- (Facoltativo) controllo sui discorsi con frasi etichettate a mano (sezione 1c).
- (Facoltativo) calendario di tutte le riunioni di politica monetaria, per il test previsivo (sezione 5, punto 3).
- (Facoltativo) rendimenti dei titoli di Stato dell'area euro dall'ECB Data Portal, per la verifica sul mercato.
