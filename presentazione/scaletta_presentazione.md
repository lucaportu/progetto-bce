# Scaletta della presentazione – Il tono della BCE: un indice falco/colomba dai discorsi

Documento di lavoro per costruire il PowerPoint. Per ogni slide: titolo, messaggio chiave (la frase che il pubblico deve
ricordare), contenuto da mettere, grafici o immagini, note per chi parla. Fonte di tutti i numeri:
`Metodologia_progetto_BCE.pdf` (relazione completa) e `README.md`, sezione 7.

## Istruzioni per chi costruisce il PPT

- **11 slide**, sintetiche: al massimo 3–4 punti per slide, frasi brevi, numeri grandi dove servono. I dettagli vanno
  nelle note del relatore, non sulla slide.
- **Lingua**: italiano. Numeri con la virgola decimale (0,63, non 0.63).
- **Stile**: tema area dell'euro, moderno e non da template. Blu UE `#003399`, blu scuro `#001233` per testate e
  copertina, oro `#ffcc00` solo per accenti (stelle, sottolineature, il numero più importante). Colori delle classi:
  falco `#d2402f` (rosso vermiglio), colomba `#1e95d0` (azzurro), neutrale `#8a94aa` (grigio). Le 12 stelle europee
  si possono usare come motivo grafico (copertina, separatori), **senza loghi ufficiali BCE**: è un progetto
  universitario, non affiliato alla BCE.
- **Grafici pronti** in `presentazione/grafici/` (PNG ad alta risoluzione, già nei colori del tema). Si rigenerano con
  `python3 presentazione/grafici/crea_grafici.py` (lanciato dalla cartella `presentazione/grafici/`; serve matplotlib):

  | File | Cosa mostra | Slide |
  |---|---|---|
  | `g_confronto.png` | F1 pesato sul test per classe più frequente, A, B2, B, C (LLM), con intervalli di confidenza | 7 |
  | `g_matrici.png` | Matrici di confusione di B (rete) e C (Kimi) affiancate | 7 (o slide di riserva) |
  | `g_cv.png` | Cross-validation contro test per i 4 modelli in A, B2, B | slide di riserva |
  | `grafico_indice_tassi.png` | Indice trimestrale e tasso sui depositi 2018–2026, con le svolte dei tassi | 8 |
  | `g_ritardi.png` | Correlazione dell'indice con la variazione del tasso a t, t+1, t+2, t+3, con la linea della persistenza | 8 |

- **Da aggiungere a mano**: lo screenshot del workflow Orange (slide 5) e gli screenshot dell'app (slide 9).
- Quando un grafico occupa la slide, il testo si riduce a un titolo che dice la conclusione e a 2–3 didascalie.

---

## Slide 1 – Copertina

- **Titolo**: Il tono della BCE
- **Sottotitolo**: un indice falco/colomba dai discorsi del Comitato esecutivo
- **Riga sotto**: Progetto AI & Computer Vision · nomi del gruppo · data
- **Grafica**: fondo blu scuro, anello di 12 stelle oro con una lancetta (come un quadrante che va da "colomba" a
  "falco"): è lo stesso simbolo dell'app.

---

## Slide 2 – Il bisogno e la domanda

**Messaggio chiave**: le decisioni sui tassi della BCE muovono mutui, prestiti e mercati; i discorsi le preparano, ma
sono troppi per leggerli a mano.

Contenuto:
- **Il bisogno**: banche, gestori di fondi e tesorerie aziendali prendono decisioni che dipendono dai tassi futuri
  (prezzo dei mutui, durata dei portafogli, coperture dal rischio di tasso).
- **Il problema**: 100–115 discorsi l'anno, oltre 60.000 frasi di politica monetaria dal 1997; una lettura a mano è
  lenta e soggettiva.
- **La domanda** (in evidenza, grande, con sottolineatura oro):
  > Si può misurare in modo automatico il tono (falco / colomba) dei discorsi BCE, e questo tono anticipa le decisioni
  > sui tassi?
- Riquadro piccolo con le tre etichette e un esempio inventato per ciascuna:
  - **falco** = spinge verso tassi più alti ("inflation remains too high, further increases may be needed");
  - **colomba** = verso tassi più bassi ("growth is weakening, we stand ready to ease");
  - **neutrale** = nessuna direzione.

Note per chi parla: l'unità di analisi è la frase. Ogni frase riceve +1 (falco), −1 (colomba), 0 (neutrale) e la media
per trimestre è l'indice. Lavorare sulle frasi rende il risultato spiegabile (si possono mostrare le frasi che hanno
spinto l'indice) ed è l'unico livello in cui esistono dati etichettati da esperti.

---

## Slide 3 – I dati e perché usiamo anche la Fed

**Messaggio chiave**: per la BCE ci sono solo 976 frasi etichettate da esperti; per compensare si prendono in
prestito 2.339 frasi della Fed, etichettate con lo stesso schema.

Contenuto, metà sinistra (tabella compatta):

| Dati | Frasi | Uso |
|---|---|---|
| Fed – *Trillion Dollar Words* (2023), etichettate | 2.339 | addestramento |
| BCE – WCB 2015–2022, etichettate (resoconti delle riunioni) | 732 | addestramento |
| BCE – WCB 2023–2024, etichettate | 244 | **test comune**, mai visto in addestramento |
| Discorsi BCE dal 2018, senza etichette | 19.038 | classificati dal modello → indice |
| Tassi BCE (tasso sui depositi) | – | verifica dell'indice |

Contenuto, metà destra ("Perché la Fed"):
- **Pro**: stesso gruppo di ricerca e stesse tre classi; il lessico delle banche centrali è in gran parte comune;
  il triplo degli esempi.
- **Contro**: mandati diversi (la Fed guarda anche all'occupazione), sigle e gergo diversi (APP, PEPP, *deposit
  facility rate* contro *federal funds rate*), stile diverso.
- In fondo, in oro: **non lo assumiamo: lo misuriamo** (slide 4).

Note per chi parla: i discorsi sono stati puliti in Python (tolte intestazioni, note, bibliografia, frasi troppo corte
o lunghe) e filtrati con parole chiave di politica monetaria: da 346.464 frasi a 60.764. Il test è diviso **per anno**
(2023–2024) e non a caso, così si simula l'uso reale (si addestra sul passato, si classifica il futuro) e frasi dello
stesso resoconto non finiscono sia in addestramento sia in test. Il 2023–2024 è anche un banco di prova difficile:
picco dei tassi e inizio dei tagli.

---

## Slide 4 – Il disegno sperimentale

**Messaggio chiave**: quattro approcci che cambiano una sola cosa, i dati di addestramento, misurati tutti sulle
stesse 244 frasi BCE.

Contenuto (quattro riquadri o una riga di quattro "carte"):
- **A – solo Fed** (2.339 frasi): un modello di un'altra banca centrale funziona sulla BCE?
- **B – Fed + BCE** (3.071): quanto aiutano gli esempi BCE?
- **B2 – solo BCE** (732): la Fed serve davvero?
- **C – LLM senza addestramento** (riceve solo la guida degli annotatori): un modello linguistico generale fa meglio?

Sotto, una riga con le tre regole metodologiche:
- il test si guarda **una volta sola**: tutte le scelte si fanno in cross-validation sull'addestramento;
- **stesso test per tutti** e intervalli di confidenza bootstrap sulle stesse frasi;
- sempre un **termine di paragone banale** (classe più frequente per i modelli; "la prossima decisione sarà come
  l'ultima" per l'indice).

Note per chi parla: i confronti che contano sono B − A (valore dei dati BCE), B − B2 (valore dei dati Fed),
C − B (guadagno dell'LLM).

---

## Slide 5 – Il workflow in Orange

**Messaggio chiave**: tutto il metodo, dalla valutazione all'indice, sta in un unico workflow Orange di 37 widget.

Contenuto:
- **Screenshot del workflow** `Workflow BCE AI&CV.ows`, a tutta slide (da fare in Orange: aprire il workflow e
  catturare la finestra).
- Sopra lo screenshot, **quattro riquadri colorati semitrasparenti** con un'etichetta ciascuno, per far seguire il
  percorso:
  1. **Valutazione A, B, B2**: Corpus → Concatenate → Document Embedding → Logistic Regression e Neural Network →
     Test and Score → Confusion Matrix
  2. **Approccio C**: Corpus del test → Python Script "LLM" (Kimi K3.1 Pro) → Confusion Matrix
  3. **Modello finale**: Fed + tutte le 976 frasi BCE → Document Embedding → Neural Network
  4. **Indice e verifica**: discorsi dal 2018 → Predictions → Formula (+1/−1/0) → Group by trimestre → Merge Data con
     i tassi → Correlations e Line Chart
- Nessun altro testo: le scelte si spiegano nella slide dopo.

Note per chi parla: seguire con il puntatore il flusso da sinistra a destra, blocco per blocco. Prima di condividere
lo screenshot o il file `.ows`, controllare che nel nodo LLM la chiave API sia cancellata.

---

## Slide 6 – Le scelte del workflow

**Messaggio chiave**: ogni scelta risponde allo stesso vincolo: pochi dati BCE e due banche che parlano in modo
diverso.

Contenuto (tre o quattro colonne con un'icona ciascuna):
- **Embedding SBERT, non Bag of Words**
  - coglie il significato: "restrictive stance" e "tightening" sono vicini anche senza parole in comune;
  - **Fed e BCE usano un gergo diverso**: Bag of Words impara il peso delle singole parole, quindi le parole tipiche
    della BCE, mai viste nei dati Fed, restano senza peso; con gli embedding il modello impara il concetto, non il
    gergo;
  - 384 colonne dense invece di migliaia di colonne quasi vuote: meno rischio di imparare a memoria.
- **Modelli semplici e regolarizzati**
  - regressione logistica (riferimento lineare) e rete neurale con 100 neuroni;
  - due livelli di regolarizzazione ciascuno (C = 1 / 10; α = 0,01 / 1): con pochi dati vince la regolarizzazione
    forte.
- **Valutazione onesta**
  - cross-validation a 10 fold per scegliere, test 2023–24 una volta sola per il numero finale;
  - **F1 pesato** (le classi non sono bilanciate: 40% neutrali) e kappa di Cohen per l'accordo con gli esperti.
- **LLM con le istruzioni degli esperti**
  - Python Script che manda una frase alla volta a Kimi K3.1 Pro con la guida di annotazione WCB completa;
  - risposte salvate in un file: ripetibile e senza pagare due volte.

Note per chi parla: l'indice usa il voto della frase (+1/−1/0) e non le probabilità, perché è più leggibile ("26% di
frasi falco contro 24% colomba") e le probabilità della rete non sono tarate.

---

## Slide 7 – Risultati del confronto

**Messaggio chiave**: i dati BCE aiutano più di quelli Fed, ma il vero salto lo fa l'LLM.

Grafico principale: **`g_confronto.png`** (barre con intervalli di confidenza). Se c'è spazio, `g_matrici.png` in
piccolo a destra; altrimenti va in una slide di riserva.

Didascalie accanto al grafico (3 righe, con il numero in grande):
- **0,59 → 0,63**: aggiungere 732 frasi BCE al modello Fed migliora (+0,04), ma la differenza resta dentro
  l'incertezza del test.
- **0,61 con le sole frasi BCE**: 732 frasi BCE rendono più di 2.339 frasi Fed. La pertinenza conta più del volume;
  la Fed aggiunge poco (+0,02).
- **0,76 con l'LLM** (+0,13): l'unica differenza netta. Riconosce il 92% delle frasi da falco, ma legge una direzione
  in molte frasi che gli esperti considerano neutrali.

Una riga in fondo, in oro: **per l'indice usiamo B**: gira in locale, è veloce e sempre uguale, mentre classificare
19.000 frasi con l'LLM costa troppo. L'LLM resta il riferimento di quanto si può migliorare.

Note per chi parla: il modello scelto (rete neurale, α = 1) è il migliore in cross-validation in tutti e tre gli
approcci ed è anche il migliore sul test. Ripetendo l'addestramento con 10 semi le medie sono A ≈ 0,57, B ≈ 0,62,
B2 ≈ 0,60: le conclusioni non cambiano. Il paper WCB riporta 0,755 con RoBERTa-large (split casuale, più facile) e 0,66
con GPT-4.1 senza addestramento. Il termine di paragone minimo (dire sempre "neutrale") vale 0,23.

---

## Slide 8 – L'indice e la verifica sui tassi

**Messaggio chiave**: il tono dei discorsi si muove prima dei tassi, soprattutto ai punti di svolta.

Grafici: **`grafico_indice_tassi.png`** grande (sinistra o in alto) e **`g_ritardi.png`** più piccolo (destra o in
basso). Se una sola slide è troppo piena, dividere in 8a (indice) e 8b (verifica).

Didascalie:
- **19.038 frasi** dal 2018 classificate: 49% neutrali, 26% falco, 24% colomba; indice da −0,34 (pandemia, 2020) a
  +0,31 (2026).
- **Correlazione +0,40 con la variazione del tasso nel trimestre successivo**: alta quanto quella nello stesso
  trimestre e costruita solo con discorsi fatti prima della decisione. Cala allungando l'orizzonte (0,36 → 0,26).
- **Le svolte**: l'indice sale 2–3 trimestri prima del primo rialzo di luglio 2022 e passa da −0,05 a +0,17 il
  trimestre prima della ripresa dei rialzi di giugno 2026. In entrambi i casi la regola "come l'ultima decisione"
  non prevedeva un rialzo.
- Cautela, in piccolo: i tassi sono molto prevedibili da soli (persistenza 0,76) e la regola banale indovina il 65%
  delle decisioni. L'indice serve dove questa regola sbaglia, cioè alle svolte.

Note per chi parla: lettura delle barre di correlazione. Quella dello stesso trimestre è in parte ingannevole, perché
molti discorsi commentano decisioni già prese; quella a t+1 è la prova che le parole anticipano. Nei tagli del 2024
l'indice resta positivo, perché misura il tono sull'inflazione (ancora sopra l'obiettivo) più che la direzione dei
tassi.

---

## Slide 9 – Dal workflow allo strumento

**Messaggio chiave**: il modello di Orange è stato ricostruito in Python ed è diventato un'app che analizza qualsiasi
discorso in tempo reale.

Contenuto:
- Piccolo schema in tre passi: **Orange** (scelta e validazione del modello) → **Python** (stesso modello in locale,
  stessi embedding: indice identico al 99%) → **App "Falchi & Colombe"** (Streamlit).
- Le quattro pagine dell'app, una riga ciascuna: Panoramica (indice e tassi, insight automatici) · Esplora un
  trimestre (le frasi più da falco e da colomba) · Il modello (cosa sa fare e cosa no) · **Analizza un discorso**
  (testo, link BCE o video YouTube: frasi colorate in tempo reale, punteggio finale e confronto con 420 discorsi dal
  2018).
- 1–2 screenshot dell'app (Panoramica con il quadrante a stelle; pannello in tempo reale dell'analizzatore).

Note per chi parla: subito dopo questa slide c'è la **demo dal vivo**. Discorsi già provati: Schnabel, 5/11/2026
("Navigating stormy waters"): punteggio +0,10, tendenza falco. Conferenza stampa del 29/10/2026 su YouTube
(`8sCMllo0zSg`): +0,38, più da falco del 95% dei discorsi.

---

## Slide 10 – Limiti e sviluppi futuri

**Messaggio chiave**: risultati incoraggianti ma descrittivi; la strada per migliorarli è chiara.

Due colonne, al massimo 3 punti ciascuna.

**Limiti**
- Accuratezza misurata sui resoconti delle riunioni, indice costruito sui discorsi: per i discorsi la verifica è solo
  indiretta.
- Pochi dati: test di 244 frasi (differenze tra A, B, B2 non significative), circa 35 trimestri e 3–4 svolte.
- L'indice misura il contenuto economico delle frasi (descrivere inflazione alta = falco), non l'intenzione
  dell'oratore.

**Sviluppi futuri**
- Far etichettare all'LLM migliaia di frasi dei discorsi e usarle per addestrare il modello locale: la qualità
  dell'LLM al costo della rete.
- Test previsivo per singola riunione, con soglie scelte sul 2018–2024 e verificate sul 2025–2026.
- Fine-tuning di un transformer finanziario (per esempio FOMC-RoBERTa) sugli stessi dati.

Note per chi parla: altri limiti, se chiesti. Dati di addestramento con licenza non commerciale. Embedding e LLM
passano da server esterni (testi pubblici). Possibile contaminazione dell'LLM, perché WCB è pubblico dal 2025. Ci sono
solo i membri del Comitato esecutivo, non i governatori nazionali.

---

## Slide 11 – In sintesi

**Messaggio chiave**: risposta alla domanda della slide 2.

Contenuto: la domanda ripresa in piccolo, poi tre risposte con un numero grande ciascuna:
- **Si può misurare il tono?** Sì: **0,63** di F1 con un modello locale, **0,76** con un LLM.
- **Servono i dati Fed?** Poco: **+0,02** rispetto alle sole frasi BCE. Pochi esempi pertinenti valgono più di molti
  esempi di un'altra banca.
- **Il tono anticipa i tassi?** In parte: correlazione **+0,40** con il trimestre successivo, con segnali in anticipo
  sulle svolte del 2022 e del 2026.

In fondo: "Grazie – domande?" e una riga di fonti: Shah et al. 2023 (Fed), Shah et al. 2025 (WCB), sito BCE, ECB Data
Portal. Progetto universitario non affiliato alla BCE.

---

## Slide di riserva (dopo i ringraziamenti, da mostrare solo se chiesto)

- **R1 – Matrici di confusione**: `g_matrici.png`, se non è entrato nella slide 7. Messaggio: la rete sbaglia un po'
  ovunque; l'LLM quasi mai sulle frasi da falco, ma legge una direzione in molte frasi neutrali.
- **R2 – Scelta del modello**: `g_cv.png`. Messaggio: la rete con α = 1 è la migliore in cross-validation in tutti
  gli approcci ed è anche la migliore sul test.
- **R3 – Indice per oratore**: 2022–23: Lane +0,07 · Elderson +0,16 · Schnabel +0,18 · de Guindos +0,19 ·
  Panetta +0,22 · Lagarde +0,28. 2024–26: Lane +0,06 · de Guindos +0,07 · Schnabel +0,09 · Cipollone +0,10 ·
  Elderson +0,11 · Lagarde +0,17. Cautela: misura il contenuto delle frasi, non la preferenza dell'oratore sui tassi.
