"""Costruisce i workflow Orange del progetto (file .ows nella cartella del progetto).

Uso, con il Python di Orange (non con il python3 di sistema):
    /Applications/Orange.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        codice/costruisci_workflow_orange.py            # crea i due .ows
    ...python3 codice/costruisci_workflow_orange.py --prova   # in più li esegue senza embedding e stampa i risultati

Workflow creati:
  orange_1_valutazione_modelli.ows   A e B, con Bag of Words e con embedding (SBERT), tre modelli, test WCB 2023-24
  orange_2_indice_e_verifica.ows     modello finale, classificazione delle frasi dei discorsi dal 2018, indice per
                                     trimestre / oratore e trimestre / discorso, confronto con i tassi

I widget vengono creati e configurati da Orange stesso (come se li montaste a mano), poi il workflow viene salvato.
I widget che dipendono dai dati (Select Rows, Group By, Merge Data, Correlations, Scatter Plot) vengono configurati
facendo passare i dati: per farlo senza chiamare il server degli embedding, il workflow 2 viene prima eseguito con
Bag of Words e poi la rappresentazione viene sostituita con Document Embedding prima di salvare.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from AnyQt.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

import importlib  # noqa: E402

from orangecanvas import config as cconfig  # noqa: E402
from orangecanvas.registry import WidgetRegistry, cache  # noqa: E402
from orangecanvas.scheme import SchemeNode, signalmanager  # noqa: E402
from orangewidget.utils.filedialogs import RecentPath  # noqa: E402
from Orange.canvas.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROVA = "--prova" in sys.argv

cfg = Config()
cconfig.set_default(cfg)
cconfig.init()
REG = WidgetRegistry()
cfg.widget_discovery(REG, cached_descriptions=cache.registry_cache()).run(cfg.widgets_entry_points())

Q = {
    "corpus": "orangecontrib.text.widgets.owcorpus.OWCorpus",
    "preprocess": "orangecontrib.text.widgets.owpreprocess.OWPreprocess",
    "bow": "orangecontrib.text.widgets.owbagofwords.OWTBagOfWords",
    "emb": "orangecontrib.text.widgets.owdocumentembedding.OWDocumentEmbedding",
    "concat": "Orange.widgets.data.owconcatenate.OWConcatenate",
    "lr": "Orange.widgets.model.owlogisticregression.OWLogisticRegression",
    "nn": "Orange.widgets.model.owneuralnetwork.OWNNLearner",
    "const": "Orange.widgets.model.owconstant.OWConstant",
    "ts": "Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
    "cm": "Orange.widgets.evaluate.owconfusionmatrix.OWConfusionMatrix",
    "pred": "Orange.widgets.evaluate.owpredictions.OWPredictions",
    "select": "Orange.widgets.data.owselectrows.OWSelectRows",
    "script": "Orange.widgets.data.owpythonscript.OWPythonScript",
    "fc": "Orange.widgets.data.owfeatureconstructor.OWFeatureConstructor",
    "gb": "Orange.widgets.data.owgroupby.OWGroupBy",
    "file": "Orange.widgets.data.owfile.OWFile",
    "merge": "Orange.widgets.data.owmergedata.OWMergeData",
    "corr": "Orange.widgets.data.owcorrelations.OWCorrelations",
    "scatter": "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
    "table": "Orange.widgets.data.owtable.OWTable",
    "save": "Orange.widgets.data.owsave.OWSave",
}

STOPWORD_FILE = ROOT / "dati_preparati/stopword_inglesi_senza_negazioni.txt"
# parole che nella lista NLTK sono stopword ma per il tono contano (negazioni e direzione)
DA_TENERE = {"no", "nor", "not", "further", "more", "most", "against", "up", "down", "over", "under", "above",
             "below", "again", "very", "too", "few", "only"}


def classe(qname):
    mod, cls = qname.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def recent(path: Path) -> RecentPath:
    return RecentPath.create(str(path), [("basedir", str(ROOT))])


class Workflow:
    def __init__(self, titolo, descrizione):
        self.m = cfg.workflow_constructor()
        self.m.set_runtime_env("basedir", str(ROOT))
        self.m.title = titolo
        self.m.description = descrizione
        self.sig = self.m.findChild(signalmanager.SignalManager)
        self.sig.pause()

    def nodo(self, chiave, titolo, pos, **props):
        n = SchemeNode(REG.widget(Q[chiave]), title=titolo, position=pos)
        if props:
            props["__version__"] = classe(Q[chiave]).settings_version
            n.properties = props
        self.m.add_node(n)
        return n

    def link(self, a, out, b, inp):
        self.m.new_link(a, out, b, inp)

    def w(self, nodo):
        return self.m.widget_for_node(nodo)

    def esegui(self, timeout=900):
        for n in self.m.nodes:
            self.w(n)
        self.sig.resume()
        t0, calmi = time.time(), 0
        while time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.05)
            occupato = self.sig.has_pending() or any(
                self.sig.is_blocking(n) or self.w(n).isInvalidated() for n in self.m.nodes)
            calmi = 0 if occupato else calmi + 1
            if calmi > 40:
                break
        self.sig.pause()

    def errori(self):
        out = []
        for n in self.m.nodes:
            for msg in n.state_messages():
                if msg.contents and msg.severity in (msg.Error, msg.Warning):
                    out.append(f"{n.title}: {msg.contents}")
        return out

    def salva(self, path: Path):
        with open(path, "wb") as f:
            self.m.save_to(f, pretty=True, pickle_fallback=True)
        print("salvato", path.relative_to(ROOT) if str(path).startswith(str(ROOT)) else path)


# ----------------------------------------------------------------------------- impostazioni comuni
def props_corpus(path):
    # il widget Corpus ricorda il file con il percorso completo (se spostate la cartella: Browse e riselezionate)
    return {"recent_files": [str(ROOT / path)]}


def props_preprocess():
    return {"storedsettings": {"name": "", "preprocessors": [
        ("preprocess.transform", {"methods": [0]}),                      # minuscolo
        ("preprocess.tokenize", {"method": 3, "pattern": r"\w+"}),        # parole
        ("preprocess.filter", {"methods": [0], "language": "(none)",      # stopword da file, senza negazioni
                               "sw_path": recent(STOPWORD_FILE), "sw_list": [recent(STOPWORD_FILE)]}),
    ]}, "autocommit": True}


PROPS_BOW = {"wlocal": "Count", "wglobal": "IDF", "normalization": "(None)", "autocommit": True}


def props_emb(autocommit=True):
    return {"method": 0, "language": "en", "aggregator": "Mean", "autocommit": autocommit}


def scrivi_stopword():
    from nltk.corpus import stopwords
    parole = [w for w in stopwords.words("english") if w not in DA_TENERE]
    STOPWORD_FILE.write_text("\n".join(parole) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- workflow 1
def workflow_valutazione(con_embedding=True):
    wf = Workflow(
        "Progetto BCE – 1. Valutazione dei modelli",
        "Approcci A (solo Fed) e B (Fed + WCB 2015-2022) con due rappresentazioni del testo (Bag of Words ed "
        "embedding SBERT) e tre modelli (regressione logistica, rete neurale, costante). Tutti valutati sulle "
        "stesse 244 frasi WCB 2023-2024 (Test & Score, 'Test on test data'). Leggere l'F1 con la media su tutte "
        "le classi (pesata). Il ramo embedding manda il testo al server di Orange: la prima volta richiede minuti.")
    fed = wf.nodo("corpus", "Fed (2.339 frasi)", (60, 120), **props_corpus("dati_fed/frasi_fed_etichettate.tab"))
    wcb = wf.nodo("corpus", "WCB addestramento 2015-22 (732)", (60, 330),
                  **props_corpus("dati_bce_wcb/frasi_bce_wcb_addestramento.tab"))
    test = wf.nodo("corpus", "WCB TEST 2023-24 (244)", (60, 640), **props_corpus("dati_bce_wcb/frasi_bce_wcb_test.tab"))
    conc = wf.nodo("concat", "Fed + WCB (addestramento B)", (220, 330),
                   merge_type=0, append_source_column=False, auto_commit=True)
    wf.link(fed, "Corpus", conc, "Primary Data")
    wf.link(wcb, "Corpus", conc, "Additional Data")

    lr = wf.nodo("lr", "Logistic Regression", (560, 760), learner_name="Logistic Regression", auto_apply=True)
    nn = wf.nodo("nn", "Neural Network", (660, 760), learner_name="Neural Network", replicable=True, auto_apply=True)
    co = wf.nodo("const", "Constant (classe più frequente)", (760, 760), learner_name="Constant", auto_apply=True)
    learners = [lr, nn, co]

    ts_props = dict(resampling=5, auto_apply=True)   # 5 = Test on test data

    def ramo_bow(sorgente, nome, y):
        p = wf.nodo("preprocess", f"Preprocess Text ({nome})", (380, y), **props_preprocess())
        b = wf.nodo("bow", f"Bag of Words ({nome})", (520, y), **PROPS_BOW)
        ts = wf.nodo("ts", f"Test & Score – {nome}, Bag of Words", (700, y), **ts_props)
        cm = wf.nodo("cm", f"Confusion Matrix – {nome}, BoW", (860, y))
        wf.link(sorgente, "Corpus" if sorgente is fed else "Data", p, "Corpus")
        wf.link(p, "Corpus", b, "Corpus")
        wf.link(b, "Corpus", ts, "Data")
        wf.link(test, "Corpus", ts, "Test Data")   # testo grezzo: Orange ricalcola le parole con il vocabolario
        for le in learners:
            wf.link(le, "Learner", ts, "Learner")
        wf.link(ts, "Evaluation Results", cm, "Evaluation Results")
        return ts

    ts_bow = {"A": ramo_bow(fed, "A", 60), "B": ramo_bow(conc, "B", 200)}
    ts_emb = {}
    if con_embedding:
        e_test = wf.nodo("emb", "Document Embedding (test)", (380, 640), **props_emb())
        wf.link(test, "Corpus", e_test, "Corpus")

        def ramo_emb(sorgente, nome, y):
            e = wf.nodo("emb", f"Document Embedding ({nome})", (450, y), **props_emb())
            ts = wf.nodo("ts", f"Test & Score – {nome}, embedding", (700, y), **ts_props)
            cm = wf.nodo("cm", f"Confusion Matrix – {nome}, embedding", (860, y))
            wf.link(sorgente, "Corpus" if sorgente is fed else "Data", e, "Corpus")
            wf.link(e, "Corpus", ts, "Data")
            wf.link(e_test, "Corpus", ts, "Test Data")
            for le in learners:
                wf.link(le, "Learner", ts, "Learner")
            wf.link(ts, "Evaluation Results", cm, "Evaluation Results")
            return ts
        ts_emb = {"A": ramo_emb(fed, "A", 360), "B": ramo_emb(conc, "B", 500)}
    return wf, ts_bow, ts_emb


# ----------------------------------------------------------------------------- workflow 2
SCRIPT_SOLO_META = '''# Toglie le colonne numeriche della rappresentazione (parole o dimensioni dell'embedding)
# e tiene solo testo, informazioni sulle frasi e previsioni del modello.
# Così Feature Constructor e Group By lavorano su una tabella piccola e uguale
# qualunque rappresentazione usiate.
from Orange.data import Domain
out_data = in_data.transform(Domain([], in_data.domain.class_vars, in_data.domain.metas))
'''

ESPRESSIONE = '1 if modello == "hawkish (falco)" else (-1 if modello == "dovish (colomba)" else 0)'


def workflow_indice(rappresentazione):
    wf = Workflow(
        "Progetto BCE – 2. Indice falco/colomba e verifica sui tassi",
        "Il modello finale (qui: embedding SBERT + regressione logistica, da cambiare se nel workflow 1 vince "
        "un'altra combinazione) è addestrato su Fed + tutte le 976 frasi WCB e classifica le frasi dei discorsi "
        "BCE dal 2018. Punteggio: +1 falco, -1 colomba, 0 neutrale; indice = media del punteggio per trimestre, "
        "per oratore e trimestre, per discorso. L'indice per trimestre è unito ai tassi trimestrali per la "
        "verifica (correlazione con la variazione dei tassi 1-3 trimestri dopo). ATTENZIONE: il Document "
        "Embedding delle frasi dei discorsi (circa 19.000) è lento e non parte da solo: aprite il widget e "
        "premete Apply.")
    fed = wf.nodo("corpus", "Fed (2.339 frasi)", (60, 60), **props_corpus("dati_fed/frasi_fed_etichettate.tab"))
    wcb = wf.nodo("corpus", "WCB tutte (976 frasi)", (60, 200), **props_corpus("dati_bce_wcb/frasi_bce_wcb_etichettate.tab"))
    conc = wf.nodo("concat", "Fed + WCB (modello finale)", (200, 130), merge_type=0, append_source_column=False,
                   auto_commit=True)
    wf.link(fed, "Corpus", conc, "Primary Data")
    wf.link(wcb, "Corpus", conc, "Additional Data")
    lr = wf.nodo("lr", "Logistic Regression (modello)", (500, 130), learner_name="modello", auto_apply=True)

    disc = wf.nodo("corpus", "Frasi dei discorsi BCE (60.764)", (60, 400),
                   **props_corpus("dati_preparati/frasi_bce_politica_monetaria.tab"))
    sel = wf.nodo("select", "Select Rows (dal 2018)", (200, 400), auto_commit=True, purge_attributes=False,
                  purge_classes=False)
    wf.link(disc, "Corpus", sel, "Data")
    pred = wf.nodo("pred", "Predictions", (640, 400))
    wf.link(lr, "Model", pred, "Predictors")

    nodi_rappr = []
    if rappresentazione == "bow":
        p = wf.nodo("preprocess", "Preprocess Text", (330, 130), **props_preprocess())
        b = wf.nodo("bow", "Bag of Words", (420, 130), **PROPS_BOW)
        wf.link(conc, "Data", p, "Corpus")
        wf.link(p, "Corpus", b, "Corpus")
        wf.link(b, "Corpus", lr, "Data")
        wf.link(sel, "Matching Data", pred, "Data")      # testo grezzo: Orange ricalcola le parole
        nodi_rappr = [p, b]
    else:
        aggiungi_embedding(wf, conc, lr, sel, pred)

    script = wf.nodo("script", "Python Script (tieni solo metadati e previsioni)", (780, 400),
                     scriptText=SCRIPT_SOLO_META,
                     scriptLibrary=[{"name": "tieni_solo_meta", "script": SCRIPT_SOLO_META, "filename": None}],
                     currentScriptIndex=0)
    wf.link(pred, "Predictions", script, "Data")
    fc = wf.nodo("fc", "Feature Constructor (punteggio +1/-1/0)", (920, 400))
    wf.link(script, "Data", fc, "Data")
    tab_frasi = wf.nodo("table", "Frasi classificate", (1060, 290))
    wf.link(fc, "Data", tab_frasi, "Data")

    gb_t = wf.nodo("gb", "Group By – indice per trimestre", (1060, 400))
    gb_ot = wf.nodo("gb", "Group By – indice per oratore e trimestre", (1060, 520))
    gb_d = wf.nodo("gb", "Group By – indice per discorso", (1060, 640))
    for g in (gb_t, gb_ot, gb_d):
        wf.link(fc, "Data", g, "Data")
    salvataggi = []
    for g, nome, y in ((gb_ot, "indice_oratore_trimestre.csv", 520), (gb_d, "indice_discorso.csv", 640)):
        t = wf.nodo("table", f"Tabella – {nome[:-4].replace('_', ' ')}", (1200, y))
        s = wf.nodo("save", f"Save Data ({nome})", (1320, y), stored_path="risultati", stored_name=nome,
                    auto_save=False)
        wf.link(g, "Data", t, "Data")
        wf.link(g, "Data", s, "Data")
        salvataggi.append(s)

    tassi = wf.nodo("file", "Tassi BCE per trimestre", (1060, 260),
                    recent_paths=[recent(ROOT / "dati_preparati/tassi_bce_trimestrali.tab")])
    merge = wf.nodo("merge", "Merge Data (indice + tassi)", (1200, 400), merging=0, auto_apply=True)
    wf.link(gb_t, "Data", merge, "Data")
    wf.link(tassi, "Data", merge, "Extra Data")
    corr = wf.nodo("corr", "Correlations (indice vs variazione dei tassi)", (1340, 330))
    sc_t = wf.nodo("scatter", "Scatter Plot – indice nel tempo", (1340, 420))
    sc_v = wf.nodo("scatter", "Scatter Plot – indice vs variazione t+1", (1340, 510))
    tab_m = wf.nodo("table", "Tabella – indice e tassi per trimestre", (1460, 420))
    s_m = wf.nodo("save", "Save Data (indice_trimestre_e_tassi.csv)", (1580, 420), stored_path="risultati",
                  stored_name="indice_trimestre_e_tassi.csv", auto_save=False)
    for n, inp in ((corr, "Data"), (sc_t, "Data"), (sc_v, "Data"), (tab_m, "Data"), (s_m, "Data")):
        wf.link(merge, "Data", n, inp)
    salvataggi.append(s_m)
    nodi = dict(fed=fed, wcb=wcb, conc=conc, lr=lr, disc=disc, sel=sel, pred=pred, script=script, fc=fc,
                gb_t=gb_t, gb_ot=gb_ot, gb_d=gb_d, tassi=tassi, merge=merge, corr=corr, sc_t=sc_t, sc_v=sc_v,
                salvataggi=salvataggi, rappr=nodi_rappr)
    return wf, nodi


def aggiungi_embedding(wf, conc, lr, sel, pred):
    e1 = wf.nodo("emb", "Document Embedding (addestramento)", (360, 130), **props_emb(autocommit=True))
    e2 = wf.nodo("emb", "Document Embedding (frasi discorsi) – premere Apply", (420, 400),
                 **props_emb(autocommit=False))
    wf.link(conc, "Data", e1, "Corpus")
    wf.link(e1, "Corpus", lr, "Data")
    wf.link(sel, "Matching Data", e2, "Corpus")
    wf.link(e2, "Corpus", pred, "Data")


def configura_indice(wf, n):
    """Configura i widget che dipendono dai dati (dopo che i dati sono passati)."""
    from Orange.widgets.data.owfeatureconstructor import ContinuousDescriptor
    from Orange.widgets.data.owselectrows import OWSelectRows
    from Orange.data import ContinuousVariable

    # Select Rows: anno >= 2018
    s = wf.w(n["sel"])
    dom = s.data.domain
    anno = dom["anno"]
    ops = [name for _, name, *_ in OWSelectRows.Operators[ContinuousVariable]]
    s.remove_all_rows() if hasattr(s, "remove_all_rows") else None
    s.add_row(anno, ops.index("is at least"), ["2018"])
    s.conditions_changed()
    s.commit.now() if hasattr(s.commit, "now") else s.commit()
    # Feature Constructor: punteggio
    fc = wf.w(n["fc"])
    fc.setDescriptors([ContinuousDescriptor("punteggio", ESPRESSIONE, 0, False)])
    fc.setCurrentIndex(0)
    fc.apply()


def configura_group_by(wf, n):
    def imposta(g, per, extra):
        w = wf.w(g)
        dom = w.data.domain
        vars_ = list(dom.variables) + list(dom.metas)
        w.gb_attrs = [dom[v] for v in per]
        agg = {v: set() for v in vars_}
        agg[dom["punteggio"]] = {"Mean", "Count"}
        for v in vars_:
            if v.name.startswith("modello (") and ("falco" in v.name or "colomba" in v.name):
                agg[v] = {"Mean"}
        for nome, a in extra.items():
            agg[dom[nome]] = a
        w.aggregations = agg
        w._set_gb_selection()
        w.agg_table_model.set_domain(dom)
        w.result = type(w.result)()      # azzera il raggruppamento calcolato prima
        w.commit.now()
    imposta(n["gb_t"], ["trimestre"], {})
    imposta(n["gb_ot"], ["oratore", "trimestre"], {})
    imposta(n["gb_d"], ["id_discorso"], {"data": {"Min. value"}, "oratore": {"Mode"}, "titolo": {"First value"}})


def configura_verifica(wf, n):
    m = wf.w(n["merge"])
    left = m.data.domain["trimestre"]
    right = m.extra_data.domain["trimestre"]
    m.attr_pairs = [(left, right)]
    m.merging = 0
    if hasattr(m, "attr_boxes"):
        m.attr_boxes.set_state(m.attr_pairs)
    m.commit.now() if hasattr(m.commit, "now") else m.commit()


def configura_grafici(wf, n):
    c = wf.w(n["corr"])
    try:
        c.feature = next(v for v in c.feature_model if v is not None and v.name == "punteggio - Mean")
        c.apply()
    except Exception as e:  # noqa: BLE001
        print("  (Correlations: filtro sulla colonna dell'indice non impostato:", e, ")")
    for nodo, x, y, col in ((n["sc_t"], "anno_decimale", "punteggio - Mean", "tasso_DFR_fine"),
                            (n["sc_v"], "punteggio - Mean", "variazione_t+1", "tasso_DFR_fine")):
        sc = wf.w(nodo)
        modelli = {"attr_x": sc.xy_model, "attr_y": sc.xy_model, "attr_color": sc.gui.color_model}
        for attr, nome in (("attr_x", x), ("attr_y", y), ("attr_color", col)):
            try:
                setattr(sc, attr, next(v for v in modelli[attr] if hasattr(v, "name") and v.name == nome))
            except Exception as e:  # noqa: BLE001
                print(f"  (Scatter Plot {attr}={nome} non impostato: {e})")
        sc.attr_changed()


# ----------------------------------------------------------------------------- main
def risultati_ts(wf, ts_nodes):
    from Orange.evaluation import F1, CA
    out = {}
    for nome, nodo in ts_nodes.items():
        w = wf.w(nodo)
        for slot in w.learners.values():
            res = slot.results
            if res is None or not res.success:
                out[(nome, slot.learner.name)] = f"errore: {getattr(res, 'exception', res)}"
                continue
            r = res.value
            out[(nome, slot.learner.name)] = (round(float(F1(r, average="weighted")[0]), 3),
                                              round(float(CA(r)[0]), 3))
    return out


def main():
    scrivi_stopword()
    (ROOT / "risultati").mkdir(exist_ok=True)

    # --- workflow 1
    wf1, ts_bow, ts_emb = workflow_valutazione(con_embedding=True)
    wf1.salva(ROOT / "orange_1_valutazione_modelli.ows")
    if PROVA:
        wfp, tsb, _ = workflow_valutazione(con_embedding=False)
        wfp.esegui()
        print("PROVA workflow 1 (solo Bag of Words), F1 pesato e accuratezza sul test WCB:")
        for k, v in risultati_ts(wfp, tsb).items():
            print("  ", k, v)
        print("  messaggi:", wfp.errori() or "nessuno")

    # --- workflow 2: configurazione con Bag of Words, poi sostituzione con embedding
    wf2, n = workflow_indice("bow")
    wf2.esegui()
    configura_indice(wf2, n)
    wf2.esegui()
    configura_group_by(wf2, n)
    wf2.esegui()
    configura_verifica(wf2, n)
    wf2.esegui()
    configura_grafici(wf2, n)
    wf2.esegui()
    if PROVA:
        stampa_prova_indice(wf2, n)
    # sostituzione della rappresentazione
    for nodo in n["rappr"]:
        wf2.m.remove_node(nodo)
    for link in list(wf2.m.links):
        if link.source_node is n["sel"] and link.sink_node is n["pred"]:
            wf2.m.remove_link(link)
    aggiungi_embedding(wf2, n["conc"], n["lr"], n["sel"], n["pred"])
    wf2.salva(ROOT / "orange_2_indice_e_verifica.ows")


def stampa_prova_indice(wf, n):
    import numpy as np
    print("PROVA workflow 2 (con Bag of Words):")
    print("  frasi selezionate:", len(wf.w(n["sel"]).data) if wf.w(n["sel"]).data is not None else None,
          "->", (lambda d: None if d is None else len(d))(getattr(wf.w(n["pred"]), "data", None)))
    fc_out = wf.w(n["gb_t"]).data
    print("  colonne dopo Python Script + Feature Constructor:", [v.name for v in fc_out.domain.variables],
          "| meta:", [v.name for v in fc_out.domain.metas][:14])
    p = fc_out.get_column("punteggio")
    print("  punteggio: valori", dict(zip(*np.unique(p, return_counts=True))))
    for nome in ("gb_t", "gb_ot", "gb_d"):
        out = wf.w(n[nome]).result.result_table
        print(f"  {nome}: {None if out is None else (len(out), [v.name for v in out.domain.variables + out.domain.metas])}")
    merged = wf.w(n["merge"]).data
    out = getattr(wf.w(n["corr"]), "data", None)
    print("  merge:", None if out is None else (len(out), [v.name for v in out.domain.variables][:12]))
    print("  messaggi:", wf.errori() or "nessuno")


if __name__ == "__main__":
    main()
