"""Grafico: l'indice falco/colomba dei discorsi BCE anticipa il tasso sui depositi?

Uso (dalla cartella progetto_BCE, con il python3 normale: servono pandas e matplotlib):
    python3 codice/grafico_indice_tassi.py

Legge
    risultati/indice_trimestre_e_tassi.csv   indice per trimestre (rete α=1, Fed + WCB, frasi dei discorsi dal 2018)
    dati_grezzi/tassi_bce.csv                 tasso sui depositi (DFR), giorno per giorno dalle date di entrata in vigore
Scrive
    risultati/grafico_indice_tassi.png / .pdf          sovrapposizione + valori reali
    risultati/grafico_correlazione_ritardi.png / .pdf  correlazione indice -> variazione del tasso, per ritardo

Scelta della scala: indice e tasso hanno unità diverse (indice da -1 a +1, tasso in %). Invece di due assi y
(che fanno sembrare incroci e distanze che dipendono solo da come si scelgono le scale), nel pannello principale
le due serie sono **standardizzate** sullo stesso periodo (2018 - oggi): 0 = valore medio del periodo, 1 = una
deviazione standard. Così si confrontano forma e tempi, su un solo asse. Sotto, due pannelli con i valori reali.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OGGI = pd.Timestamp("2026-11-20")
INIZIO = pd.Timestamp("2018-01-01")
MIN_FRASI = 200            # sotto questa soglia il trimestre è disegnato con il punto vuoto

# colori (palette di riferimento, chiaro): serie 1 = indice, serie 2 = tasso
C_INDICE, C_TASSO = "#2a78d6", "#eb6834"
TESTO, TESTO_2, GRIGLIA, SFONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRIGLIA, "axes.labelcolor": TESTO_2, "xtick.color": TESTO_2,
    "ytick.color": TESTO_2, "axes.titlecolor": TESTO, "figure.facecolor": SFONDO, "axes.facecolor": SFONDO,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIGLIA,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})

# ----------------------------------------------------------------------------- dati
ind = pd.read_csv(BASE / "risultati/indice_trimestre_e_tassi.csv")
anno = ind.trimestre.str[:4].astype(int)
q = ind.trimestre.str[-1].astype(int)
ind["data"] = pd.to_datetime(dict(year=anno, month=3 * (q - 1) + 2, day=15))   # metà trimestre
ind["completo"] = (ind.n_frasi >= MIN_FRASI) & (ind.data + pd.Timedelta(days=45) <= OGGI)

t = pd.read_csv(BASE / "dati_grezzi/tassi_bce.csv", parse_dates=["data_entrata_in_vigore"])
giorni = pd.date_range(INIZIO, OGGI, freq="D")
tasso = (t.set_index("data_entrata_in_vigore").tasso_depositi_DFR
         .reindex(t.data_entrata_in_vigore.tolist() + list(giorni)).sort_index()
         .ffill().loc[lambda s: ~s.index.duplicated(keep="last")].reindex(giorni))

# standardizzazione sullo stesso periodo
base_ind = ind.loc[ind.completo, "indice"]
ind["z"] = (ind.indice - base_ind.mean()) / base_ind.std()
tasso_z = (tasso - tasso.mean()) / tasso.std()

# decisioni che fanno svolta (data di entrata in vigore; la riunione è circa una settimana prima)
SVOLTE = [("2022-07-27", "1° rialzo"), ("2023-09-20", "ultimo rialzo"),
          ("2024-06-12", "1° taglio"), ("2026-06-17", "nuovi rialzi")]
# finestre in cui l'indice si è già mosso e il tasso non ancora
ANTICIPI = [("2021-10-01", "2022-07-27", "indice già da falco,\ntassi ancora fermi"),
            ("2026-01-01", "2026-06-17", "indice risale,\ntassi fermi")]


MESI = "gen feb mar apr mag giu lug ago set ott nov dic".split()


def virgola(v, dec=2):
    return f"{v:.{dec}f}".replace("-", "\u2212").replace(".", ",")


def svolte(ax, etichette=False):
    for d, nome in SVOLTE:
        ax.axvline(pd.Timestamp(d), color=TESTO_2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        if etichette:
            ax.annotate(f"{nome}\n{MESI[pd.Timestamp(d).month - 1]} {pd.Timestamp(d).year}", (pd.Timestamp(d), 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -2), textcoords="offset points", va="top", fontsize=8.5, color=TESTO_2)
    for a, b, _ in ANTICIPI:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=C_INDICE, alpha=0.07, lw=0, zorder=0)


def punti_indice(ax, y):
    ax.plot(ind.data, y, color=C_INDICE, lw=2, zorder=3)
    piena = ind.completo
    ax.scatter(ind.data[piena], y[piena], s=22, color=C_INDICE, edgecolor=SFONDO, linewidth=1.5, zorder=4)
    ax.scatter(ind.data[~piena], y[~piena], s=22, facecolor=SFONDO, edgecolor=C_INDICE, linewidth=1.5, zorder=4)


# ----------------------------------------------------------------------------- figura 1
fig, (ax, ax_i, ax_t) = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3.2, 1.25, 1.25], hspace=0.28))

# pannello principale: le due serie standardizzate su un solo asse
ax.axhline(0, color=TESTO_2, lw=0.8, zorder=1)
svolte(ax, etichette=True)
ax.plot([], [], color=C_INDICE, lw=2, marker="o", ms=5, label="Indice falco/colomba dei discorsi (trimestre)")
ax.step(tasso_z.index, tasso_z, where="post", color=C_TASSO, lw=2, zorder=2, label="Tasso sui depositi BCE (DFR)")
punti_indice(ax, ind.z)
ax.set_ylabel("deviazioni standard dalla media 2018–2026")
ax.set_ylim(-2.3, 2.6)
ax.yaxis.set_major_formatter(lambda v, _: virgola(v, 0))
ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0, 0.86))
for a, b, testo in ANTICIPI:
    ax.annotate(testo, (pd.Timestamp(b), -2.2), xytext=(-5, 0), textcoords="offset points", ha="right",
                va="bottom", fontsize=8, color=C_INDICE)
# etichette dirette a fine linea
ax.annotate("indice", (ind.data.iloc[-2], ind.z.iloc[-2]), xytext=(8, 6), textcoords="offset points",
            color=TESTO, fontsize=9, fontweight="bold")
ax.annotate("tasso", (tasso_z.index[-1], tasso_z.iloc[-1]), xytext=(4, -14), textcoords="offset points",
            color=TESTO, fontsize=9, fontweight="bold")
ax.set_title("Il tono dei discorsi BCE si muove prima dei tassi", loc="left", fontsize=14, fontweight="bold", pad=26)
ax.text(0, 1.025, "Indice falco/colomba (media delle frasi: +1 falco, −1 colomba, 0 neutrale) e tasso sui depositi, "
        "entrambi standardizzati per confrontarne i tempi", transform=ax.transAxes, fontsize=9.5, color=TESTO_2)

# valori reali: indice
ax_i.axhline(0, color=TESTO_2, lw=0.8)
svolte(ax_i)
punti_indice(ax_i, ind.indice)
ax_i.set_ylim(-0.42, 0.42)
ax_i.set_yticks([-0.3, 0, 0.3])
ax_i.yaxis.set_major_formatter(lambda v, _: virgola(v, 1))
ax_i.set_ylabel("indice")
ax_i.text(1.005, 0.92, "falco", transform=ax_i.transAxes, fontsize=8.5, color=TESTO_2, va="top")
ax_i.text(1.005, 0.08, "colomba", transform=ax_i.transAxes, fontsize=8.5, color=TESTO_2, va="bottom")
ax_i.set_title("Valori reali – indice (da −1 tutto colomba a +1 tutto falco)", loc="left", fontsize=9.5, color=TESTO)

# valori reali: tasso
ax_t.axhline(0, color=TESTO_2, lw=0.8)
svolte(ax_t)
ax_t.step(tasso.index, tasso, where="post", color=C_TASSO, lw=2)
ax_t.set_ylim(-1, 4.8)
ax_t.set_yticks([0, 2, 4])
ax_t.yaxis.set_major_formatter(lambda v, _: f"{virgola(v, 0)}%")
ax_t.set_ylabel("tasso")
ax_t.set_title("Valori reali – tasso sui depositi BCE (data di entrata in vigore)", loc="left", fontsize=9.5,
               color=TESTO)
for d, v in [("2019-01-01", -0.5), ("2023-11-15", 4.0), ("2025-09-15", 2.0), ("2026-09-20", 2.5)]:
    ax_t.annotate(f"{virgola(v)}%", (pd.Timestamp(d), v),
                  xytext=(0, 6), textcoords="offset points", fontsize=8, color=TESTO_2, ha="center")

ax_t.xaxis.set_major_locator(mdates.YearLocator())
ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_t.set_xlim(INIZIO - pd.Timedelta(days=20), OGGI + pd.Timedelta(days=60))

fig.text(0.125, 0.025,
         "Fonte: discorsi del Comitato esecutivo BCE (19.038 frasi di politica monetaria dal 2018), classificate con "
         "una rete neurale su embedding SBERT addestrata su frasi Fed e BCE (WCB);\ntassi: ECB Data Portal. "
         f"Punti vuoti = trimestri con meno di {MIN_FRASI} frasi o non ancora conclusi. Le linee tratteggiate sono "
         "le date di entrata in vigore (la decisione è circa una settimana prima).",
         fontsize=8, color=TESTO_2, va="bottom")
fig.subplots_adjust(left=0.08, right=0.94, top=0.9, bottom=0.11)
for ext in ("png", "pdf"):
    fig.savefig(BASE / f"risultati/grafico_indice_tassi.{ext}", dpi=200)

# ----------------------------------------------------------------------------- figura 2: correlazione per ritardo
rit = []
for k, col in enumerate(["variazione_t", "variazione_t+1", "variazione_t+2", "variazione_t+3"]):
    ok = ind[col].notna()
    rit.append((k, np.corrcoef(ind.indice[ok], ind[col][ok])[0, 1], ok.sum()))
fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))
x = [r[0] for r in rit]
ax2.bar(x, [r[1] for r in rit], width=0.55, color=C_INDICE, zorder=2)
for k, r, n in rit:
    ax2.annotate("+" + virgola(r), (k, r), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10,
                 color=TESTO)
ax2.set_xticks(x, ["stesso\ntrimestre", "1 trimestre\ndopo", "2 trimestri\ndopo", "3 trimestri\ndopo"])
ax2.set_ylim(0, 0.5)
ax2.yaxis.set_major_formatter(lambda v, _: virgola(v, 1))
ax2.set_ylabel("correlazione")
ax2.grid(axis="x", visible=False)
ax2.set_title("Indice del trimestre e variazione del tasso…", loc="left", fontsize=12, fontweight="bold")
fig2.text(0.02, 0.01, f"Correlazione di Pearson su {min(r[2] for r in rit)}–{max(r[2] for r in rit)} trimestri (2018–2026), solo quelli con il dato. "
          "Il legame con il trimestre successivo usa solo discorsi\nprecedenti alla decisione. Evidenza descrittiva: "
          "pochi trimestri e pochi punti di svolta.", fontsize=7.5, color=TESTO_2)
fig2.subplots_adjust(bottom=0.27, top=0.88, left=0.12, right=0.97)
for ext in ("png", "pdf"):
    fig2.savefig(BASE / f"risultati/grafico_correlazione_ritardi.{ext}", dpi=200)
print("salvati: risultati/grafico_indice_tassi.(png|pdf), risultati/grafico_correlazione_ritardi.(png|pdf)")
print("correlazioni per ritardo:", [f"{r:+.3f} (n={n})" for _, r, n in rit])
