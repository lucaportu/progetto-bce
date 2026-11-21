"""Falchi & Colombe – il tono dei discorsi BCE. Dashboard e analizzatore di discorsi (README, sezione 8).

Avvio (dalla cartella progetto_BCE):
    .venv/bin/streamlit run app/app.py

Dati usati (creati da codice/modello_locale.py):
    modello/rete_finale.joblib, modello/rete_finale.json
    risultati/frasi_discorsi_classificate.csv, risultati/indice_trimestre_locale.csv, risultati/indice_discorsi.csv
    dati_preparati/tassi_bce_trimestrali.tab, dati_grezzi/tassi_bce.csv
Tema grafico: .streamlit/config.toml (colori e caratteri) + lo stile qui sotto (STILE) + app/assets/ (logo).
Riassunto con l'LLM: chiave in .streamlit/secrets.toml (KIMI_API_KEY = "...").
"""
import html
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(BASE / "codice"))
import analisi_discorso as ad  # noqa: E402

st.set_page_config(page_title="Falchi & Colombe · il tono della BCE", page_icon=str(ASSETS / "emblema.svg"),
                   layout="wide")

# ============================================================================ palette
# Blu della bandiera europea, blu "abisso" per testata e barra laterale, oro delle stelle usato con parsimonia
# (stelle, voce di menu attiva, lancetta). Fondo carta fredda, non avorio. Falco vermiglio, colomba azzurro cielo
# (ben distinto dal blu UE, che resta il colore dell'indice).
BLU_UE, NAVY, ABISSO, ORO, ORO_SCURO = "#003399", "#0b1b3f", "#001233", "#ffcc00", "#a87800"
AVORIO, CARTA, BORDO = "#f3f5fa", "#ffffff", "#dde3ef"
INCHIOSTRO, INCHIOSTRO_2 = "#0b1b3f", "#56627e"
FALCO, COLOMBA, NEUTRALE, NON_RIL = "#d2402f", "#1e95d0", "#8a94aa", "#aab2c3"
C_INDICE, C_TASSO = BLU_UE, ORO_SCURO
COLORE_CLASSE = {"falco": FALCO, "colomba": COLOMBA, "neutrale": NEUTRALE}
F_TITOLI = "'Bricolage Grotesque', 'IBM Plex Sans', sans-serif"
F_TESTO = "'IBM Plex Sans', system-ui, sans-serif"

OGGI = pd.Timestamp.today().normalize()
MIN_FRASI_TRIM, MIN_FRASI_DISCORSO = 200, 10
MESI = "gen feb mar apr mag giu lug ago set ott nov dic".split()


def it(v, dec=2, segno=False):
    s = f"{v:+.{dec}f}" if segno else f"{v:.{dec}f}"
    return s.replace("-", "−").replace(".", ",")


def data_it(d):
    d = pd.Timestamp(d)
    return f"{d.day} {MESI[d.month - 1]} {d.year}"


def e(x):
    return html.escape(str(x))


# ============================================================================ simboli: stelle, quadrante, guilloche
def _stella(cx, cy, R, colore, extra=""):
    pts = []
    for i in range(10):
        r = R if i % 2 == 0 else R * 0.382
        a = -math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{colore}"{extra}/>'


def anello_stelle(dim=220, colore=ORO, con_euro=True):
    c, raggio, R = dim / 2, dim * 0.36, dim * 0.06
    stelle = "".join(_stella(c + raggio * math.cos(2 * math.pi * k / 12 - math.pi / 2),
                             c + raggio * math.sin(2 * math.pi * k / 12 - math.pi / 2), R, colore) for k in range(12))
    euro = (f'<text x="{c}" y="{c + dim * 0.13}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="{dim * 0.36}" font-weight="700" fill="rgba(255,255,255,0.92)">€</text>') if con_euro else ""
    return f'<svg viewBox="0 0 {dim} {dim}" xmlns="http://www.w3.org/2000/svg">{stelle}{euro}</svg>'


def _mescola(c1, c2, t):
    a = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


GAMMA_QUADRANTE = 0.4   # il quadrante va da −0,4 a +0,4: copre tutti i trimestri dal 2018 (minimo −0,34, massimo +0,31)


def quadrante_stelle(valore, animato=True):
    """Le 12 stelle europee disposte a semicerchio, da colomba (sinistra) a falco (destra). La lancetta d'oro
    indica l'indice; la stella più vicina si accende d'oro. Unico momento animato dell'app (all'apertura)."""
    W, H, cx, cy, R = 360, 196, 180, 176, 150
    f = min(1.0, max(0.0, (valore + GAMMA_QUADRANTE) / (2 * GAMMA_QUADRANTE)))
    k_sel = round(f * 11)
    stelle = []
    for i in range(12):
        ang = math.pi - i * math.pi / 11
        x, y = cx + R * math.cos(ang), cy - R * math.sin(ang)
        t = i / 11
        col = _mescola(COLOMBA, "#c3cde0", t * 2) if t <= 0.5 else _mescola("#c3cde0", FALCO, (t - 0.5) * 2)
        if i == k_sel:
            stelle.append(f'<g class="s sel" style="animation-delay:{i * 70}ms">'
                          + _stella(x, y, 15, ORO, ' filter="url(#alone)"') + "</g>")
        else:
            stelle.append(f'<g class="s" style="animation-delay:{i * 70}ms">' + _stella(x, y, 10.5, col) + "</g>")
    # tacche della scala interna
    tacche = []
    for j in range(9):
        a = math.pi - j * math.pi / 8
        r1, r2 = (R - 30, R - 40) if j % 4 == 0 else (R - 32, R - 37)
        tacche.append(f'<line x1="{cx + r1 * math.cos(a):.1f}" y1="{cy - r1 * math.sin(a):.1f}" '
                      f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy - r2 * math.sin(a):.1f}" '
                      'stroke="rgba(255,255,255,0.35)" stroke-width="1.4" stroke-linecap="round"/>')
    gradi = f * 180 - 90
    anim = "" if not animato else f"""
      .s {{opacity: 0; animation: entra .45s cubic-bezier(.2,.8,.2,1) forwards; transform-box: fill-box;
           transform-origin: center;}}
      .sel {{animation: entra .45s cubic-bezier(.2,.8,.2,1) forwards, accendi .6s 1.1s cubic-bezier(.2,1.6,.4,1) both;}}
      .ago {{animation: ruota 1.3s .25s cubic-bezier(.25,1,.35,1) both;}}
      @keyframes entra {{from {{opacity: 0; transform: scale(.4);}} to {{opacity: 1; transform: scale(1);}}}}
      @keyframes accendi {{0% {{transform: scale(1);}} 50% {{transform: scale(1.35);}} 100% {{transform: scale(1);}}}}
      @keyframes ruota {{from {{transform: rotate(-90deg);}} to {{transform: rotate({gradi:.1f}deg);}}}}
      @media (prefers-reduced-motion: reduce) {{ .s, .sel, .ago {{animation: none; opacity: 1;}} }}"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
    <defs>
      <linearGradient id="arco" x1="0" x2="1"><stop offset="0" stop-color="{COLOMBA}"/>
        <stop offset=".5" stop-color="#c3cde0"/><stop offset="1" stop-color="{FALCO}"/></linearGradient>
      <filter id="alone" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="3.2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <style>.ago {{transform-box: view-box; transform-origin: {cx}px {cy}px; transform: rotate({gradi:.1f}deg);}}
      {anim}</style>
    </defs>
    <path d="M {cx - (R - 22)} {cy} A {R - 22} {R - 22} 0 0 1 {cx + (R - 22)} {cy}" fill="none" stroke="url(#arco)"
          stroke-width="3" stroke-linecap="round" opacity=".9"/>
    {"".join(tacche)}
    <g class="ago"><line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - (R - 44)}" stroke="{ORO}" stroke-width="3"
       stroke-linecap="round"/><circle cx="{cx}" cy="{cy - (R - 44)}" r="3.5" fill="{ORO}"/></g>
    <circle cx="{cx}" cy="{cy}" r="9" fill="{ABISSO}" stroke="{ORO}" stroke-width="2.5"/>
    {"".join(stelle)}
    </svg>"""


def guilloche(dim=640):
    """Rosetta a linee sottili, come i motivi di sicurezza delle banconote."""
    c = dim / 2
    linee = []
    for i in range(60):
        a = i * 3
        linee.append(f'<ellipse cx="{c}" cy="{c}" rx="{dim * 0.46:.0f}" ry="{dim * 0.17:.0f}" '
                     f'transform="rotate({a} {c} {c})"/>')
    for i in range(36):
        a = i * 5
        linee.append(f'<ellipse cx="{c}" cy="{c}" rx="{dim * 0.30:.0f}" ry="{dim * 0.09:.0f}" '
                     f'transform="rotate({a} {c} {c})"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dim} {dim}"><g fill="none" '
            f'stroke="rgba(160,190,255,0.10)" stroke-width="0.7">{"".join(linee)}</g></svg>')


def svg_uri(svg):
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


ANELLO_TESTATA = svg_uri(anello_stelle(120, con_euro=False))
ANELLO_PIEDE = svg_uri(anello_stelle(60, colore=ORO_SCURO, con_euro=False))
GUILLOCHE = svg_uri(guilloche())
STELLINA = svg_uri('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' + _stella(12, 13, 11, ORO) + '</svg>')

# ============================================================================ stile
STILE = f"""
<style>
:root {{
  --eu: {BLU_UE}; --abisso: {ABISSO}; --navy: {NAVY}; --oro: {ORO}; --oro-scuro: {ORO_SCURO};
  --carta: {CARTA}; --fondo: {AVORIO}; --bordo: {BORDO}; --ink: {INCHIOSTRO}; --ink-2: {INCHIOSTRO_2};
  --falco: {FALCO}; --colomba: {COLOMBA}; --neutrale: {NEUTRALE};
  --f-titoli: {F_TITOLI}; --f-testo: {F_TESTO};
}}
/* in alto: 3.75rem è la barra di Streamlit (logo, menu e, su Streamlit Cloud, "Fork"), che copre il contenuto */
.block-container {{max-width: 1200px; padding-top: calc(3.75rem + 1.2rem); padding-bottom: 2rem;}}
h1, h2, h3, h4 {{font-family: var(--f-titoli); letter-spacing: -0.02em;}}
.stApp {{font-variant-numeric: tabular-nums;}}
[data-testid="stSidebar"] {{background-image: radial-gradient(120% 60% at 0% 0%, rgba(0,51,153,.55), rgba(0,18,51,0) 60%);}}
[data-testid="stSidebarNav"] a span {{font-weight: 500;}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{background: rgba(255,204,0,0.10); box-shadow: inset 3px 0 0 {ORO};}}
.stTabs [data-baseweb="tab"] {{font-weight: 600;}}
:focus-visible {{outline: 2px solid {ORO} !important; outline-offset: 2px;}}

/* ---------------------------------------------------------------- movimento
   Un solo vocabolario: le celle del registro "si timbrano" (filo d'oro che scorre e svanisce), i numeri emergono
   da sotto, le barre si svelano da sinistra, le frasi entrano dal lato della loro classe (colomba da sinistra,
   falco da destra). La rosetta delle testate ruota lentissima, come una filigrana. Tutto spento con
   prefers-reduced-motion. */
:root {{--ease: cubic-bezier(.2,.8,.2,1); --molla: cubic-bezier(.3,1.35,.5,1);}}
@keyframes sali {{from {{opacity: 0; transform: translateY(12px);}} to {{opacity: 1; transform: none;}}}}
@keyframes emergi {{from {{transform: translateY(110%);}} to {{transform: none;}}}}
@keyframes timbro {{0% {{transform: scaleX(0); opacity: 1;}} 55% {{transform: scaleX(1); opacity: 1;}}
  100% {{transform: scaleX(1); opacity: 0;}}}}
@keyframes svela {{from {{clip-path: inset(0 100% 0 0);}} to {{clip-path: inset(0 0 0 0);}}}}
@keyframes cresci {{from {{transform: scaleX(0);}} to {{transform: scaleX(1);}}}}
@keyframes da-sx {{from {{opacity: 0; transform: translateX(-22px);}} to {{opacity: 1; transform: none;}}}}
@keyframes da-dx {{from {{opacity: 0; transform: translateX(22px);}} to {{opacity: 1; transform: none;}}}}
@keyframes filigrana {{to {{transform: rotate(360deg);}}}}
@keyframes respiro {{0%, 100% {{opacity: .55;}} 50% {{opacity: 1;}}}}

/* testata: blu abisso con rosetta guilloche (che ruota piano) e luce blu UE */
.testata {{position: relative; overflow: hidden; border-radius: 22px; color: #fff; margin: 0 0 18px;
  background: radial-gradient(70% 120% at 85% 40%, rgba(0,51,153,.95) 0%, rgba(0,51,153,0) 70%), {ABISSO};
  box-shadow: 0 1px 0 rgba(255,255,255,.06) inset, 0 18px 40px -18px rgba(0,18,51,.55);}}
.testata::before {{content: ""; position: absolute; width: 620px; height: 620px; right: -140px; top: 50%;
  margin-top: -310px; background: url("{GUILLOCHE}") center / contain no-repeat; pointer-events: none;
  animation: filigrana 240s linear infinite;}}
.testata > * {{position: relative; z-index: 1;}}
.testata h1 {{animation: sali .6s var(--ease) both;}}
.testata p {{animation: sali .6s .12s var(--ease) both;}}
.testata h1 {{font-family: var(--f-titoli); color: #fff; font-weight: 700; margin: 0; padding: 0;
  letter-spacing: -0.03em; line-height: 1.02;}}
.testata p {{color: #c6d2ee; margin: 0; line-height: 1.6;}}
.testata p b {{color: #fff; font-weight: 600;}}
.testata.compatta {{padding: 26px 34px; display: flex; align-items: center; gap: 28px;}}
.testata.compatta h1 {{font-size: 2.1rem; margin-bottom: 8px;}}
.testata.compatta p {{max-width: 680px; font-size: .98rem;}}
.testata.compatta .anello {{flex: none; width: 96px; height: 96px; margin-left: auto; opacity: .95;
  background: url("{ANELLO_TESTATA}") center / contain no-repeat;}}

.testata.grande {{display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px; padding: 38px 40px 30px;}}
.testata.grande h1 {{font-size: clamp(2.1rem, 3.6vw, 3.1rem); margin: 0 0 18px; max-width: 600px; text-wrap: balance;}}
.testata.grande p {{max-width: 520px; font-size: 1.02rem;}}
.testata .nota {{margin-top: 22px; font-size: .8rem; color: #8fa0c8;}}
.quadrante {{display: flex; flex-direction: column; align-items: center; justify-content: flex-end;}}
.quadrante .arco {{width: 100%; max-width: 400px; aspect-ratio: 360 / 196;}}
.quadrante .estremi {{width: 100%; max-width: 400px; display: flex; justify-content: space-between;
  font-size: .78rem; color: #9fb0d6; margin-top: 4px;}}
.quadrante .estremi b {{font-weight: 600;}}
.quadrante .valore {{font-family: var(--f-titoli); font-size: 4rem; font-weight: 700; line-height: 1;
  letter-spacing: -0.04em; margin-top: 4px; color: #fff;}}
.quadrante .didascalia {{font-size: .88rem; color: #c6d2ee; margin-top: 8px; text-align: center;}}
@media (max-width: 900px) {{
  .testata.grande {{grid-template-columns: 1fr; padding: 28px 22px;}}
  .testata.compatta {{padding: 22px;}} .testata.compatta .anello {{display: none;}}
  .testata.compatta h1 {{font-size: 1.7rem;}}
}}

/* titoli di sezione */
.sez {{display: flex; align-items: baseline; gap: 12px; margin: 30px 0 4px;}}
.sez h3 {{font-family: var(--f-titoli); font-size: 1.5rem; font-weight: 700; margin: 0; padding: 0; color: {INCHIOSTRO};
  letter-spacing: -0.02em;}}
.sez .stellina {{width: 12px; height: 12px; flex: none; transform: translateY(-1px);
  background: url("{STELLINA}") center / contain no-repeat;}}
.sez-sub {{color: {INCHIOSTRO_2}; font-size: .92rem; margin: 2px 0 12px; max-width: 760px;}}

/* registro: pannello unico diviso in celle (schede numeriche e insight) */
.kpis {{display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1px; background: {BORDO};
  border: 1px solid {BORDO}; border-radius: 16px; overflow: hidden; margin: 4px 0 8px;}}
.kpi {{background: {CARTA}; padding: 16px 20px 15px; position: relative;
  animation: sali .5s var(--ease) both; animation-delay: calc(var(--i, 0) * 90ms);}}
.kpi::before {{content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: {ORO};
  transform-origin: left; animation: timbro 1.4s var(--ease) both; animation-delay: calc(var(--i, 0) * 90ms + 150ms);}}
.kpi .lbl {{font-size: .82rem; color: {INCHIOSTRO_2}; font-weight: 500;}}
.kpi .val {{font-family: var(--f-titoli); font-size: 2.1rem; font-weight: 700; color: {INCHIOSTRO};
  line-height: 1.05; margin-top: 6px; letter-spacing: -0.03em; overflow: hidden; padding-bottom: .05em;}}
.kpi .val .num {{display: inline-block; animation: emergi .7s var(--ease) both;
  animation-delay: calc(var(--i, 0) * 90ms + 120ms);}}
.kpi .sub {{font-size: .82rem; color: {INCHIOSTRO_2}; margin-top: 7px;}}
.chip {{display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .76rem; font-weight: 600;}}

/* pastiglie di classe */
.pill {{display: inline-block; color: #fff; border-radius: 999px; padding: 1px 10px; font-size: .74rem;
  font-weight: 600; vertical-align: 1px;}}
.legenda {{display: flex; flex-wrap: wrap; gap: 16px; align-items: center; font-size: .85rem; color: {INCHIOSTRO_2};
  margin: 4px 0 12px;}}

/* insight */
.insights {{display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1px; background: {BORDO};
  border: 1px solid {BORDO}; border-radius: 16px; overflow: hidden;}}
.ins {{background: {CARTA}; padding: 16px 20px 16px 52px; position: relative;
  animation: sali .5s var(--ease) both; animation-delay: calc(var(--i, 0) * 70ms + 100ms);}}
.ins::before {{content: ""; position: absolute; left: 20px; top: 19px; width: 18px; height: 18px; border-radius: 6px;
  background: var(--tono, {BLU_UE}); opacity: .14;}}
.ins::after {{content: ""; position: absolute; left: 26px; top: 25px; width: 6px; height: 6px; border-radius: 50%;
  background: var(--tono, {BLU_UE});}}
.ins .t {{font-weight: 600; color: {INCHIOSTRO}; font-size: .95rem;}}
.ins .c {{margin-top: 4px; font-size: .92rem; line-height: 1.55; color: #33405f;}}
.ins .c b {{color: {INCHIOSTRO};}}

/* frasi */
.fr {{background: {CARTA}; border: 1px solid {BORDO}; border-radius: 12px; padding: 12px 16px 11px 18px; margin: 8px 0;
  position: relative; overflow: hidden;}}
.fr::before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c);}}
.fr.sx {{animation: da-sx .55s var(--ease) both; animation-delay: calc(var(--i, 0) * 60ms + 80ms);}}
.fr.dx {{animation: da-dx .55s var(--ease) both; animation-delay: calc(var(--i, 0) * 60ms + 80ms);}}
.fr {{transition: border-color .2s, box-shadow .2s;}}
.fr:hover {{border-color: color-mix(in srgb, var(--c) 45%, {BORDO}); box-shadow: 0 6px 20px -12px rgba(0,18,51,.35);}}
.fr .txt {{font-size: .95rem; line-height: 1.55; color: {INCHIOSTRO};}}
.fr .rank {{font-family: var(--f-titoli); font-weight: 700; color: var(--c); font-size: .9rem; margin-right: 4px;}}
/* misuratore divergente: da −1 (colomba, a sinistra) a +1 (falco, a destra), lo zero al centro */
.metro {{position: relative; width: 132px; height: 6px; border-radius: 999px; background: #e9edf5; flex: none;}}
.metro::after {{content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1.5px; margin-left: -.75px;
  background: #9aa5bd;}}
.metro i {{position: absolute; top: 0; bottom: 0; border-radius: 999px; background: var(--c);
  animation: cresci .8s var(--ease) both; animation-delay: calc(var(--i, 0) * 60ms + 350ms);}}
.metro i.pos {{left: 50%; transform-origin: left;}}
.metro i.neg {{right: 50%; transform-origin: right;}}
.fr .meta {{font-size: .78rem; color: {INCHIOSTRO_2}; margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px 12px;
  align-items: center;}}
.testo-discorso {{background: {CARTA}; border: 1px solid {BORDO}; border-radius: 14px; padding: 16px 18px;
  line-height: 1.9; font-size: 1.0rem; max-height: 540px; overflow-y: auto;}}

/* barra delle classi */
.barra {{display: flex; gap: 3px; border-radius: 10px; overflow: hidden; margin: 12px 0 6px; height: 28px;
  animation: svela 1s .15s var(--ease) both;}}
.cresci {{transform-origin: left; animation: cresci 1s var(--ease) both; animation-delay: calc(var(--i, 0) * 120ms + 200ms);}}
.barra div {{color: #fff; font-size: .8rem; font-weight: 600; display: flex; align-items: center;
  justify-content: center; white-space: nowrap; overflow: hidden;}}

/* contenitori-scheda (st.container con key che inizia con "card") */
[class*="st-key-card"] {{background: {CARTA}; border: 1px solid {BORDO}; border-radius: 16px; padding: 16px 18px;}}
/* selettore del trimestre: frecce sempre affiancate al cursore, anche sul telefono */
.st-key-card_scelta [data-testid="stHorizontalBlock"] {{flex-wrap: nowrap !important; gap: .6rem;}}
.st-key-card_scelta [data-testid="stColumn"] {{min-width: 0 !important;}}
.st-key-card_scelta [data-testid="stColumn"]:not(:nth-child(2)) {{flex: 0 0 48px !important; width: 48px !important;}}
[class*="st-key-live"] {{background: {CARTA}; border: 1px solid {BORDO}; border-radius: 16px; padding: 14px 16px;
  box-shadow: inset 0 3px 0 {ORO};}}

/* verdetto */
.verdetto {{display: grid; grid-template-columns: 1fr 1.4fr; gap: 1px; background: {BORDO}; border: 1px solid {BORDO};
  border-radius: 18px; overflow: hidden; margin: 8px 0 14px;}}
.verdetto > div {{background: {CARTA}; padding: 20px 24px;}}
.verdetto .big {{font-family: var(--f-titoli); font-size: 3.4rem; font-weight: 700; color: {INCHIOSTRO};
  line-height: 1; letter-spacing: -0.04em;}}
.verdetto .lbl {{font-size: .84rem; color: {INCHIOSTRO_2}; font-weight: 500; margin-bottom: 8px;}}
.verdetto .cls {{font-family: var(--f-titoli); font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em;}}
@media (max-width: 760px) {{ .verdetto {{grid-template-columns: 1fr;}} }}
.scala {{position: relative; height: 10px; border-radius: 999px; margin: 16px 0 6px;
  background: linear-gradient(90deg, {COLOMBA} 0%, #c3cde0 50%, {FALCO} 100%);}}
.scala .seg {{position: absolute; top: -7px; width: 24px; height: 24px; margin-left: -12px;
  background: url("{STELLINA}") center / contain no-repeat;
  filter: drop-shadow(0 1px 2px rgba(0,18,51,.45));}}
.scala-lbl {{display: flex; justify-content: space-between; font-size: .76rem; color: {INCHIOSTRO_2};}}
.cambi {{display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1px; background: {BORDO};
  border: 1px solid {BORDO}; border-radius: 14px; overflow: hidden; margin: 6px 0 4px;}}
.cambio {{background: {CARTA}; padding: 12px 16px; font-size: .9rem;}}
.cambio .t {{font-size: .8rem; color: {INCHIOSTRO_2}; font-weight: 500;}}
.cambio .v {{font-weight: 600; font-size: 1.05rem; margin-top: 3px;}}
.vivo {{width: 8px; height: 8px; border-radius: 50%; background: {FALCO}; animation: pulsa 1.6s infinite;}}
@keyframes pulsa {{0% {{box-shadow: 0 0 0 0 {FALCO}66;}} 70% {{box-shadow: 0 0 0 8px {FALCO}00;}} 100% {{box-shadow: 0 0 0 0 {FALCO}00;}}}}
@media (prefers-reduced-motion: reduce) {{ .vivo {{animation: none;}} }}
.riassunto {{background: #eef3ff; border: 1px solid #cfdcf7; border-radius: 14px; padding: 16px 18px; line-height: 1.65;}}

/* piè di pagina */
.piede {{margin-top: 44px; padding-top: 16px; border-top: 1px solid {BORDO}; font-size: .78rem; color: #6b7590;
  display: flex; gap: 14px; align-items: flex-start;}}
.piede .stelle {{width: 34px; height: 34px; flex: none; background: url("{ANELLO_PIEDE}") center / contain no-repeat;}}

/* ---------------------------------------------------------------- esplora un trimestre */
.trim {{display: grid; grid-template-columns: 1.05fr 1fr .95fr; gap: 26px; padding: 28px 32px 26px; align-items: center;}}
.trim .periodo {{font-size: .9rem; color: #9fb0d6;}}
.trim .nome {{font-family: var(--f-titoli); font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 2px;
  animation: sali .5s var(--ease) both;}}
.trim .valore {{font-family: var(--f-titoli); font-size: 4.4rem; font-weight: 700; line-height: 1; letter-spacing: -0.045em;
  margin-top: 10px; overflow: hidden; padding-bottom: .04em;}}
.trim .valore span {{display: inline-block; animation: emergi .75s .1s var(--ease) both;}}
.trim .did {{font-size: .88rem; color: #c6d2ee; margin-top: 10px; line-height: 1.5; animation: sali .5s .3s var(--ease) both;}}
.trim .delta {{display: inline-block; margin-top: 12px; padding: 3px 11px; border-radius: 999px; font-size: .82rem;
  font-weight: 600; background: rgba(255,255,255,.08); animation: sali .5s .4s var(--ease) both;}}
.trim .quadrante .arco {{max-width: 330px;}}
.trim .quadrante .estremi {{max-width: 330px;}}
.fatti {{display: flex; flex-direction: column; gap: 1px; background: rgba(255,255,255,.08); border-radius: 14px;
  overflow: hidden; border: 1px solid rgba(255,255,255,.08);}}
.fatto {{background: rgba(0,18,51,.55); padding: 12px 16px; animation: da-dx .55s var(--ease) both;
  animation-delay: calc(var(--i, 0) * 90ms + 200ms); backdrop-filter: blur(2px);}}
.fatto .t {{font-size: .78rem; color: #9fb0d6;}}
.fatto .v {{font-family: var(--f-titoli); font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 2px;}}
.fatto .s {{font-size: .8rem; color: #c6d2ee; margin-top: 1px;}}
@media (max-width: 900px) {{ .trim {{grid-template-columns: 1fr; padding: 24px 20px;}} .trim .valore {{font-size: 3.4rem;}} }}
.composizione {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 4px;}}
.comp {{font-size: .86rem; color: {INCHIOSTRO_2}; display: flex; align-items: baseline; gap: 8px;
  animation: sali .5s var(--ease) both; animation-delay: calc(var(--i, 0) * 80ms + 500ms);}}
.comp b {{font-family: var(--f-titoli); font-size: 1.35rem; color: var(--c); letter-spacing: -0.02em;}}
.colonna-testa {{display: flex; align-items: center; gap: 10px; margin: 6px 0 2px;}}
.colonna-testa h3 {{font-family: var(--f-titoli); font-size: 1.35rem; font-weight: 700; margin: 0; padding: 0; color: {INCHIOSTRO};}}
.colonna-testa .punto {{width: 12px; height: 12px; border-radius: 4px; background: var(--c); flex: none;}}
.colonna-testa .sub {{margin-left: auto; font-size: .8rem; color: {INCHIOSTRO_2};}}

@media (prefers-reduced-motion: reduce) {{
  .testata::before, .testata h1, .testata p, .kpi, .kpi::before, .kpi .val .num, .ins, .fr.sx, .fr.dx, .metro i,
  .barra, .cresci, .trim .nome, .trim .valore span, .trim .did, .trim .delta, .fatto, .comp
  {{animation: none !important;}}
  .kpi::before {{display: none;}}
}}
</style>
"""


def applica_stile():
    st.html(STILE)
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "emblema.svg"))
    with st.sidebar:
        st.html(f"""<div style="font-size:.8rem;line-height:1.55;color:#b9c6e4;margin-top:10px;padding:12px 14px;
            border:1px solid rgba(255,255,255,.08);border-radius:12px;background:rgba(255,255,255,.03)">
            <div style="color:#fff;font-weight:600;margin-bottom:4px">Come si legge l'indice</div>
            Ogni frase di politica monetaria vale <b style="color:{FALCO_CHIARO}">+1</b> se falco,
            <b style="color:{COLOMBA_CHIARO}">−1</b> se colomba, <b style="color:#fff">0</b> se neutrale.
            L'indice è la media: da −1 (tutto colomba) a +1 (tutto falco).</div>""")


FALCO_CHIARO, COLOMBA_CHIARO = "#ff8a7a", "#7cc8f0"


def banner(titolo, testo):
    st.html(f'<div class="testata compatta"><div><h1>{titolo}</h1><p>{testo}</p></div><div class="anello"></div></div>')


def sezione(titolo, sotto=None):
    st.html(f'<div class="sez"><span class="stellina"></span><h3>{titolo}</h3></div>'
            + (f'<div class="sez-sub">{sotto}</div>' if sotto else ""))


def kpi(lbl, val, sub="", chip=None):
    ch = ""
    if chip:
        testo, colore = chip
        ch = f'<span class="chip" style="background:{colore}1f;color:{colore}">{testo}</span> '
    return (f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val"><span class="num">{val}</span></div>'
            f'<div class="sub">{ch}{sub}</div></div>')


def kpis(*schede):
    """Registro di schede: ognuna entra con un piccolo ritardo (--i) e viene 'timbrata' da un filo d'oro."""
    st.html('<div class="kpis">' + "".join(s.replace('<div class="kpi"', f'<div class="kpi" style="--i:{i}"', 1)
                                           for i, s in enumerate(schede)) + "</div>")


def insights_html(voci, colonne=None):
    stile = f' style="grid-template-columns:{colonne}"' if colonne else ""
    return (f'<div class="insights"{stile}>' + "".join(
        f'<div class="ins" style="--tono:{c};--i:{i}"><div class="t">{t}</div><div class="c">{x}</div></div>'
        for i, (t, c, x) in enumerate(voci)) + "</div>")


def chip_variazione(d, soglia=0.0):
    if d > soglia:
        return (f"▲ {it(d, segno=True)}", FALCO)
    if d < -soglia:
        return (f"▼ {it(d, segno=True)}", COLOMBA)
    return (f"= {it(d, segno=True)}", NEUTRALE)


def pill(classe):
    return f'<span class="pill" style="background:{COLORE_CLASSE.get(classe, NON_RIL)}">{classe}</span>'


def legenda():
    voci = [f'{pill("falco")} verso tassi più alti', f'{pill("colomba")} verso tassi più bassi',
            f'{pill("neutrale")} nessuna direzione',
            f'<span style="color:{NON_RIL};font-weight:600">testo grigio</span> non di politica monetaria, escluso',
            '<span style="font-weight:600">colore più intenso</span> = modello più sicuro']
    st.html('<div class="legenda">' + "".join(f"<span>{v}</span>" for v in voci) + "</div>")


def barra_classi(quote):
    parti = "".join(f'<div style="width:{quote.get(k, 0) * 100:.1f}%;background:{COLORE_CLASSE[k]}">'
                    f'{(k + " ") if quote.get(k, 0) >= 0.18 else ""}{quote.get(k, 0) * 100:.0f}%</div>'
                    for k in ("falco", "neutrale", "colomba")
                    if quote.get(k, 0) > 0)
    return f'<div class="barra">{parti}</div>'


def metro(punteggio_c):
    """Misuratore divergente del punteggio continuo: barra dal centro verso destra (falco) o sinistra (colomba)."""
    lato = "pos" if punteggio_c >= 0 else "neg"
    return (f'<span class="metro" title="punteggio {it(punteggio_c, segno=True)} su una scala da −1 a +1">'
            f'<i class="{lato}" style="width:{abs(punteggio_c) * 50:.1f}%"></i></span>')


def card_frase(testo, classe, punteggio_c, sicurezza, meta="", lato=None, i=0, rango=None):
    """Una frase classificata. lato = "sx"/"dx": entra da sinistra (colomba) o da destra (falco)."""
    cls = f"fr {lato}" if lato else "fr"
    rk = f'<span class="rank">{rango}</span>' if rango else ""
    return (f'<div class="{cls}" style="--c:{COLORE_CLASSE[classe]};--i:{i}"><div class="txt">{rk}{e(testo)}</div>'
            f'<div class="meta">{pill(classe)}{metro(punteggio_c)}<span><b>{it(punteggio_c, segno=True)}</b></span>'
            f'<span>sicurezza {it(sicurezza * 100, 0)}%</span>{("<span>" + meta + "</span>") if meta else ""}</div></div>')


def sfondo(classe, sicurezza):
    c = COLORE_CLASSE[classe].lstrip("#")
    r, g, b = int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)
    a = 0.12 + 0.42 * max(0.0, (sicurezza - 1 / 3) / (2 / 3))   # da 1/3 (incerto) a 1 (sicuro)
    return f"rgba({r},{g},{b},{a:.2f})"


def piede():
    st.html('<div class="piede"><div class="stelle"></div><div>'
            '<b>Falchi &amp; Colombe</b> è un progetto universitario (AIDA, AI and Computer Vision): strumento '
            'indipendente, <b>non affiliato né approvato dalla Banca centrale europea</b>. Fonti: discorsi e tassi '
            'ufficiali della BCE; frasi etichettate della Fed (Shah, Paturi e Chava, 2023) e della BCE (dataset WCB, '
            'Shah et al., 2025), licenze CC BY-NC. Non è un consiglio di investimento.</div></div>')


def stile_grafico(fig, altezza):
    fig.update_layout(height=altezza, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=CARTA, font=dict(family="IBM Plex Sans, sans-serif", color=INCHIOSTRO, size=12),
                      margin=dict(l=8, r=8, t=46, b=48),
                      hoverlabel=dict(bgcolor=ABISSO, bordercolor=ABISSO,
                                      font=dict(family="IBM Plex Sans", color="#fff", size=12)),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, x=0, font=dict(size=12),
                                  entrywidth=230))
    fig.update_xaxes(gridcolor="#edf0f6", linecolor=BORDO, ticks="outside", tickcolor=BORDO,
                     tickfont=dict(color=INCHIOSTRO_2))
    fig.update_yaxes(gridcolor="#edf0f6", zeroline=False, linecolor=BORDO, tickfont=dict(color=INCHIOSTRO_2))
    return fig


# ============================================================================ dati
@st.cache_data
def carica():
    frasi = pd.read_csv(BASE / "risultati/frasi_discorsi_classificate.csv")
    q = pd.read_csv(BASE / "risultati/indice_trimestre_locale.csv")
    tassi_q = pd.read_csv(BASE / "dati_preparati/tassi_bce_trimestrali.tab", sep="\t", skiprows=[1, 2])
    q = q.merge(tassi_q, on="trimestre", how="left")
    anno, n = q.trimestre.str[:4].astype(int), q.trimestre.str[-1].astype(int)
    q["inizio"] = pd.to_datetime(dict(year=anno, month=3 * (n - 1) + 1, day=1))
    q["meta"] = q.inizio + pd.Timedelta(days=45)
    q["fine"] = q.inizio + pd.offsets.QuarterEnd(0)
    q["concluso"] = q.fine <= OGGI
    q["affidabile"] = q.concluso & (q.n_frasi >= MIN_FRASI_TRIM)
    t = pd.read_csv(BASE / "dati_grezzi/tassi_bce.csv", parse_dates=["data_entrata_in_vigore"])
    giorni = pd.date_range("2018-01-01", OGGI, freq="D")
    tasso = t.set_index("data_entrata_in_vigore").tasso_depositi_DFR.sort_index()
    tasso = tasso[~tasso.index.duplicated(keep="last")].reindex(tasso.index.union(giorni)).ffill().reindex(giorni)
    disc = pd.read_csv(BASE / "risultati/indice_discorsi.csv", parse_dates=["data"])
    info = json.loads((BASE / "modello/rete_finale.json").read_text())
    return frasi, q, t, tasso, disc, info


@st.cache_resource(show_spinner="Carico il modello (SBERT + rete neurale)…")
def classificatore():
    return ad.Classificatore()


frasi_all, Q, TASSI, TASSO, DISC, INFO = carica()
DISC_OK = DISC[DISC.n_frasi >= MIN_FRASI_DISCORSO]
_corr = Q[Q["variazione_t+1"].notna()]
INSIGHT_R1 = float(np.corrcoef(_corr.indice, _corr["variazione_t+1"])[0, 1])


# ============================================================================ insight
def insight():
    conclusi = Q[Q.concluso].reset_index(drop=True)
    u, p = conclusi.iloc[-1], conclusi.iloc[-2]
    out = []
    pct = (conclusi.indice < u.indice).mean() * 100
    out.append(("Livello", FALCO if u.indice > 0 else COLOMBA, f"Nel {u.trimestre}, ultimo trimestre concluso, l'indice è <b>{it(u.indice, segno=True)}</b>: "
                f"più da falco del <b>{it(pct, 0)}%</b> dei trimestri dal 2018."
                + (f" Calcolato su sole {u.n_frasi} frasi: va preso con cautela." if u.n_frasi < MIN_FRASI_TRIM else "")))
    d1 = u.indice - p.indice
    out.append(("Cambiamento", FALCO if d1 > 0 else COLOMBA, f"Rispetto al {p.trimestre} il tono è <b>{'salito' if d1 > 0 else 'sceso'} di "
                f"{it(abs(d1))}</b>, cioè {'più da falco' if d1 > 0 else 'più da colomba'}."))
    c = conclusi.copy()
    c["d2"] = c.indice - c.indice.shift(2)
    d2 = c.d2.iloc[-1]
    if pd.notna(d2) and abs(d2) >= 0.05:
        segno = np.sign(d2)
        simili = c.iloc[:-1][(np.sign(c.d2.iloc[:-1]) == segno) & (c.d2.iloc[:-1].abs() >= 0.75 * abs(d2))]
        simili = simili[simili["variazione_t+2"].notna()]
        mossa = "rialzo" if segno > 0 else "taglio"
        segui = ((simili["variazione_t+1"] * segno > 0) | (simili["variazione_t+2"] * segno > 0)).sum()
        if len(simili):
            out.append(("Precedenti", FALCO if segno > 0 else COLOMBA, f"In due trimestri l'indice è {'salito' if segno > 0 else 'sceso'} di "
                        f"{it(abs(d2))}. Dal 2018 movimenti simili ci sono stati {len(simili)} volte: in "
                        f"<b>{segui} casi su {len(simili)}</b> è seguito un {mossa} entro due trimestri. "
                        "Pochi casi: un'indicazione, non una previsione."))
    ultima = TASSI.iloc[-1]
    verso = "rialzo" if ultima.variazione_DFR > 0 else "taglio"
    coerente = (ultima.variazione_DFR > 0) == (u.indice > 0)
    out.append(("Regola banale", BLU_UE, f"L'ultima decisione è stata un <b>{verso}</b> ({it(ultima.variazione_DFR, segno=True)} "
                f"punti, depositi al {it(ultima.tasso_depositi_DFR)}% dal {data_it(ultima.data_entrata_in_vigore)}). "
                f"La regola «la prossima sarà come l'ultima» dice {verso}: il tono dei discorsi è "
                f"<b>{'coerente' if coerente else 'in contrasto'}</b>."))
    corr = Q[Q["variazione_t+1"].notna()]
    r1 = np.corrcoef(corr.indice, corr["variazione_t+1"])[0, 1]
    out.append(("Quanto fidarsi", NEUTRALE, f"Correlazione tra l'indice e la variazione del tasso nel trimestre successivo: "
                f"<b>{it(r1)}</b>. Un legame moderato, utile soprattutto ai punti di svolta."))
    corrente = Q[~Q.concluso]
    if len(corrente):
        k = corrente.iloc[-1]
        out.append((f"Trimestre in corso ({k.trimestre})", ORO_SCURO, f"Indice provvisorio <b>{it(k.indice, segno=True)}</b> su "
                    f"{k.n_frasi} frasi: da confermare a trimestre concluso."))
    return out


# ============================================================================ pagina 1: panoramica
def pagina_panoramica():
    conclusi = Q[Q.concluso]
    u, p = conclusi.iloc[-1], conclusi.iloc[-2]
    pct = (conclusi.indice < u.indice).mean() * 100
    d = u.indice - p.indice
    testo_chip, colore_chip = chip_variazione(d)
    chiaro = {FALCO: FALCO_CHIARO, COLOMBA: COLOMBA_CHIARO}.get(colore_chip, "#c6d2ee")
    st.html(f"""<div class="testata grande">
      <div style="display:flex;flex-direction:column;justify-content:center">
        <h1>Il tono dei discorsi BCE anticipa i tassi?</h1>
        <p>Ogni frase di politica monetaria detta dal Comitato esecutivo è letta da una rete neurale e classificata
        come <b>falco</b> (verso tassi più alti), <b>colomba</b> (verso tassi più bassi) o <b>neutrale</b>.
        La media per trimestre è l'indice, da −1 a +1.</p>
        <div class="nota">{len(frasi_all):,} frasi da {DISC.id_discorso.nunique()} discorsi, dal 2018 a oggi</div>
      </div>
      <div class="quadrante">
        <div class="arco" style="background:url('{svg_uri(quadrante_stelle(u.indice))}') center / contain no-repeat"
             role="img" aria-label="Indice {it(u.indice, segno=True)} su una scala da colomba a falco"></div>
        <div class="estremi"><span><b style="color:{COLOMBA_CHIARO}">colomba</b> −0,4</span>
          <span>+0,4 <b style="color:{FALCO_CHIARO}">falco</b></span></div>
        <div class="valore">{it(u.indice, segno=True)}</div>
        <div class="didascalia">Indice del {u.trimestre}, ultimo trimestre concluso:
          più da falco del {it(pct, 0)}% dei trimestri dal 2018<br>
          <span style="color:{chiaro};font-weight:600">{testo_chip}</span> rispetto al {p.trimestre}</div>
      </div>
    </div>""".replace(f"{len(frasi_all):,}", f"{len(frasi_all):,}".replace(",", ".")))

    ultima = TASSI.iloc[-1]
    schede = [kpi("Tasso sui depositi", f"{it(ultima.tasso_depositi_DFR)}%",
                  f"in vigore dal {data_it(ultima.data_entrata_in_vigore)}",
                  (f"{'▲' if ultima.variazione_DFR > 0 else '▼'} {it(ultima.variazione_DFR, segno=True)} punti",
                   FALCO if ultima.variazione_DFR > 0 else COLOMBA))]
    corrente = Q[~Q.concluso]
    if len(corrente):
        k = corrente.iloc[-1]
        schede.append(kpi(f"Trimestre in corso ({k.trimestre})", it(k.indice, segno=True),
                          f"provvisorio, {k.n_frasi} frasi finora"))
    schede.append(kpi("Correlazione con il trimestre dopo", it(INSIGHT_R1),
                      "indice e variazione del tasso, legame moderato"))
    schede.append(kpi("Discorsi analizzati", f"{DISC.id_discorso.nunique()}",
                      f"{len(frasi_all):,} frasi di politica monetaria".replace(",", ".")))
    kpis(*schede)

    sezione("Indice e tasso sui depositi", "Le fasce colorate sono i cicli di rialzi (rosso) e di tagli (azzurro). "
            "Punti vuoti: trimestri con meno di 200 frasi o non ancora conclusi.")
    with st.container(key="card_grafico"):
        vista = st.segmented_control("Vista", ["Stessa scala", "Valori reali"], default="Stessa scala",
                                     label_visibility="collapsed")
        st.plotly_chart(grafico_indice_tasso(vista != "Valori reali"), width="stretch",
                        config={"displayModeBar": False})
        if vista != "Valori reali":
            st.caption("Le due serie sono standardizzate sul periodo 2018–oggi (0 = media, 1 = una deviazione "
                       "standard): così si confrontano i tempi su un solo asse.")

    sezione("Cosa dicono i dati", "Frasi scritte da regole semplici sui dati, ricalcolate a ogni apertura.")
    st.html(insights_html(insight()))
    piede()


def cicli_tassi():
    """Cicli di rialzi o di tagli dal 2018: sequenze di variazioni dello stesso segno."""
    t = TASSI[TASSI.data_entrata_in_vigore >= "2018-01-01"].sort_values("data_entrata_in_vigore")
    cicli, cur = [], None
    for r in t.itertuples():
        segno = np.sign(r.variazione_DFR)
        if cur and cur[0] == segno:
            cur[2] = r.data_entrata_in_vigore
        else:
            cur = [segno, r.data_entrata_in_vigore, r.data_entrata_in_vigore]
            cicli.append(cur)
    return [(s, a, max(b, a + pd.Timedelta(days=40))) for s, a, b in cicli]


def grafico_indice_tasso(standardizzato=True):
    base = Q[Q.affidabile].indice
    if standardizzato:
        fig = make_subplots(rows=1, cols=1)
        yi = (Q.indice - base.mean()) / base.std()
        yt = (TASSO - TASSO.mean()) / TASSO.std()
        righe = (1, 1)
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                            subplot_titles=("Indice falco/colomba (−1 colomba, +1 falco)", "Tasso sui depositi (%)"))
        yi, yt = Q.indice, TASSO
        righe = (1, 2)
    for segno, a, b in cicli_tassi():
        rialzo = segno > 0
        fig.add_vrect(x0=a, x1=b, fillcolor=FALCO if rialzo else COLOMBA, opacity=0.08, line_width=0, layer="below")
        fig.add_annotation(x=a + (b - a) / 2, y=1.0 if standardizzato else 1.07, xref="x", yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{'rialzi' if rialzo else 'tagli'} {a.year}" + (f"–{str(b.year)[2:]}" if b.year != a.year else ""),
                           font=dict(size=11, color=FALCO if rialzo else COLOMBA, family="IBM Plex Sans"))
    fig.add_trace(go.Scatter(x=TASSO.index, y=yt, mode="lines", line=dict(color=C_TASSO, width=2.5, shape="hv"),
                             name="Tasso sui depositi BCE", customdata=TASSO.values,
                             hovertemplate="%{x|%d/%m/%Y}<br><b>tasso %{customdata:.2f}%</b><extra></extra>"),
                  row=righe[1], col=1)
    pieno = Q.affidabile
    fig.add_trace(go.Scatter(x=Q.meta, y=yi, mode="lines+markers", name="Indice falco/colomba",
                             line=dict(color=C_INDICE, width=2.8),
                             marker=dict(size=8, color=np.where(pieno, C_INDICE, CARTA), line=dict(color=C_INDICE, width=2)),
                             customdata=np.c_[Q.trimestre, Q.indice, Q.n_frasi],
                             hovertemplate="<b>%{customdata[0]}</b><br>indice %{customdata[1]:+.3f}<br>"
                                           "%{customdata[2]} frasi<extra></extra>"),
                  row=righe[0], col=1)
    # etichette dirette a fine linea, al posto della legenda
    fig.add_annotation(x=Q.meta.iloc[-1], y=yi.iloc[-1], text="<b>indice</b>", showarrow=False, xanchor="left",
                       xshift=10, font=dict(color=C_INDICE, size=12), row=righe[0], col=1)
    fig.add_annotation(x=TASSO.index[-1], y=yt.iloc[-1], text="<b>tasso</b>", showarrow=False, xanchor="left",
                       xshift=8, font=dict(color=C_TASSO, size=12), row=righe[1], col=1)
    fig.add_hline(y=0, line=dict(color="#b6bfd2", width=1), row=1, col=1)
    stile_grafico(fig, 500 if standardizzato else 600)
    fig.update_layout(showlegend=False, margin=dict(r=64, t=34 if standardizzato else 70, b=30),
                      transition=dict(duration=600, easing="cubic-in-out"))
    if standardizzato:
        fig.update_yaxes(title_text="deviazioni standard dalla media", title_font=dict(size=11, color=INCHIOSTRO_2))
    else:
        fig.update_annotations(selector=dict(text="Tasso sui depositi (%)"), font=dict(size=13, color=INCHIOSTRO))
    return fig


# ============================================================================ pagina 2: esplora un trimestre
PERIODI = {1: "gennaio – marzo", 2: "aprile – giugno", 3: "luglio – settembre", 4: "ottobre – dicembre"}


def _a_capo(testo, larghezza=64, massimo=260):
    import textwrap
    t = testo if len(testo) <= massimo else testo[:massimo].rsplit(" ", 1)[0] + "…"
    return "<br>".join(textwrap.wrap(html.escape(t), larghezza))


def _colore_indice(v, forza=0.32):
    """Colore di un valore dell'indice: dal grigio neutro verso falco (positivo) o colomba (negativo)."""
    t = min(1.0, abs(v) / forza)
    return _mescola("#c3cde0", FALCO if v >= 0 else COLOMBA, 0.25 + 0.75 * t)


def grafico_trimestri(sel):
    """Una barra per trimestre (cliccabile) con il tasso sui depositi a gradini: è il selettore della pagina."""
    colori = [_colore_indice(v) for v in Q.indice]
    bordo = [ORO_SCURO if t == sel else "rgba(0,0,0,0)" for t in Q.trimestre]
    larg = [3 if t == sel else 0 for t in Q.trimestre]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=Q.trimestre, y=Q.indice, marker=dict(color=colori, line=dict(color=bordo, width=larg)),
                         opacity=1, selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
                         customdata=np.c_[Q.n_frasi], name="indice",
                         hovertemplate="<b>%{x}</b><br>indice %{y:+.2f}<br>%{customdata[0]} frasi"
                                       "<br><i>clic per aprirlo</i><extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=Q.trimestre, y=Q.tasso_DFR_fine, mode="lines", line=dict(color=C_TASSO, width=2,
                             shape="hv"), name="tasso", hovertemplate="tasso a fine trimestre %{y:.2f}%<extra></extra>"),
                  secondary_y=True)
    r = Q.set_index("trimestre").loc[sel]
    fig.add_annotation(x=sel, y=r.indice, yshift=14 if r.indice >= 0 else -14, showarrow=False,
                       text=f"<b>{it(r.indice, segno=True)}</b>", font=dict(color=ORO_SCURO, size=12),
                       yanchor="bottom" if r.indice >= 0 else "top")
    anni = [t for t in Q.trimestre if t.endswith("T1")]
    stile_grafico(fig, 250)
    fig.update_layout(showlegend=False, margin=dict(l=6, r=6, t=14, b=6), bargap=0.18, dragmode=False,
                      clickmode="event+select", transition=dict(duration=450, easing="cubic-in-out"))
    fig.update_xaxes(tickvals=anni, ticktext=[t[:4] for t in anni], ticks="")
    fig.update_yaxes(range=[-0.42, 0.42], tickvals=[-0.3, 0, 0.3], ticktext=["−0,3", "0", "+0,3"],
                     zeroline=True, zerolinecolor="#b6bfd2", secondary_y=False)
    fig.update_yaxes(range=[-1.2, 5], showgrid=False, ticksuffix="%", tickvals=[0, 2, 4],
                     tickfont=dict(color=C_TASSO), secondary_y=True)
    return fig


def grafico_sciame(f, indice):
    """Tutte le frasi del trimestre su una riga da −1 (colomba) a +1 (falco): uno 'sciame' allargato dove le frasi
    sono più fitte. Passando il mouse si legge la frase."""
    x = f.punteggio_continuo.to_numpy()
    dens, bordi = np.histogram(x, bins=40, range=(-1, 1))
    k = np.clip(np.digitize(x, bordi) - 1, 0, 39)
    larg = 0.12 + 0.88 * dens[k] / max(dens.max(), 1)
    rng = np.random.default_rng(7)
    y = rng.uniform(-1, 1, len(x)) * larg
    fig = go.Figure()
    for cl in ("colomba", "neutrale", "falco"):
        m = (f.classe == cl).to_numpy()
        if not m.any():
            continue
        g = f[m]
        fig.add_trace(go.Scatter(
            x=x[m], y=y[m], mode="markers", name=cl,
            marker=dict(size=7, color=COLORE_CLASSE[cl], opacity=(0.3 + 0.65 * (g.sicurezza - 1 / 3) / (2 / 3)).clip(0.3, 0.95),
                        line=dict(width=0)),
            customdata=np.c_[[_a_capo(t) for t in g.frase], g.oratore.fillna(""),
                             [data_it(d) for d in g.data], [it(v, segno=True) for v in g.punteggio_continuo]],
            hovertemplate="%{customdata[0]}<br><br><b>%{customdata[1]}</b>, %{customdata[2]}<br>"
                          f"{cl}, punteggio " + "%{customdata[3]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#b6bfd2", width=1))
    fig.add_vline(x=indice, line=dict(color=ORO_SCURO, width=2, dash="dot"))
    fig.add_annotation(x=indice, y=1.12, yref="y", text=f"indice {it(indice, segno=True)}", showarrow=False,
                       font=dict(color=ORO_SCURO, size=11), yanchor="bottom")
    for xx, testo, anc in ((-1, "← più colomba", "left"), (1, "più falco →", "right")):
        fig.add_annotation(x=xx, y=-1.2, text=testo, showarrow=False, xanchor=anc, yanchor="top",
                           font=dict(color=COLOMBA if xx < 0 else FALCO, size=12))
    stile_grafico(fig, 300)
    fig.update_layout(showlegend=False, margin=dict(l=6, r=6, t=26, b=28), dragmode=False,
                      hoverlabel=dict(align="left"))
    fig.update_xaxes(range=[-1.04, 1.04], tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["−1", "−0,5", "0", "+0,5", "+1"],
                     showgrid=False)
    fig.update_yaxes(visible=False, range=[-1.45, 1.35])
    return fig


def grafico_discorsi(dd):
    """Discorsi del trimestre come barre divergenti, dal più da falco (in alto) al più da colomba."""
    d = dd.sort_values("indice").copy()
    cognome = d.oratore.fillna("").str.split().str[-1]
    d["etichetta"] = cognome + ", " + d.data.dt.strftime("%d/%m")
    d["etichetta"] = d.etichetta + [" " * k for k in d.groupby("etichetta").cumcount()]   # etichette uniche
    fig = go.Figure(go.Bar(
        x=d.indice, y=d.etichetta, orientation="h", marker=dict(color=[_colore_indice(v, 0.5) for v in d.indice]),
        text=[f"{it(v, segno=True)}  ({n} frasi)" for v, n in zip(d.indice, d.frasi)], textposition="outside",
        textfont=dict(size=11, color=INCHIOSTRO_2), cliponaxis=False,
        customdata=np.c_[[_a_capo(t, 56, 140) for t in d.titolo], d.oratore, [data_it(x) for x in d.data],
                         d.falco, d.colomba, d.frasi],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}, %{customdata[2]}<br>"
                      "%{customdata[3]} falco, %{customdata[4]} colomba su %{customdata[5]} frasi<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#9aa5bd", width=1.2))
    stile_grafico(fig, 70 + 26 * len(d))
    fig.update_layout(margin=dict(l=6, r=10, t=10, b=30), bargap=0.28, dragmode=False,
                      transition=dict(duration=450, easing="cubic-in-out"))
    fig.update_xaxes(range=[-1.25, 1.25], tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["−1", "−0,5", "0", "+0,5", "+1"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=INCHIOSTRO))
    return fig


def _sposta_trimestre(passo):
    tr = Q.trimestre.tolist()
    i = tr.index(st.session_state.tq)
    st.session_state.tq = tr[min(max(i + passo, 0), len(tr) - 1)]


def _scegli_da_grafico():
    ev = st.session_state.get("grafico_trim")
    try:
        punti = ev["selection"]["points"]
    except (KeyError, TypeError):
        return
    if punti and punti[0].get("x") in set(Q.trimestre):
        st.session_state.tq = punti[0]["x"]


def pagina_esplora():
    banner("Dentro un trimestre",
           "Quali frasi hanno spinto l'indice verso l'alto o verso il basso, e in quali discorsi sono state dette.")
    trimestri = Q.trimestre.tolist()
    if st.session_state.get("tq") not in trimestri:
        st.session_state.tq = Q[Q.concluso].trimestre.iloc[-1]

    with st.container(key="card_scelta"):
        st.html(f'<div style="font-size:.9rem;color:{INCHIOSTRO_2};margin-bottom:-6px">Cliccate una barra, '
                'trascinate il cursore o usate le frecce. Barre: indice di ogni trimestre; linea d\'oro: tasso sui '
                'depositi a fine trimestre.</div>')
        st.plotly_chart(grafico_trimestri(st.session_state.tq), width="stretch", key="grafico_trim",
                        on_select=_scegli_da_grafico, selection_mode="points", config={"displayModeBar": False})
        a, b, c = st.columns([1, 10, 1], vertical_alignment="center")
        i = trimestri.index(st.session_state.tq)
        a.button("", icon=":material/chevron_left:", on_click=_sposta_trimestre, args=(-1,), disabled=i == 0,
                 help="Trimestre precedente", key="prec", width="stretch")
        b.select_slider("Trimestre", trimestri, key="tq", label_visibility="collapsed")
        c.button("", icon=":material/chevron_right:", on_click=_sposta_trimestre, args=(1,),
                 disabled=i == len(trimestri) - 1, help="Trimestre successivo", key="succ", width="stretch")

    tq = st.session_state.tq
    r = Q.set_index("trimestre").loc[tq]
    i = trimestri.index(tq)
    f = frasi_all[frasi_all.trimestre == tq]
    conteggi = f.classe.value_counts()
    quote = (conteggi / max(len(f), 1)).to_dict()

    # --- testata del trimestre: indice, quadrante, fatti
    pct = (Q[Q.concluso].indice < r.indice).mean() * 100
    if i > 0:
        prec = Q.iloc[i - 1]
        testo_chip, colore_chip = chip_variazione(r.indice - prec.indice)
        chiaro = {FALCO: FALCO_CHIARO, COLOMBA: COLOMBA_CHIARO}.get(colore_chip, "#c6d2ee")
        delta = f'<span class="delta" style="color:{chiaro}">{testo_chip} rispetto al {prec.trimestre}</span>'
    else:
        delta = ""
    anno, n = tq.split("-T")
    stato = "" if r.concluso else " (in corso)"
    var_t = r.variazione_t if pd.notna(r.variazione_t) else None
    if pd.notna(r["decisione_t+1"]) and pd.notna(r["variazione_t+1"]):
        v1 = r["variazione_t+1"]
        simbolo, col = ("▲", FALCO_CHIARO) if v1 > 0 else (("▼", COLOMBA_CHIARO) if v1 < 0 else ("=", "#c6d2ee"))
        dec_v = f'<span style="color:{col}">{simbolo} {r["decisione_t+1"]}</span>'
        dec_s = f"{it(v1, segno=True)} punti sul tasso sui depositi" if v1 else "tasso invariato"
    else:
        dec_v, dec_s = "non ancora nota", "il trimestre successivo non è concluso"
    fatti = [("Frasi di politica monetaria", f"{len(f)}", f"da {f.id_discorso.nunique()} discorsi"),
             ("Tasso sui depositi a fine trimestre",
              f"{it(r.tasso_DFR_fine)}%" if pd.notna(r.tasso_DFR_fine) else "–",
              f"variazione nel trimestre {it(var_t, segno=True)}" if var_t is not None else "trimestre in corso"),
             ("Decisione nel trimestre dopo", dec_v, dec_s)]
    st.html(f"""<div class="testata trim">
      <div>
        <div class="periodo">{PERIODI[int(n)]} {anno}{stato}</div>
        <div class="nome">{tq}</div>
        <div class="valore"><span>{it(r.indice, segno=True)}</span></div>
        <div class="did">Indice del trimestre: più da falco del {it(pct, 0)}% dei trimestri conclusi dal 2018.</div>
        {delta}
      </div>
      <div class="quadrante">
        <div class="arco" style="background:url('{svg_uri(quadrante_stelle(r.indice))}') center / contain no-repeat"
             role="img" aria-label="Indice {it(r.indice, segno=True)} su una scala da colomba a falco"></div>
        <div class="estremi"><span><b style="color:{COLOMBA_CHIARO}">colomba</b> −0,4</span>
          <span>+0,4 <b style="color:{FALCO_CHIARO}">falco</b></span></div>
      </div>
      <div class="fatti">{"".join(f'<div class="fatto" style="--i:{k}"><div class="t">{t}</div><div class="v">{v}</div>'
                                  f'<div class="s">{s}</div></div>' for k, (t, v, s) in enumerate(fatti))}</div>
    </div>""")
    if r.n_frasi < MIN_FRASI_TRIM or not r.concluso:
        st.warning("Trimestre con poche frasi o non ancora concluso: l'indice è meno affidabile.", icon=":material/info:")

    # --- composizione e sciame di tutte le frasi
    sezione("Tutte le frasi del trimestre", "Ogni punto è una frase, da quella più da colomba (a sinistra) a quella "
            "più da falco (a destra). I punti più pieni sono quelli di cui il modello è più sicuro. Passate il mouse "
            "per leggerla.")
    with st.container(key="card_sciame"):
        st.html(barra_classi(quote) + '<div class="composizione">' + "".join(
            f'<div class="comp" style="--c:{COLORE_CLASSE[k]};--i:{j}"><b>{int(conteggi.get(k, 0))}</b>'
            f'frasi {nome}</div>' for j, (k, nome) in enumerate(
                [("falco", "da falco"), ("neutrale", "neutrali"), ("colomba", "da colomba")])) + "</div>")
        st.plotly_chart(grafico_sciame(f, float(r.indice)), width="stretch",
                        config={"displayModeBar": False}, key=f"sciame_{tq}")

    # --- le frasi agli estremi
    sezione("Le frasi agli estremi", "Le frasi di cui il modello è più convinto in un senso o nell'altro, ordinate "
            "per punteggio: P(falco) − P(colomba).")
    with st.expander("Come si decide che una frase è più da falco di un'altra", icon=":material/help:"):
        st.markdown(
            "Per ogni frase la rete neurale dà tre probabilità che sommano a 1: falco, colomba, neutrale.\n\n"
            "- **Classe**: la più probabile. È quella che entra nell'indice (+1, −1, 0).\n"
            "- **Punteggio**: P(falco) − P(colomba), da −1 a +1, disegnato dalla barretta: verso destra falco, "
            "verso sinistra colomba. Più è lunga, più il modello è convinto.\n"
            "- **Sicurezza**: la probabilità più alta.\n\n"
            "Il punteggio misura la convinzione del modello, non l'intensità economica: un rialzo di 75 punti base "
            "non vale automaticamente più di uno di 25. Le probabilità servono a ordinare, non sono percentuali "
            "esatte.")
    quante = st.segmented_control("Frasi per lato", [5, 10, 20], default=5, key="quante",
                                  format_func=lambda k: f"{k} per lato") or 5
    s, t = st.columns(2, gap="large")
    with s:
        st.html(f'<div class="colonna-testa" style="--c:{COLOMBA}"><span class="punto"></span>'
                f'<h3>Più da colomba</h3><span class="sub">verso tassi più bassi</span></div>')
        st.html("".join(card_frase(x.frase, x.classe, x.punteggio_continuo, x.sicurezza,
                                   f"{e(x.oratore)}, {data_it(x.data)}", lato="sx", i=k, rango=k + 1)
                        for k, x in enumerate(f.nsmallest(quante, "punteggio_continuo").itertuples())))
    with t:
        st.html(f'<div class="colonna-testa" style="--c:{FALCO}"><span class="punto"></span>'
                f'<h3>Più da falco</h3><span class="sub">verso tassi più alti</span></div>')
        st.html("".join(card_frase(x.frase, x.classe, x.punteggio_continuo, x.sicurezza,
                                   f"{e(x.oratore)}, {data_it(x.data)}", lato="dx", i=k, rango=k + 1)
                        for k, x in enumerate(f.nlargest(quante, "punteggio_continuo").itertuples())))

    # --- i discorsi
    dd = (f.groupby(["id_discorso", "data", "oratore", "titolo"])
          .agg(indice=("punteggio", "mean"), frasi=("punteggio", "size"),
               falco=("classe", lambda s: (s == "falco").sum()), colomba=("classe", lambda s: (s == "colomba").sum()))
          .reset_index().sort_values("indice", ascending=False))
    dd["data"] = pd.to_datetime(dd.data)
    pochi = (dd.frasi < 5).sum()
    dd = dd[dd.frasi >= 5]
    sezione("I discorsi del trimestre", f"Dal più da falco (in alto) al più da colomba. Esclusi {pochi} discorsi con "
            "meno di 5 frasi di politica monetaria; sotto le 10 frasi l'indice di un discorso è poco affidabile.")
    if len(dd):
        with st.container(key="card_discorsi"):
            st.plotly_chart(grafico_discorsi(dd), width="stretch", config={"displayModeBar": False},
                            key=f"discorsi_{tq}")
        with st.expander("Tabella dei discorsi", icon=":material/table:"):
            st.dataframe(dd.drop(columns="id_discorso"), hide_index=True, width="stretch",
                         column_config={"indice": st.column_config.ProgressColumn("indice", format="%+.2f",
                                                                                  min_value=-1, max_value=1),
                                        "data": st.column_config.DateColumn("data", format="DD/MM/YYYY"),
                                        "titolo": st.column_config.TextColumn("titolo", width="large")})
    else:
        st.caption("Nessun discorso con almeno 5 frasi di politica monetaria in questo trimestre.")
    piede()


# ============================================================================ pagina 3: il modello
def pagina_modello():
    banner("Come vengono lette le frasi",
           "Una rete neurale applicata agli embedding SBERT della frase: 384 numeri che ne riassumono il "
           "significato. È il modello vincente del confronto in Orange, ricostruito in Python.")
    v = INFO["verifica_test_F1_pesato"]["B"]
    kpis(kpi("F1 pesato sul test", it(v["seme_1"]), "244 frasi BCE 2023–24 mai viste"),
         kpi("Accuratezza", f"{it(v['accuratezza_seme_1'] * 100, 0)}%", "circa 6 frasi su 10"),
         kpi("Kappa con gli esperti", it(v["kappa_seme_1"]), "accordo al netto del caso"),
         kpi("Variabilità", f"{it(v['media_10_semi'])} ± {it(v['dev_st_10_semi'])}", "F1 su 10 addestramenti"))
    s, t = st.columns([1.1, 1], gap="large")
    with s:
        sezione("Matrice di confusione")
        nomi = ["colomba", "falco", "neutrale"]
        m = np.array(v["matrice_seme_1"])
        fig = go.Figure(go.Heatmap(z=m, x=nomi, y=nomi, colorscale=[[0, "#f3f6fc"], [1, BLU_UE]], showscale=False,
                                   text=m, texttemplate="<b>%{text}</b>", textfont=dict(size=17),
                                   hovertemplate="esperti: %{y}<br>modello: %{x}<br>%{z} frasi<extra></extra>"))
        stile_grafico(fig, 360)
        fig.update_layout(margin=dict(t=10), xaxis_title="classe data dal modello", yaxis_title="classe degli esperti",
                          yaxis=dict(autorange="reversed"))
        with st.container(key="card_matrice"):
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Addestrato su Fed + frasi BCE 2015–2022 e provato sulle 244 frasi del 2023–2024. Il modello "
                   "usato nell'app è poi riaddestrato su tutte le frasi etichettate (2.339 Fed + 976 BCE).")
    with t:
        sezione("F1 per classe")
        st.html("".join(f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0">{pill(n)}'
                        f'<div style="flex:1;height:10px;background:#e9edf5;border-radius:6px">'
                        f'<div class="cresci" style="--i:{i};width:{val * 100:.0f}%;height:10px;'
                        f'background:{COLORE_CLASSE[n]};border-radius:6px"></div></div><b>{it(val)}</b></div>'
                        for i, (n, val) in enumerate(zip(["colomba", "falco", "neutrale"], v["f1_per_classe_seme_1"]))))
        sezione("Cosa sa fare e cosa no")
        st.html(insights_html([(t, BLU_UE, c) for t, c in [
                ("Meglio in media che sulla singola frase", "Gli errori si compensano in parte sulle centinaia di "
                 "frasi di un trimestre: per questo l'indice è più affidabile della classe di una frase."),
                ("Addestrato su testi scritti", "Ha imparato dai resoconti delle riunioni: sui discorsi, e ancora "
                 "di più su un discorso parlato e trascritto, può essere meno preciso."),
                ("Un LLM fa meglio, ma costa", "Con la guida degli annotatori un LLM arriva a 0,76 sullo stesso test, "
                 "ma usarlo su tutte le frasi costerebbe troppo."),
                ("Non è un segnale di trading", "Il legame con i tassi è moderato: correlazione di circa 0,4.")]],
            colonne="1fr"))
    piede()


# ============================================================================ pagina 4: analizza un discorso
def classifica_frasi(frasi):
    clf = classificatore()
    pm = [f for f in frasi if f.politica_monetaria]
    P = clf.probabilita([f.testo for f in pm])
    out, k = [], 0
    for f in frasi:
        r = {"testo": f.testo, "politica_monetaria": f.politica_monetaria, "inizio": f.inizio}
        if f.politica_monetaria:
            r.update(ad.descrivi(P[k]))
            k += 1
        out.append(r)
    return out


def classe_relativa(pct):
    for soglia, nome in [(20, "colomba"), (40, "tendenza colomba"), (60, "nella media"), (80, "tendenza falco")]:
        if pct < soglia:
            return nome
    return "falco"


def colore_relativo(nome):
    return {"colomba": COLOMBA, "tendenza colomba": "#5b9fd0", "nella media": INCHIOSTRO_2,
            "tendenza falco": "#d0665a", "falco": FALCO}[nome]


def risultato_finale(rs, oratore):
    pm = [r for r in rs if r["politica_monetaria"]]
    if not pm:
        return None
    punteggio = float(np.mean([r["punteggio"] for r in pm]))
    pct = float((DISC_OK.indice < punteggio).mean() * 100 + (DISC_OK.indice == punteggio).mean() * 50)
    ris = {"punteggio": punteggio, "n_pm": len(pm), "n_tot": len(rs), "percentile": pct,
           "classe_relativa": classe_relativa(pct),
           "quote": {k: sum(r["classe"] == k for r in pm) / len(pm) for k in ("falco", "colomba", "neutrale")}}
    u = Q[Q.concluso].iloc[-1]
    recenti = DISC_OK.sort_values("data").tail(10)
    ris["vs_trimestre"] = (u.trimestre, float(u.indice), punteggio - float(u.indice))
    ris["vs_recenti"] = (float(recenti.indice.mean()), punteggio - float(recenti.indice.mean()))
    if oratore:
        suoi = DISC_OK[DISC_OK.oratore == oratore]
        if len(suoi) >= 3:
            ris["vs_oratore"] = (oratore, float(suoi.indice.mean()), float((suoi.indice < punteggio).mean() * 100),
                                 len(suoi))
    ris["top_falco"] = sorted(pm, key=lambda r: -r["punteggio_continuo"])[:3]
    ris["top_colomba"] = sorted(pm, key=lambda r: r["punteggio_continuo"])[:3]
    return ris


def frasi_html(rs, corrente=None):
    parti = []
    for i, r in enumerate(rs):
        t = e(r["testo"])
        bordo = f"box-shadow:0 0 0 2px {ORO};" if i == corrente else ""
        if r["politica_monetaria"]:
            parti.append(f"<span title='{r['classe']} · punteggio {r['punteggio_continuo']:+.2f} · sicurezza "
                         f"{r['sicurezza'] * 100:.0f}%' style='background:{sfondo(r['classe'], r['sicurezza'])};"
                         f"border-radius:4px;padding:2px 3px;{bordo}'>{t}</span>")
        else:
            parti.append(f"<span style='color:{NON_RIL};{bordo}'>{t}</span>")
    return '<div class="testo-discorso">' + " ".join(parti) + "</div>"


def cambiamento(delta):
    if abs(delta) < 0.05:
        return "in linea", INCHIOSTRO_2, "="
    return (f"più da falco ({it(delta, segno=True)})", FALCO, "▲") if delta > 0 else \
        (f"più da colomba ({it(delta, segno=True)})", COLOMBA, "▼")


def mostra_risultato(ris, rs, chiave_sessione):
    if ris is None:
        st.info("Nessuna frase di politica monetaria: non c'è un punteggio da calcolare.", icon=":material/info:")
        return
    sezione("Il verdetto")
    if ris["n_pm"] < 5:
        st.warning(f"Solo {ris['n_pm']} frasi di politica monetaria: giudizio poco affidabile.", icon=":material/info:")
    cr = ris["classe_relativa"]
    st.html(f"""<div class="verdetto">
      <div><div class="lbl">Punteggio finale</div><div class="big">{it(ris['punteggio'], segno=True)}</div>
        <div style="color:{INCHIOSTRO_2};font-size:.86rem;margin-top:8px">{ris['n_pm']} frasi di politica monetaria
        su {ris['n_tot']}</div>{barra_classi(ris['quote'])}</div>
      <div><div class="lbl">Rispetto ai discorsi BCE dal 2018</div>
        <div class="cls" style="color:{colore_relativo(cr)}">{cr[0].upper() + cr[1:]}</div>
        <div style="color:{INCHIOSTRO_2};font-size:.9rem">più da falco del <b>{it(ris['percentile'], 0)}%</b> dei
        {len(DISC_OK)} discorsi con almeno {MIN_FRASI_DISCORSO} frasi</div>
        <div class="scala"><div class="seg" style="left:{ris['percentile']:.1f}%"></div></div>
        <div class="scala-lbl"><span>più colomba</span><span>nella media</span><span>più falco</span></div></div>
    </div>""")
    tq, iq, dq = ris["vs_trimestre"]
    mr, dr = ris["vs_recenti"]
    cambi = [(f"vs ultimo trimestre ({tq}, {it(iq, segno=True)})", dq), (f"vs ultimi 10 discorsi ({it(mr, segno=True)})", dr)]
    if "vs_oratore" in ris:
        o, mo, po, n = ris["vs_oratore"]
        cambi.append((f"vs {o} ({n} discorsi, media {it(mo, segno=True)})", ris["punteggio"] - mo))
    st.html('<div class="sez-sub" style="margin-top:6px">Cambiamento del tono: è il cambiamento, più del livello, '
            'ad aver anticipato le svolte dei tassi.</div><div class="cambi">' + "".join(
                f'<div class="cambio"><div class="t">{e(t)}</div><div class="v" style="color:{c}">{s} {v}</div></div>'
                for t, d in cambi for v, c, s in [cambiamento(d)]) + "</div>")
    s, t = st.columns(2, gap="large")
    with s:
        sezione("Frasi più da falco")
        st.html("".join(card_frase(r["testo"], r["classe"], r["punteggio_continuo"], r["sicurezza"])
                        for r in ris["top_falco"]))
    with t:
        sezione("Frasi più da colomba")
        st.html("".join(card_frase(r["testo"], r["classe"], r["punteggio_continuo"], r["sicurezza"])
                        for r in ris["top_colomba"]))

    sezione("Riassunto con l'LLM", "Kimi K3.1 Pro legge le frasi classificate e scrive 3–4 frasi di sintesi. "
            "Serve circa un minuto.")
    chiave = ad.chiave_kimi(st.secrets if hasattr(st, "secrets") else None)
    if not chiave:
        st.caption("Per attivarlo: chiave Kimi in `.streamlit/secrets.toml` (`KIMI_API_KEY = \"...\"`), poi riavviate.")
    elif not riassunto_sbloccato(chiave_sessione):
        pass
    elif st.button("Genera il riassunto", key=f"llm_{chiave_sessione}", icon=":material/auto_awesome:"):
        with st.spinner("L'LLM sta leggendo il discorso…"):
            try:
                st.session_state[f"riassunto_{chiave_sessione}"] = ad.riassunto_llm(rs, ris["punteggio"], chiave)
            except Exception as ex:  # noqa: BLE001
                st.error(f"Riassunto non disponibile: {ex}")
    if st.session_state.get(f"riassunto_{chiave_sessione}"):
        st.html(f'<div class="riassunto">{e(st.session_state[f"riassunto_{chiave_sessione}"])}</div>')


def riassunto_sbloccato(chiave_sessione):
    """Online la chiave Kimi è a carico nostro: se nei secrets c'è PASSWORD_RIASSUNTO, il riassunto si attiva solo
    dopo averla inserita (una volta per sessione). Senza PASSWORD_RIASSUNTO (in locale) è sempre attivo."""
    import hmac
    try:
        attesa = str(st.secrets.get("PASSWORD_RIASSUNTO", "")).strip()
    except Exception:  # noqa: BLE001  (nessun file secrets)
        attesa = ""
    if not attesa or st.session_state.get("riassunto_ok"):
        return True
    pw = st.text_input("Password per il riassunto", type="password", key=f"pw_{chiave_sessione}",
                       help="Il riassunto usa una chiave API a pagamento: è riservato a chi ha la password.")
    if pw and hmac.compare_digest(pw.strip(), attesa):
        st.session_state.riassunto_ok = True
        return True
    if pw:
        st.error("Password errata.")
    return False


def pannello_live(contenitore, rs, passo=0):
    pm = [r for r in rs if r["politica_monetaria"]]
    with contenitore.container(key=f"live_{passo}"):
        st.html(f'<div style="display:flex;align-items:center;gap:8px;color:{INCHIOSTRO};font-weight:600;'
                f'font-size:.9rem"><span class="vivo"></span>In tempo reale</div>')
        if not pm:
            st.caption("Ancora nessuna frase di politica monetaria.")
            return
        idx = np.mean([r["punteggio"] for r in pm])
        st.html(f'<div style="font-family:{F_TITOLI};font-size:2.8rem;font-weight:700;color:{INCHIOSTRO};'
                f'letter-spacing:-0.04em;line-height:1;margin-top:6px">{it(idx, segno=True)}</div>'
                f'<div style="font-size:.82rem;color:{INCHIOSTRO_2}">indice del discorso, su {len(pm)} frasi di '
                "politica monetaria</div>")
        serie = np.cumsum([r["punteggio"] for r in pm]) / np.arange(1, len(pm) + 1)
        fig = go.Figure(go.Scatter(y=serie, mode="lines", line=dict(color=BLU_UE, width=2.5), hoverinfo="skip"))
        fig.add_hline(y=0, line=dict(color="#9aa0b4", width=1))
        stile_grafico(fig, 150)
        fig.update_layout(margin=dict(l=4, r=4, t=8, b=4), showlegend=False,
                          yaxis=dict(range=[-1, 1], tickvals=[-1, 0, 1]), xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "staticPlot": True},
                        key=f"spark_{passo}")
        msg = "Servono almeno 12 frasi per confrontare l'andamento."
        if len(pm) >= 12:
            ult = np.mean([r["punteggio"] for r in pm[-10:]])
            diff = ult - idx
            msg = (f"Le ultime 10 frasi sono <b>più da {'falco' if diff > 0 else 'colomba'}</b> della media "
                   f"({it(ult, segno=True)})." if abs(diff) >= 0.1 else "Le ultime 10 frasi sono in linea con il resto.")
        ultima = pm[-1]
        st.html(f'<div style="font-size:.86rem;line-height:1.5">{msg}<br>Ultima frase: {pill(ultima["classe"])} '
                f'sicurezza {it(ultima["sicurezza"] * 100, 0)}%</div>')


def pagina_analizza():
    banner("Un discorso, frase per frase",
           "Incollate un testo, un link al sito della BCE o un video YouTube: ogni frase si colora mentre scorre, "
           "e alla fine arrivano il punteggio e il confronto con i discorsi BCE dal 2018.")
    oratori = ["(non indicato)"] + sorted(DISC_OK.oratore.dropna().unique())
    with st.container(key="card_input"):
        oratore = st.selectbox("Oratore (facoltativo: confronta il discorso con i suoi precedenti)", oratori)
        oratore = None if oratore == oratori[0] else oratore
        tab_testo, tab_video = st.tabs([":material/description: Testo o link BCE", ":material/smart_display: Video YouTube"])

        with tab_testo:
            url = st.text_input("Link a un discorso sul sito della BCE",
                                placeholder="https://www.ecb.europa.eu/press/key/date/…")
            testo = st.text_area("…oppure incollate il testo", height=150)
            x, y = st.columns([1, 2])
            tempo_reale = x.toggle("Scorrimento in tempo reale", value=True)
            velocita = y.slider("Frasi al secondo", 1, 10, 3, disabled=not tempo_reale)
            vai = st.button("Analizza il discorso", type="primary", icon=":material/play_arrow:")

        with tab_video:
            st.caption("Si usano i sottotitoli inglesi del video con i loro tempi: mentre il video scorre le frasi "
                       "compaiono colorate e l'indice si aggiorna. Video senza sottotitoli: incollate il testo.")
            link = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=…")
            prepara = st.button("Prepara il video", type="primary", icon=":material/smart_display:")

    if vai:
        try:
            st.session_state.pop("titolo", None)
            if url.strip():
                titolo, testo = ad.testo_da_url_bce(url.strip())
                st.session_state.titolo = titolo
            if not testo.strip():
                st.warning("Inserite un link o un testo.")
                st.stop()
            with st.spinner("Divido il testo in frasi e le classifico…"):
                rs = classifica_frasi(ad.dividi_in_frasi(testo))
            st.session_state.analisi_testo = {"rs": rs, "ris": risultato_finale(rs, oratore), "animare": tempo_reale,
                                              "velocita": velocita, "id": time.time()}
            st.session_state.pop("analisi_video", None)
        except Exception as ex:  # noqa: BLE001
            st.error(f"Non sono riuscito a leggere il discorso: {ex}")
    if prepara:
        try:
            with st.spinner("Scarico i sottotitoli e classifico le frasi…"):
                vid, frasi = ad.frasi_da_youtube(link)
                rs = classifica_frasi(frasi)
            st.session_state.analisi_video = {"vid": vid, "rs": rs, "ris": risultato_finale(rs, oratore),
                                              "id": time.time()}
            st.session_state.pop("analisi_testo", None)
        except Exception as ex:  # noqa: BLE001
            st.error(f"Sottotitoli non disponibili per questo video ({type(ex).__name__}). Incollate il testo.")

    a = st.session_state.get("analisi_testo")
    if a:
        rs = a["rs"]
        sezione(e(st.session_state.get("titolo") or "Il discorso"))
        if a["animare"]:
            # scorrimento nel browser: nessun rerun di Streamlit, si può mettere in pausa, cambiare velocità e saltare
            st.components.v1.html(live_html(rs, a["ris"], velocita=a["velocita"]), height=altezza_live(False),
                                  scrolling=True)
            with st.expander("Il verdetto completo e il riassunto, senza aspettare la fine", icon=":material/fact_check:"):
                mostra_risultato(a["ris"], rs, f"testo_{a['id']}")
        else:
            legenda()
            box, lato = st.columns([2.3, 1], gap="medium")
            box.html(frasi_html(rs))
            pannello_live(lato.empty(), rs, passo=len(rs) + 1)
            mostra_risultato(a["ris"], rs, f"testo_{a['id']}")

    v = st.session_state.get("analisi_video")
    if v:
        sezione("Il video", "Premete play: ogni frase compare colorata nel momento in cui viene pronunciata. Cliccate "
                "il sismografo o una riga della trascrizione per saltare a quel punto.")
        st.components.v1.html(live_html(v["rs"], v["ris"], vid=v["vid"]), height=altezza_live(True), scrolling=True)
        with st.expander("Il verdetto completo e il riassunto, senza aspettare la fine del video",
                         icon=":material/fact_check:"):
            mostra_risultato(v["ris"], v["rs"], f"video_{v['id']}")
    piede()


GAMMA_LIVE = 0.6     # quadrante del discorso: da −0,6 a +0,6 (il 90% dei discorsi BCE dal 2018 sta tra −0,46 e +0,35)
LIVE_TEMPLATE = (ASSETS.parent / "live.html")


def altezza_live(video):
    """Altezza iniziale dell'iframe; poi il pannello la adatta da solo al contenuto (ResizeObserver in live.html)."""
    return 1150 if video else 1120


def live_html(rs, ris, vid=None, velocita=3):
    """Pannello in tempo reale (app/live.html): quadrante a stelle con la lancetta che segue l'indice del discorso,
    didascalia della frase in ascolto, sismografo del tono lungo tutto il discorso, trascrizione che scorre e
    verdetto finale animato. Con vid segue il video YouTube; senza, simula lo scorrimento del testo."""
    dati = []
    for i, r in enumerate(rs):
        pm = r["politica_monetaria"]
        dati.append({"t": (r["inizio"] or 0) if vid else i, "testo": r["testo"], "pm": pm,
                     "classe": r.get("classe"), "p": r.get("punteggio", 0), "pc": round(r.get("punteggio_continuo", 0), 4),
                     "sic": round(r.get("sicurezza", 0), 4), "pf": round(r.get("p_falco", 0), 4),
                     "pn": round(r.get("p_neutrale", 0), 4), "pco": round(r.get("p_colomba", 0), 4),
                     "bg": sfondo(r["classe"], r["sicurezza"]) if pm else ""})
    fin = None
    if ris:
        cr = ris["classe_relativa"]
        fin = {"valore": round(ris["punteggio"], 4), "classe": cr[0].upper() + cr[1:], "colore": colore_relativo(cr),
               "pct": it(ris["percentile"], 0), "pos": round(ris["percentile"], 1), "n": ris["n_pm"],
               "barra": barra_classi(ris["quote"]).replace('class="barra"', 'class="fbarra"'),
               "trim": f"{ris['vs_trimestre'][0]}: {cambiamento(ris['vs_trimestre'][2])[0]}",
               "recenti": cambiamento(ris["vs_recenti"][1])[0]}
    js = lambda x: json.dumps(x, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731
    if vid:
        media, dida, invito = '<div class="video"><div id="player"></div></div>', "", \
            "Premete play: qui compare la frase che si sta ascoltando, colorata secondo il modello."
        comandi = ""
        inizio = "0:00"
    else:
        media, dida, invito = "", "grande", "Il discorso sta per cominciare…"
        comandi = (f'<button class="btn oro" id="play">Pausa</button><label class="velocita">velocità '
                   f'<input type="range" id="vel" min="1" max="10" value="{int(velocita)}"> '
                   f'<b id="vel-n">{int(velocita)}</b> frasi al secondo</label>')
        inizio = "prima frase"
    sost = {"__DATI__": js(dati), "__FIN__": js(fin), "__VIDEO__": js(vid), "__GAMMA_NUM__": str(GAMMA_LIVE),
            "__GAMMA__": it(GAMMA_LIVE, 1), "__VELOCITA__": str(int(velocita)), "__MEDIA__": media,
            "__DIDA_CLASSE__": dida, "__INVITO__": invito, "__COMANDI__": comandi, "__INIZIO_LBL__": inizio,
            "__STELLINA__": STELLINA, "__BLU_UE__": BLU_UE, "__ABISSO__": ABISSO, "__ORO__": ORO,
            "__ORO_SCURO__": ORO_SCURO, "__INK__": INCHIOSTRO, "__INK2__": INCHIOSTRO_2, "__CARTA__": CARTA,
            "__BORDO__": BORDO, "__FALCO__": FALCO, "__COLOMBA__": COLOMBA, "__NEUTRALE__": NEUTRALE,
            "__NONRIL__": NON_RIL}
    pagina = LIVE_TEMPLATE.read_text(encoding="utf-8")
    for k in ("__DATI__", "__FIN__"):                       # prima i dati: il loro testo non va toccato dopo
        pagina = pagina.replace(k, "\x00" + k + "\x00")
    for k, v in sost.items():
        if k not in ("__DATI__", "__FIN__"):
            pagina = pagina.replace(k, v)
    for k in ("__DATI__", "__FIN__"):
        pagina = pagina.replace("\x00" + k + "\x00", sost[k])
    return pagina


applica_stile()
pg = st.navigation([st.Page(pagina_panoramica, title="Panoramica", icon=":material/monitoring:", default=True),
                    st.Page(pagina_esplora, title="Esplora un trimestre", icon=":material/travel_explore:",
                            url_path="trimestre"),
                    st.Page(pagina_modello, title="Il modello", icon=":material/neurology:", url_path="modello"),
                    st.Page(pagina_analizza, title="Analizza un discorso", icon=":material/record_voice_over:",
                            url_path="discorso")])
pg.run()
