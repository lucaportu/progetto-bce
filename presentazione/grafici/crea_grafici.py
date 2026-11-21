import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib import rcParams
rcParams.update({"font.family":"Helvetica Neue","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                 "axes.edgecolor":"#9aa3b8","axes.labelcolor":"#33405e","xtick.color":"#33405e","ytick.color":"#33405e"})
BLU="#003399"; ABISSO="#001233"; ORO="#a87800"; FALCO="#d2402f"; COLOMBA="#1e95d0"; NEUTRO="#8a94aa"
P="/Users/lucaportugalli/Desktop/AIDA/AI and Computer Vision/progetto_BCE/"
it=lambda x: f"{x:.2f}".replace(".",",")

# 1. Confronto F1 sul test
lab=["Classe più\nfrequente","A\nsolo Fed","B2\nsolo BCE","B\nFed + BCE","C\nLLM (Kimi)"]
f1=[0.226,0.592,0.611,0.633,0.763]; lo=[None,0.53,0.55,0.57,0.71]; hi=[None,0.65,0.67,0.69,0.82]
seed=[None,0.566,0.603,0.615,None]
col=["#c9cfdc","#7f93c8","#4d6bbd",BLU,ORO]
fig,ax=plt.subplots(figsize=(7.2,3.3),dpi=220)
x=np.arange(len(lab)); ax.bar(x,f1,color=col,width=0.62,zorder=2)
for i in range(len(lab)):
    if lo[i] is not None:
        ax.errorbar(i,f1[i],yerr=[[f1[i]-lo[i]],[hi[i]-f1[i]]],color=ABISSO,capsize=4,lw=1.2,zorder=3)
    ax.text(i,(hi[i] or f1[i])+0.02,it(f1[i]),ha="center",va="bottom",fontsize=10.5,fontweight="bold",color=ABISSO)
    if seed[i]: ax.plot(i,seed[i],marker="D",color="white",mec=ABISSO,ms=5.5,zorder=4)
ax.set_xticks(x,lab); ax.set_ylim(0,0.95); ax.set_ylabel("F1 pesato sul test (244 frasi)")
ax.yaxis.grid(True,color="#e3e7f0",zorder=0); ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:.1f}".replace(".",",")))
ax.plot([],[],marker="D",color="white",mec=ABISSO,ls="",label="media su 10 addestramenti (Python)")
ax.errorbar([],[],yerr=[],color=ABISSO,capsize=4,label="intervallo di confidenza 95% (bootstrap)")
ax.legend(frameon=False,fontsize=8.5,loc="upper left")
fig.tight_layout(); fig.savefig("g_confronto.png"); plt.close(fig)

# 2. Matrici di confusione
cms={"B – rete neurale (Fed + BCE)":np.array([[41,16,11],[14,51,14],[17,18,62]]),
     "C – LLM Kimi K3.1 Pro":np.array([[56,10,2],[5,73,1],[16,23,58]])}
cl=["colomba","falco","neutrale"]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),dpi=220)
for ax,(t,m) in zip(axs,cms.items()):
    r=m/m.sum(1,keepdims=True)
    ax.imshow(r,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b",["#ffffff","#003399"]),vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{m[i,j]}\n{r[i,j]*100:.0f}%",ha="center",va="center",fontsize=9,
                    color="white" if r[i,j]>0.5 else ABISSO,fontweight="bold" if i==j else "normal")
    ax.set_xticks(range(3),cl); ax.set_yticks(range(3),cl); ax.set_title(t,fontsize=10,color=ABISSO,fontweight="bold")
    ax.set_xlabel("previsto dal modello"); ax.set_ylabel("etichetta degli esperti")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
fig.tight_layout(); fig.savefig("g_matrici.png"); plt.close(fig)

# 3. Correlazioni per ritardo
fig,ax=plt.subplots(figsize=(6.4,2.7),dpi=220)
v=[0.40,0.40,0.36,0.26]; lb=["stesso\ntrimestre (t)","1 trimestre\ndopo (t+1)","2 trimestri\ndopo (t+2)","3 trimestri\ndopo (t+3)"]
c=[ "#7f93c8",BLU,"#4d6bbd","#9fb0dc"]
ax.bar(range(4),v,color=c,width=0.58,zorder=2)
for i,val in enumerate(v): ax.text(i,val+0.012,"+"+it(val),ha="center",fontweight="bold",color=ABISSO)
ax.axhline(0.76,color=ORO,ls="--",lw=1.2); ax.text(3.35,0.77,"persistenza dei tassi (variazione in t e in t+1): +0,76",ha="right",va="bottom",color=ORO,fontsize=8.5)
ax.set_xticks(range(4),lb); ax.set_ylim(0,0.9); ax.set_ylabel("correlazione con\nla variazione del DFR")
ax.yaxis.grid(True,color="#e3e7f0",zorder=0); ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:.1f}".replace(".",",")))
fig.tight_layout(); fig.savefig("g_ritardi.png"); plt.close(fig)

# 4. CV vs test per approccio e modello
mod=["logistica C=10","logistica C=1","rete α=0,01","rete α=1"]
cv={"A":[0.577,0.598,0.608,0.612],"B":[0.585,0.608,0.616,0.634],"B2":[0.583,0.627,0.623,0.643]}
te={"A":[0.564,0.559,0.569,0.592],"B":[0.596,0.606,0.611,0.633],"B2":[0.574,0.595,0.596,0.611]}
fig,axs=plt.subplots(1,3,figsize=(7.4,2.6),dpi=220,sharey=True)
names={"A":"A – solo Fed","B":"B – Fed + BCE","B2":"B2 – solo BCE"}
for ax,k in zip(axs,["A","B2","B"]):
    xx=np.arange(4)
    ax.bar(xx-0.18,cv[k],0.34,color="#c9cfdc",label="cross-validation (addestramento)",zorder=2)
    ax.bar(xx+0.18,te[k],0.34,color=BLU,label="test 2023–24",zorder=2)
    ax.set_xticks(xx,mod,rotation=35,ha="right",fontsize=8); ax.set_ylim(0.5,0.67); ax.set_title(names[k],fontsize=9.5,color=ABISSO,fontweight="bold")
    ax.yaxis.grid(True,color="#e3e7f0",zorder=0); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:.2f}".replace(".",",")))
axs[0].set_ylabel("F1 pesato"); axs[0].legend(frameon=False,fontsize=7.5,loc="upper left")
fig.tight_layout(); fig.savefig("g_cv.png"); plt.close(fig)
print("ok")
