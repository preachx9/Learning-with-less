from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LANGS={'yor':'Yorùbá','ibo':'Igbo','hau':'Hausa','pcm':'Nigerian Pidgin'}
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
frames=[];audits={};counts={}
for code,name in LANGS.items():
    folder=ROOT/'languages'/code/'results'
    d=pd.read_csv(folder/'summary.csv');d['language']=name;d['code']=code;frames.append(d)
    audits[code]=json.loads((folder/'audit.json').read_text())
    counts[code]=json.loads((folder/'class_counts.json').read_text())
summary=pd.concat(frames,ignore_index=True)
summary.to_csv(ROOT/'combined_summary.csv',index=False)
comparison=summary.pivot(index=['code','language','fraction','n_train'],columns='approach',values='macro_f1_mean').reset_index()
comparison['character_minus_word_points']=100*(comparison.Character-comparison.Word)
comparison.to_csv(ROOT/'comparison.csv',index=False)
print(comparison.to_string(index=False),flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for ax,(code,name) in zip(axes.flat,LANGS.items()):
    part=summary[summary.code==code]
    for approach,color,marker in [('Word','#176B8C','o'),('Character','#C45B29','s')]:
        d=part[part.approach==approach].sort_values('fraction')
        ax.plot(d.fraction*100,d.macro_f1_mean*100,color=color,marker=marker,lw=2.3,label=approach)
        e=d[d.n_runs>1]
        ax.errorbar(e.fraction*100,e.macro_f1_mean*100,yerr=e.macro_f1_sd*100,fmt='none',ecolor=color,capsize=4)
    ntest=audits[code]['retained']['test'];nclass=len(counts[code]['train'])
    ax.set(title=f'{name} | {nclass} categories | {ntest} test articles',ylim=(45,95),xticks=[25,50,100],xlabel='Training budget (%)',ylabel='Test macro-F1 (0–100)')
    sizes=part[part.approach=='Word'].sort_values('fraction').n_train.tolist()
    ax.set_xticklabels([f'{f}%\n(n={n:,})' for f,n in zip([25,50,100],sizes)])
    ax.grid(axis='y',alpha=.17)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
for ext in ['png','pdf']:fig.savefig(FIG/f'four_language_learning_curves.{ext}',dpi=230)
plt.close(fig)

