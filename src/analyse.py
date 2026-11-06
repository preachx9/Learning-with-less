from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,classification_report
ROOT=Path(__file__).resolve().parent
LANGS={'yor':'Yorùbá','ibo':'Igbo','hau':'Hausa','pcm':'Nigerian Pidgin'}
rows=[]; classes=[]; disagreements=[]
for li,(code,name) in enumerate(LANGS.items()):
 d=pd.read_csv(ROOT/'languages'/code/'results/test_predictions.csv')
 base=d[(d.fraction==1)&(d.approach=='Word')].sort_values('row_id')
 truth=base.true.to_numpy(); ids=base.row_id.to_numpy(); labels=sorted(set(truth)); n=len(truth)
 rng=np.random.default_rng(20261019+li)
 weights=rng.multinomial(n,np.full(n,1/n),size=10000).astype(float)
 def scores(pred,w):
  vals=[]
  for label in labels:
   t=(truth==label);p=(pred==label)
   tp=w@(t&p);den=w@t+w@p
   vals.append(np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0))
  return np.mean(vals,axis=0)*100
 for fraction in [.25,.5,1.]:
  boot={};point={};preds={}
  for approach in ['Word','Character']:
   arrays=[];points=[]
   for seed,g in d[(d.fraction==fraction)&(d.approach==approach)].groupby('seed'):
    g=g.sort_values('row_id');assert np.array_equal(g.row_id,ids) and np.array_equal(g.true,truth)
    pred=g.predicted.to_numpy();arrays.append(scores(pred,weights));points.append(f1_score(truth,pred,average='macro',labels=labels)*100)
    assert np.isclose(scores(pred,np.ones((1,n)))[0],points[-1])
    if fraction==1:
     preds[approach]=pred
     report=classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0)
     for label in labels:classes.append(dict(code=code,language=name,approach=approach,category=label,**report[label]))
   boot[approach]=np.mean(arrays,axis=0);point[approach]=np.mean(points)
  delta=boot['Character']-boot['Word'];lo,hi=np.quantile(delta,[.025,.975])
  rows.append(dict(code=code,language=name,fraction=fraction,n_test=n,word=point['Word'],character=point['Character'],difference=point['Character']-point['Word'],ci_low=lo,ci_high=hi,bootstrap_samples=10000))
  if fraction==1:
   for i in np.flatnonzero(preds['Word']!=preds['Character']):disagreements.append(dict(code=code,row_id=ids[i],true=truth[i],word=preds['Word'][i],character=preds['Character'][i],word_correct=preds['Word'][i]==truth[i],character_correct=preds['Character'][i]==truth[i]))
out=pd.DataFrame(rows);out.to_csv(ROOT/'bootstrap_intervals.csv',index=False)
pd.DataFrame(classes).to_csv(ROOT/'per_class_full_data.csv',index=False)
pd.DataFrame(disagreements).to_csv(ROOT/'full_data_disagreements.csv',index=False)
print(out.to_string(index=False))
