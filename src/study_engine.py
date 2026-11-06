"""Frozen comparison: 2 representations x 3 budgets, matched stratified subsets."""
from pathlib import Path
from collections import Counter, defaultdict
import json, time, warnings, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from data_utils import load_split, word_tokens, REVISION

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
SEEDS=[11,22,33,44,55]
PROTOCOL={
 'recorded_utc':datetime.now(timezone.utc).isoformat(),
 'data_revision':REVISION,'seeds':SEEDS,'fractions':[0.25,0.5,1.0],
 'inputs':'NFC/lowercased article bodies only; preserve combining marks; no headlines or URLs',
 'word_features':'word unigrams and bigrams; Unicode letter/mark/number tokenizer',
 'character_features':'character 3-5 grams, including spaces',
 'shared_tfidf':{'min_df':2,'max_features':50000,'sublinear_tf':True,'norm':'l2'},
 'classifier':{'type':'LogisticRegression','C':1.0,'solver':'lbfgs','max_iter':1000},
 'sampling':'For each seed, permute each training class; take floor(fraction * class size) from each class. Nested matched subsets. Full training set fitted once per approach.',
 'deduplication':'Remove empty bodies; greedy first-occurrence dedup by normalized exact body, nonempty URL, or >=0.80 Jaccard of word 5-gram sets. Training is priority, then dev, then test. Do not use labels for overlap filtering.',
 'evaluation':'Official test split after prespecified overlap filtering; no model selection or tuning from test results. Macro-F1 primary; accuracy secondary. Mean and sample SD over five subsets at 25/50%; 100% single deterministic fit.',
 'scope':'Exploratory comparison on one corpus; no significance or state-of-the-art claim; subset SD is not a test-population confidence interval.'}

class DuplicateIndex:
    def __init__(self):
        self.texts={}; self.urls={}; self.shingles=[]; self.refs=[]; self.postings=defaultdict(list)
    def check_add(self,text,url,ref):
        if not text:return {'ref':ref,'reason':'empty'}
        if text in self.texts:return {'ref':ref,'matches':self.texts[text],'reason':'exact_text'}
        if url and url in self.urls:return {'ref':ref,'matches':self.urls[url],'reason':'url'}
        tokens=word_tokens(text)
        shingles=set(zip(*(tokens[i:] for i in range(5))))
        counts=Counter()
        for s in shingles:counts.update(self.postings.get(s,[]))
        for i,intersection in counts.items():
            jaccard=intersection/(len(shingles)+len(self.shingles[i])-intersection)
            if jaccard>=0.8:return {'ref':ref,'matches':self.refs[i],'reason':'near_duplicate','jaccard':jaccard}
        i=len(self.refs);self.refs.append(ref);self.shingles.append(shingles)
        self.texts[text]=ref
        if url:self.urls[url]=ref
        for s in shingles:self.postings[s].append(i)
        return None

def main():
    # Write protocol before loading/evaluating test. No alterations based on outcomes.
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    idx=DuplicateIndex();frames={};audit={'original':{},'retained':{},'sha256':{},'removed':[]}
    for split in ['train','dev','test']:
        df,sha=load_split(split);audit['original'][split]=len(df);audit['sha256'][split]=sha
        keep=[]
        for rowid,row in df.iterrows():
            removal=idx.check_add(row['input'],row['url'],f'{split}:{rowid}')
            if removal:audit['removed'].append(removal)
            else:keep.append(rowid)
        frames[split]=df.loc[keep].copy();audit['retained'][split]=len(keep)
        print('AUDIT',split,len(df),'->',len(keep),flush=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    train=frames['train'];test=frames['test'];labels=sorted(train['category'].unique())
    assert set(test['category'])==set(labels)
    counts={split:{str(k):int(v) for k,v in df['category'].value_counts().sort_index().items()} for split,df in frames.items()}
    (OUT/'class_counts.json').write_text(json.dumps(counts,indent=2))
    majority=train['category'].value_counts().idxmax()
    (OUT/'majority_baseline.json').write_text(json.dumps({'category':majority,
       'macro_f1':f1_score(test['category'],[majority]*len(test),average='macro',labels=labels,zero_division=0),
       'accuracy':accuracy_score(test['category'],[majority]*len(test))},indent=2))
    rows=[];allpred=[];samples={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        permutations={c:rng.permutation(train.index[train['category']==c].to_numpy()) for c in labels}
        for frac in [0.25,0.5,1.0]:
            if frac==1 and seed!=SEEDS[0]:continue
            ids=np.concatenate([permutations[c][:max(1,int(np.floor(frac*len(permutations[c]))))] for c in labels])
            ids=np.sort(ids);subset=train.loc[ids]
            samples[f'{seed}_{frac}']=[int(i) for i in ids]
            for approach in ['Word','Character']:
                opts=dict(min_df=2,max_features=50000,sublinear_tf=True,lowercase=False)
                vec=TfidfVectorizer(tokenizer=word_tokens,token_pattern=None,ngram_range=(1,2),**opts) if approach=='Word' else TfidfVectorizer(analyzer='char',ngram_range=(3,5),**opts)
                model=Pipeline([('tfidf',vec),('classifier',LogisticRegression(C=1.,solver='lbfgs',max_iter=1000))])
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(subset['input'],subset['category'])
                seconds=time.perf_counter()-start
                n_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
                if n_warnings:raise RuntimeError('Convergence warning: results not ready to report.')
                predicted=model.predict(test['input'])
                result={'approach':approach,'fraction':frac,'seed':seed,'n_train':len(subset),'n_test':len(test),
                    'macro_f1':f1_score(test['category'],predicted,average='macro',labels=labels,zero_division=0),
                    'accuracy':accuracy_score(test['category'],predicted),'fit_seconds':seconds,'features':len(vec.vocabulary_)}
                rows.append(result)
                print('RESULT',result,flush=True)
                for rowid,true,pred in zip(test.index,test['category'],predicted):allpred.append({'approach':approach,'fraction':frac,'seed':seed,'row_id':int(rowid),'true':true,'predicted':pred})
                pd.DataFrame(rows).to_csv(OUT/'runs.csv',index=False)
                if frac==1:
                    (OUT/f'{approach.lower()}_classification_report.json').write_text(json.dumps(classification_report(test['category'],predicted,labels=labels,output_dict=True,zero_division=0),indent=2))
                    pd.DataFrame(confusion_matrix(test['category'],predicted,labels=labels),index=labels,columns=labels).to_csv(OUT/f'{approach.lower()}_confusion.csv')
    pd.DataFrame(allpred).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'training_row_ids.json').write_text(json.dumps(samples))
    runs=pd.DataFrame(rows)
    summary=runs.groupby(['approach','fraction','n_train']).agg(n_runs=('seed','count'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    print('SUMMARY\n'+summary.to_string(index=False),flush=True)
    assert len(rows)==22 and len(allpred)==22*len(test)

if __name__=='__main__':main()
