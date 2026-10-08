import json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from train import feature, estimator, WINDOWS, HORIZON

root=Path(__file__).resolve().parent
r=json.loads((root/'docs/results.json').read_text(encoding='utf-8'))
f=pd.read_csv(root/'data/close.csv'); p=f.close.to_numpy(); dates=f.date.to_numpy()
assert dates[-1]<r['as_of'] and all(d.startswith('2026-') for d in r['test_dates'])
assert len(r['test_dates'])==len(r['actual']) and f.date.is_unique
assert r['selected']==min(r['rows'],key=lambda row:row['validation_mae'])['id']
assert r['selected100']==min((row for row in r['rows'] if row['window']==100),key=lambda row:row['validation_mae'])['id']
models=json.loads((root/'docs/models.json').read_text())
idx=np.arange(max(WINDOWS)+HORIZON-1,len(p)); pre=dates[idx]<'2026-01-01'; test=(dates[idx]>='2026-01-01')&(dates[idx-HORIZON]>='2026-01-01')
assert r['horizon']==20
assert all(d>='2026-01-01' for d in r['origin_dates'])
for origin,target_date in zip(r['origin_dates'],r['test_dates']):
    assert list(dates).index(target_date)-list(dates).index(origin)==20
cases=[]
for row in r['rows']:
    w=row['window']; i=idx[test][0]; sample=p[i-HORIZON-w+1:i-HORIZON+1]
    if row['id']!='naive':
        X=np.array([feature(p[j-HORIZON-w+1:j-HORIZON+1]) for j in idx]); y=p[idx]/p[idx-HORIZON]-1
        m=estimator(row['kind'],row['alpha'] or 100).fit(X[pre],y[pre])
        assert np.allclose(p[idx[test]-HORIZON]*(1+m.predict(X[test])),r['predictions'][row['id']])
    cases.append({'key':row['id'],'prices':sample.tolist(),'expected':r['predictions'][row['id']][0]})
node="""const fs=require('fs'),vm=require('vm');const ctx={document:{getElementById:()=>({})},fetch:()=>new Promise(()=>{})};vm.createContext(ctx);vm.runInContext(fs.readFileSync('docs/app.js','utf8'),ctx);const data=JSON.parse(fs.readFileSync(0,'utf8'));for(const c of data.cases){const q=ctx.infer(data.models[c.key],c.prices);if(Math.abs(q-c.expected)>1e-6)throw Error(c.key+' parity mismatch');}console.log('Browser inference parity: '+data.cases.length+' models');"""
out=subprocess.run(['node','-e',node],input=json.dumps({'models':models,'cases':cases}),text=True,capture_output=True,cwd=root)
assert out.returncode==0,out.stderr
print(out.stdout.strip()); print('Temporal split, selection, all test predictions: PASS')
