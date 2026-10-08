"""Reproducible, chronological Samsung close-only forecast benchmark."""
import argparse, json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
WINDOWS = [5, 20, 60, 100, 250]
HORIZON = 20

def feature(p):
    # Exactly w closes: w-1 daily simple returns, oldest first.
    return np.diff(p) / p[:-1]

def metrics(y, pred, last):
    e = y-pred
    return dict(mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
                mape=float(np.mean(abs(e)/y)*100),
                direction=float(np.mean(np.sign(pred-last)==np.sign(y-last))*100))

def estimator(kind, alpha=100):
    if kind == 'Ridge': return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_leaf=25,
                                 random_state=42, n_jobs=-1)

def export_model(model, kind, w):
    out = dict(kind=kind, window=w)
    if kind == 'Ridge':
        scale, reg = model.steps[0][1], model.steps[1][1]
        out.update(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                   coef=reg.coef_.tolist(), intercept=float(reg.intercept_))
    else:
        out['trees'] = [dict(left=t.tree_.children_left.tolist(), right=t.tree_.children_right.tolist(),
                            feature=t.tree_.feature.tolist(), threshold=t.tree_.threshold.tolist(),
                            value=t.tree_.value[:,0,0].tolist()) for t in model.estimators_]
    return out

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--as-of', default='2026-10-08'); args=parser.parse_args()
    (ROOT/'data').mkdir(exist_ok=True); (ROOT/'docs').mkdir(exist_ok=True)
    rawpath=ROOT/'data/yahoo.json'
    if args.refresh or not rawpath.exists():
        r=requests.get('https://query1.finance.yahoo.com/v8/finance/chart/005930.KS',
                       params={'period1':1420070400,'period2':int(pd.Timestamp(args.as_of, tz='Asia/Seoul').timestamp()),'interval':'1d'},
                       headers={'User-Agent':'Mozilla/5.0'}, timeout=60)
        r.raise_for_status(); rawpath.write_text(r.text, encoding='utf-8')
    raw=json.loads(rawpath.read_text(encoding='utf-8'))['chart']['result'][0]
    frame=pd.DataFrame({'date':pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('Asia/Seoul').strftime('%Y-%m-%d'),
                        'close':raw['indicators']['quote'][0]['close']}).dropna()
    frame=frame[(frame.date < args.as_of)&(frame.date>='2018-05-04')].sort_values('date').reset_index(drop=True)
    assert frame.date.is_unique and (frame.close>0).all()
    frame.to_csv(ROOT/'data/close.csv',index=False)
    frame.to_csv(ROOT/'docs/close.csv',index=False)
    p=frame.close.to_numpy(); dates=frame.date.to_numpy(); idx=np.arange(max(WINDOWS)+HORIZON-1,len(p))
    y=p[idx]; last=p[idx-HORIZON]; target=y/last-1
    origin=dates[idx-HORIZON]
    test=(dates[idx]>='2026-01-01')&(dates[idx]<'2027-01-01')&(origin>='2026-01-01'); pre=dates[idx]<'2026-01-01'
    if not test.any(): raise RuntimeError('No real 2026 observations; cannot evaluate.')
    rows=[]; models={}; preds={}
    # Expanding annual validation. 2026 is never used to select alpha/model/window.
    folds=[(dates[idx]<'2024-01-01',(origin>='2024-01-01')&(dates[idx]<'2025-01-01')),
           (dates[idx]<'2025-01-01',(origin>='2025-01-01')&(dates[idx]<'2026-01-01'))]
    baseline=metrics(y[test],last[test],last[test])
    rows.append(dict(id='naive',kind='기준일 종가 유지',window=1,validation_mae=float(np.mean(np.concatenate([abs(y[v]-last[v]) for _,v in folds]))),**baseline))
    preds['naive']=last[test].tolist(); models['naive']={'kind':'Naive','window':1}
    for w in WINDOWS:
        X=np.array([feature(p[i-HORIZON-w+1:i-HORIZON+1]) for i in idx])
        for kind in ['Ridge','RandomForest']:
            candidates=[1,100,10000] if kind=='Ridge' else [100]
            scores=[]
            for alpha in candidates:
                errors=[]
                for tr,va in folds:
                    m=estimator(kind,alpha).fit(X[tr],target[tr]); q=last[va]*(1+m.predict(X[va]))
                    errors.extend(abs(y[va]-q))
                scores.append(float(np.mean(errors)))
            j=int(np.argmin(scores)); m=estimator(kind,candidates[j]).fit(X[pre],target[pre])
            pred=last[test]*(1+m.predict(X[test])); key=f'{kind}-{w}'
            rows.append(dict(id=key,kind=kind,window=w,alpha=candidates[j] if kind=='Ridge' else None,
                             validation_mae=scores[j],**metrics(y[test],pred,last[test])))
            models[key]=export_model(m,kind,w); preds[key]=pred.tolist()
            print(key, round(scores[j],2), round(rows[-1]['mae'],2),flush=True)
    selected=min(rows,key=lambda x:x['validation_mae'])['id']
    chosen100=min((r for r in rows if r['window']==100),key=lambda x:x['validation_mae'])['id']
    # Empirical calibration band from validation residuals only, selected model frozen.
    bands={}
    for row in rows:
        errors=[]
        for tr,va in folds:
            if row['id']=='naive': q=last[va]
            else:
                w=row['window']; X=np.array([feature(p[i-HORIZON-w+1:i-HORIZON+1]) for i in idx])
                m=estimator(row['kind'],row['alpha'] or 100).fit(X[tr],target[tr]); q=last[va]*(1+m.predict(X[va]))
            errors.extend(abs(y[va]-q)/last[va])
        bands[row['id']]=float(np.quantile(errors,.9))
    result=dict(horizon=HORIZON,origin_dates=origin[test].tolist(),as_of=args.as_of,generated_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                data_start=str(dates[0]),data_end=str(dates[-1]),observations=len(p),
                source='Yahoo Finance / 005930.KS / unadjusted daily Close',
                data_sha256=hashlib.sha256((ROOT/'data/close.csv').read_bytes()).hexdigest(),
                selected=selected,selected100=chosen100,rows=rows,bands=bands,
                test_dates=dates[idx][test].tolist(),actual=y[test].tolist(),last=last[test].tolist(),predictions=preds,
                recent=frame.tail(250).to_dict(orient='records'))
    (ROOT/'docs/results.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    (ROOT/'docs/models.json').write_text(json.dumps(models),encoding='utf-8')
    print(json.dumps({'selected':selected,'selected100':chosen100,'test_n':int(test.sum()),'end':dates[-1]},ensure_ascii=False))

if __name__=='__main__': main()
