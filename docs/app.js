'use strict';
let R,M,P,inputContext=null,randomIndex=null; const $=id=>document.getElementById(id), fmt=x=>Math.round(x).toLocaleString('ko-KR'), pct=x=>x.toFixed(2)+'%';
function fillRecent(){
 inputContext=null;
 const w=M[$('model').value].window,rows=R.recent.slice(-w);
 $('prices').value=rows.map(r=>r.close).join(', ');
 $('setupStatus').textContent=`가장 최근 ${w}거래일 데이터 기준으로 설정되었습니다. (${rows[0].date} ~ ${rows.at(-1).date})`;
 predict();
}
function renderRandom(){
 if(randomIndex===null)return;
 const i=randomIndex,key=$('randomModel').value,row=R.rows.find(r=>r.id===key),actual=R.actual[i],base=R.predictions.naive[i],pred=R.predictions[key][i],baseError=Math.abs(base-actual),modelError=Math.abs(pred-actual);
 const target=P.findIndex(r=>r.date===R.test_dates[i]),end=target-(R.horizon||1)+1;
 const inputs=P.slice(end-row.window,end).map(r=>r.close);
 inputContext={prices:inputs.join(', '),actual,date:R.test_dates[i],origin:R.origin_dates?R.origin_dates[i]:P[end-1].date};
 $('model').value=key;$('prices').value=inputContext.prices;$('setupStatus').textContent=`목표 ${inputContext.date}의 예측을 위해 ${P[end-row.window].date} ~ ${inputContext.origin}의 ${row.window}거래일 데이터로 설정되었습니다.`;predict();
 $('randomResult').hidden=false;$('randomDate').textContent=`목표 ${R.test_dates[i]} · 기준일 ${R.origin_dates[i]} · 20거래일 뒤`;
 $('randomActual').textContent=fmt(actual)+'원';$('randomBase').textContent=fmt(base)+'원';$('randomPrediction').textContent=fmt(pred)+'원';
 $('randomModelLabel').textContent=`모델 예측 종가 · ${row.kind} · ${row.window}거래일`;
 $('randomBaseError').textContent=`오차 ${fmt(baseError)}원 · ${pct(baseError/actual*100)}`;
 $('randomModelError').textContent=`오차 ${fmt(modelError)}원 · ${pct(modelError/actual*100)}`;
 const diff=baseError-modelError;
 $('randomVerdict').textContent=Math.abs(diff)<1e-6?'이 날짜에는 두 모델의 오차가 같습니다.':`이 날짜에는 ${diff>0?'학습 모델':'기준 모델'}이 실제 종가에 ${fmt(Math.abs(diff))}원 더 가까웠습니다.`;
}
function randomCompare(){
 const eligible=R.test_dates.map((date,i)=>({date,i})).filter(x=>x.date>='2026-01-01'&&x.i!==randomIndex);
 if(!eligible.length)return;
 randomIndex=eligible[Math.floor(Math.random()*eligible.length)].i;renderRandom();
}
function infer(model,prices){
 if(model.kind==='Naive')return prices.at(-1);
 const x=prices.slice(1).map((v,i)=>(v-prices[i])/prices[i]);let ret;
 if(model.kind==='Ridge')ret=model.intercept+x.reduce((s,v,i)=>s+(v-model.mean[i])/model.scale[i]*model.coef[i],0);
 else ret=model.trees.reduce((s,t)=>{let n=0;while(t.left[n]!==-1)n=x[t.feature[n]]<=t.threshold[n]?t.left[n]:t.right[n];return s+t.value[n];},0)/model.trees.length;
 return prices.at(-1)*(1+ret);
}
function parsePrices(text){const lines=text.trim().split(/\r?\n/);if(lines[0].trim().toLowerCase()==='date,close'){return lines.slice(1).filter(l=>l.trim()).map(l=>{const parts=l.trim().split(',');if(parts.length!==2||!/^\d{4}-\d{2}-\d{2}$/.test(parts[0]))throw Error('CSV 형식은 date,close 헤더와 날짜,종가 두 열이어야 합니다.');return Number(parts[1]);});}return text.trim().split(/[\s,;]+/).filter(Boolean).map(Number);}
function predict(){try{const key=$('model').value,m=M[key],p=parsePrices($('prices').value);if(p.length!==m.window)throw Error(`${m.window}개 종가가 필요합니다. 현재 ${p.length}개입니다.`);if(p.some(v=>!Number.isFinite(v)||v<=0))throw Error('모든 종가는 0보다 큰 숫자여야 합니다.');const q=infer(m,p);if(!Number.isFinite(q)||q<=0)throw Error('유효한 예측을 만들 수 없는 입력입니다.');$('forecast').textContent=fmt(q)+'원';
 if(inputContext&&$('prices').value===inputContext.prices){const e=Math.abs(q-inputContext.actual);$('forecastActual').textContent=fmt(inputContext.actual)+'원';$('forecastError').textContent=`${fmt(e)}원 · ${pct(e/inputContext.actual*100)}`;$('forecastContext').textContent=`목표 ${inputContext.date} · 입력 마지막 날짜 ${inputContext.origin} · 실제보다 ${q>inputContext.actual?'높게':q<inputContext.actual?'낮게':'같게'} 예측`;}else{$('forecastActual').textContent='아직 미확정';$('forecastError').textContent='아직 평가할 수 없음';$('forecastContext').textContent=inputContext?'실제값과 연결되지 않은 입력입니다.':$('setupStatus').textContent.startsWith('가장 최근')?'최신 종가 이후의 미래 예측입니다. 실제 종가가 확정되면 오차를 계산할 수 있습니다.':'날짜가 없는 입력은 실제값과 연결할 수 없습니다. 랜덤 날짜 비교를 누르면 실제값과 오차를 확인할 수 있습니다.';}
 const margin=p.at(-1)*R.bands[key];$('band').textContent=`검증 기반 경험적 오차 범위: ${fmt(Math.max(0,q-margin))} ~ ${fmt(q+margin)}원 · 기준일 대비 ${pct((q/p.at(-1)-1)*100)}`;$('inputStatus').className='';$('inputStatus').textContent=`${p.length}거래일 입력 · ${key} · 모델 학습 종료: 2025년`; }catch(e){$('forecast').textContent='—';$('forecastActual').textContent='—';$('forecastError').textContent='입력을 확인해 주세요.';$('band').textContent='입력을 확인해 주세요.';$('inputStatus').className='error';$('inputStatus').textContent=e.message;}}
function draw(){if(!R)return;const key=$('chartModel').value,canvas=$('chart'),ctx=canvas.getContext('2d'),W=canvas.clientWidth,H=300,dpr=window.devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;ctx.scale(dpr,dpr);const series=[R.actual,R.predictions[key],R.predictions.naive],all=series.flat(),lo=Math.min(...all)*.97,hi=Math.max(...all)*1.03,left=68,right=14,top=14,bottom=35;ctx.font='11px sans-serif';ctx.fillStyle='#92a3bc';for(let i=0;i<=4;i++){const yy=top+(H-top-bottom)*i/4;ctx.strokeStyle='#253044';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(W-right,yy);ctx.stroke();ctx.fillText(fmt(hi-(hi-lo)*i/4),2,yy+4);}series.forEach((data,j)=>{ctx.strokeStyle=['#72e6c6','#9daeff','#8490a0'][j];ctx.lineWidth=j===2?1:2;ctx.beginPath();data.forEach((v,i)=>{const x=left+(W-left-right)*i/Math.max(1,data.length-1),y=top+(hi-v)/(hi-lo)*(H-top-bottom);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();});ctx.fillStyle='#92a3bc';[0,Math.floor((R.test_dates.length-1)/2),R.test_dates.length-1].forEach((i,j)=>ctx.fillText(R.test_dates[i],left+(W-left-right)*j/2-(j?65:0),H-8));const row=R.rows.find(r=>r.id===key);$('chartStats').textContent=`${key} · MAE ${fmt(row.mae)}원 · MAPE ${pct(row.mape)} · 방향 일치율 ${pct(row.direction)}`;}
async function init(){try{[R,M]=await Promise.all(['results.json','models.json'].map(async p=>{const r=await fetch(p);if(!r.ok)throw Error(`데이터 로드 실패: ${r.status}`);return r.json()}));const closeResponse=await fetch('close.csv');if(!closeResponse.ok)throw Error('종가 이력을 불러올 수 없습니다.');P=(await closeResponse.text()).trim().split(/\r?\n/).slice(1).map(l=>{const [date,close]=l.split(',');return {date,close:Number(close)}});const winner=R.rows.find(r=>r.id===R.selected),base=R.rows[0];$('mae').textContent=fmt(winner.mae)+'원';$('winner').textContent='검증 선택: '+R.selected;$('improve').textContent=pct((base.mae-winner.mae)/base.mae*100);$('count').textContent=R.test_dates.length+'일';$('period').textContent=R.test_dates[0]+' ~ '+R.test_dates.at(-1);$('status').textContent=`실제 데이터: ${R.data_start} ~ ${R.data_end} · 2026년은 확보된 거래일까지 평가`;
for(const row of R.rows){const label=row.id==='naive'?'기준일 종가 기준':`${row.kind} · ${row.window}거래일`;for(const id of ['model','chartModel']){const opt=document.createElement('option');opt.value=row.id;opt.textContent=label+(row.id===R.selected?' (검증 선택)':'');$(id).append(opt);}const tr=document.createElement('tr');if(row.id===R.selected)tr.className='chosen';[label,fmt(row.validation_mae),fmt(row.mae),fmt(row.rmse),pct(row.mape),pct(row.direction)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});$('rows').append(tr);}
for(const row of R.rows.filter(r=>r.id!=='naive')){const opt=document.createElement('option');opt.value=row.id;opt.textContent=`모델 예측 종가 · ${row.kind} · ${row.window}거래일`;$('randomModel').append(opt);}
$('randomModel').value=R.selected100;$('randomModel').onchange=renderRandom;$('random').onclick=randomCompare;$('random').disabled=false;$('randomRange').textContent=`선택 가능: ${R.test_dates[0]} ~ ${R.test_dates.at(-1)} · 실제 종가가 확보된 ${R.test_dates.length}거래일. 클릭할 때마다 다른 날짜를 선택합니다.`;
$('model').value=R.selected100;$('chartModel').value=R.selected;$('model').onchange=()=>{if(inputContext&&$('model').value!=='naive'){$('randomModel').value=$('model').value;renderRandom();}else{fillRecent();}};$('prices').oninput=()=>{inputContext=null;$('setupStatus').textContent='직접 입력한 종가 데이터입니다.';};$('predict').onclick=predict;$('chartModel').onchange=draw;$('upload').onclick=()=>$('file').click();$('file').onchange=async()=>{if($('file').files[0]){inputContext=null;$('setupStatus').textContent='불러온 파일의 종가 데이터로 설정되었습니다.';$('prices').value=await $('file').files[0].text();predict()}};
$('conclusion').textContent=`검증으로 선택한 모델은 ${R.selected}입니다. 100거래일 후보 중 검증 MAE가 작은 모델은 ${R.selected100}입니다. 2026년 기준 모델 MAE는 ${fmt(base.mae)}원이며, 선택 모델은 ${fmt(winner.mae)}원입니다. ${winner.mae<base.mae?'선택 모델의 오차가 기준보다 작았습니다.':'선택 모델이 기준 모델보다 낮은 오차를 달성하지 못했습니다.'} 테스트 결과가 가장 좋은 모델을 사후 선택하면 과대평가될 수 있습니다.`;
$('provenance').textContent=`데이터: ${R.source}. ${R.observations}개 종가. 기준일 ${R.as_of}의 당일 봉은 제외했습니다. 실제 마지막 데이터는 ${R.data_end}입니다. SHA-256: ${R.data_sha256}`;$('generated').textContent='생성: '+R.generated_at;const header=['target_date','origin_date','actual',...R.rows.map(r=>r.id)];const csv=[header.join(','),...R.test_dates.map((d,i)=>[d,R.origin_dates[i],R.actual[i],...R.rows.map(r=>R.predictions[r.id][i])].join(','))].join('\n');$('download').href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));fillRecent();draw();window.addEventListener('resize',draw);
}catch(e){$('status').className='error';$('status').textContent=e.message;}}
init();
