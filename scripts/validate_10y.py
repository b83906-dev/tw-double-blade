import argparse, io, json, math, re, zipfile, requests
from pathlib import Path
import numpy as np
import pandas as pd

OWNER_REPO="yukishirotsubasa/tw-stock-data-release"
TAG="daily-close-csv"
START="2016-08-01"
END="2026-08-31"
FORM=60
CORR_MIN=.90
ENTRY=.05
HORIZON=30
ORD=re.compile(r"^[1-9][0-9]{3}$")

def release_assets():
    r=requests.get(f"https://api.github.com/repos/{OWNER_REPO}/releases/tags/{TAG}",timeout=60)
    r.raise_for_status()
    return r.json()["assets"]

def wanted_assets(assets):
    out=[]
    for a in assets:
        n=a["name"]
        m=re.fullmatch(r"yearly_(\d{4})\.zip",n)
        if m and 2016 <= int(m.group(1)) <= 2025: out.append(a)
        if re.fullmatch(r"weekly_2026_W\d{2}\.zip",n): out.append(a)
    return sorted(out,key=lambda x:x["name"])

def norm_cols(df):
    df.columns=[str(c).strip().lower() for c in df.columns]
    aliases={
      "date":["date","日期"],"code":["code","stock_id","證券代號","代號"],
      "close":["close","收盤價"],"volume":["volume","成交股數","成交量"]
    }
    ren={}
    for k,opts in aliases.items():
        for o in opts:
            if o.lower() in df.columns: ren[o.lower()]=k; break
    df=df.rename(columns=ren)
    if not {"date","code","close"}.issubset(df.columns): return None
    df["code"]=df["code"].astype(str).str.extract(r"(\d{4})",expand=False)
    df=df[df["code"].map(lambda x: bool(ORD.fullmatch(x)) if isinstance(x,str) else False)]
    s=df["date"].astype(str).str.replace(r"\D","",regex=True)
    df["date"]=pd.to_datetime(s.str[:8],format="%Y%m%d",errors="coerce")
    df["close"]=pd.to_numeric(df["close"].astype(str).str.replace(",","",regex=False),errors="coerce")
    return df[["date","code","close"]].dropna()

def load_zip(url):
    b=requests.get(url,timeout=180); b.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(b.content)); parts=[]
    for n in z.namelist():
        if n.lower().endswith(".csv"):
            raw=z.read(n)
            df=None
            for enc in ("utf-8-sig","utf-8","cp950","big5"):
                try: df=pd.read_csv(io.BytesIO(raw),encoding=enc); break
                except Exception: pass
            if df is not None:
                x=norm_cols(df)
                if x is not None and len(x): parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["date","code","close"])

def load_all():
    parts=[]
    assets=wanted_assets(release_assets())
    print("assets",len(assets))
    for i,a in enumerate(assets,1):
        print(i,a["name"],flush=True)
        parts.append(load_zip(a["browser_download_url"]))
    d=pd.concat(parts,ignore_index=True)
    d=d[(d.date>=START)&(d.date<=END)].sort_values(["date","code"])
    d=d.drop_duplicates(["date","code"],keep="last")
    return d

def evaluate_pair(dates, a, b, corr, mean_ratio, std_ratio, start_idx):
    ratio=a/b
    dev=ratio/mean_ratio-1
    z=(ratio-mean_ratio)/std_ratio
    events=[]
    active_until=-1
    for t in range(max(FORM,start_idx),len(dates)-1):
        if t<=active_until or not np.isfinite(corr[t]) or corr[t]<CORR_MIN: continue
        if not np.isfinite(dev[t]) or abs(dev[t])<ENTRY: continue
        if t>0 and np.isfinite(dev[t-1]) and abs(dev[t-1])>=ENTRY: continue
        sign=1 if dev[t]>0 else -1
        base=mean_ratio[t]
        end=min(len(dates)-1,t+HORIZON)
        rec={.03:None,.02:None,.01:None,0.0:None}
        mae=0.0
        for j in range(t+1,end+1):
            dj=ratio[j]/base-1 if np.isfinite(ratio[j]) and base else np.nan
            if not np.isfinite(dj): continue
            adverse=max(0, sign*dj-sign*dev[t])
            mae=max(mae,adverse)
            for th in rec:
                if rec[th] is None:
                    if th==0.0:
                        if sign*dj<=0: rec[th]=j-t
                    elif abs(dj)<=th: rec[th]=j-t
        active_until=end
        events.append((t,dev[t],z[t],corr[t],mae,rec))
    return events

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-stocks",type=int,default=0,help="0=all; useful for smoke test")
    args=ap.parse_args()
    d=load_all()
    px=d.pivot(index="date",columns="code",values="close").sort_index()
    valid=px.notna().sum()
    codes=valid[valid>=FORM+HORIZON+20].sort_values(ascending=False).index.tolist()
    if args.max_stocks: codes=codes[:args.max_stocks]
    px=px[codes]
    dates=px.index.to_numpy()
    ret=px.pct_change(fill_method=None)
    # Candidate discovery monthly: exact daily event monitoring is then performed only for pairs
    # that achieved >=.90 in at least one month-end 60d formation window. This reduces all-market cost.
    month_ends=pd.Series(np.arange(len(px)),index=px.index).groupby(px.index.to_period("M")).last().to_numpy()
    candidates=set()
    for k,t in enumerate(month_ends):
        if t<FORM: continue
        w=ret.iloc[t-FORM+1:t+1]
        good=w.notna().sum()>=55
        cols=list(w.columns[good])
        if len(cols)<2: continue
        c=w[cols].corr(min_periods=55).to_numpy()
        ii,jj=np.where(np.triu(c,1)>=CORR_MIN)
        candidates.update((cols[i],cols[j]) for i,j in zip(ii,jj))
        if k%12==0: print("candidate month",k,"pairs",len(candidates),flush=True)
    print("candidate pairs",len(candidates))
    # Daily exact stats per candidate.
    rows=[]; start_idx=int(np.searchsorted(dates,np.datetime64(START)))
    for n,(ca,cb) in enumerate(sorted(candidates),1):
        sub=px[[ca,cb]]
        ra=ret[ca]; rb=ret[cb]
        corr=ra.rolling(FORM,min_periods=55).corr(rb).to_numpy()
        ratio=(sub[ca]/sub[cb])
        mean=ratio.shift(1).rolling(FORM,min_periods=55).mean().to_numpy()
        std=ratio.shift(1).rolling(FORM,min_periods=55).std(ddof=1).to_numpy()
        ev=evaluate_pair(dates,sub[ca].to_numpy(),sub[cb].to_numpy(),corr,mean,std,start_idx)
        for t,dev,z,co,mae,rec in ev:
            rows.append({"date":str(pd.Timestamp(dates[t]).date()),"a":ca,"b":cb,
              "corr":co,"entry_dev":dev,"z":z,"mae":mae,
              "d3":rec[.03],"d2":rec[.02],"d1":rec[.01],"d0":rec[0.0]})
        if n%10000==0: print("pairs done",n,"events",len(rows),flush=True)
    ev=pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    ev.to_csv("data/validation_events.csv",index=False)
    summary={"period":[START,END],"formation_days":FORM,"corr_min":CORR_MIN,"entry":ENTRY,
             "horizon":HORIZON,"stocks":len(codes),"candidate_pairs":len(candidates),"events":len(ev)}
    if len(ev):
        for th,col in [("3pct","d3"),("2pct","d2"),("1pct","d1"),("mean","d0")]:
            for h in (5,10,20,30):
                summary[f"conv_{th}_{h}d"]=float((ev[col].fillna(999)<=h).mean())
        summary["avg_mae"]=float(ev.mae.mean())
        summary["median_abs_z"]=float(ev.z.abs().median())
    Path("data/validation_baseline.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
