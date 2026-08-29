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
MIN_OBS=55
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
        if m and 2016 <= int(m.group(1)) <= 2025:
            out.append(a)
        if re.fullmatch(r"weekly_2026_W\d{2}\.zip",n):
            out.append(a)
    return sorted(out,key=lambda x:x["name"])

def norm_cols(df):
    df.columns=[str(c).strip().lower() for c in df.columns]
    aliases={
      "date":["date","日期"],"code":["code","stock_id","證券代號","代號"],
      "close":["close","收盤價"]
    }
    ren={}
    for k,opts in aliases.items():
        for o in opts:
            if o.lower() in df.columns:
                ren[o.lower()]=k
                break
    df=df.rename(columns=ren)
    if not {"date","code","close"}.issubset(df.columns):
        return None
    df["code"]=df["code"].astype(str).str.extract(r"(\d{4})",expand=False)
    df=df[df["code"].map(lambda x: bool(ORD.fullmatch(x)) if isinstance(x,str) else False)]
    s=df["date"].astype(str).str.replace(r"\D","",regex=True)
    df["date"]=pd.to_datetime(s.str[:8],format="%Y%m%d",errors="coerce")
    df["close"]=pd.to_numeric(df["close"].astype(str).str.replace(",","",regex=False),errors="coerce")
    return df[["date","code","close"]].dropna()

def load_zip(url):
    b=requests.get(url,timeout=180)
    b.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(b.content))
    parts=[]
    for n in z.namelist():
        if not n.lower().endswith(".csv"):
            continue
        raw=z.read(n)
        df=None
        for enc in ("utf-8-sig","utf-8","cp950","big5"):
            try:
                df=pd.read_csv(io.BytesIO(raw),encoding=enc)
                break
            except Exception:
                pass
        if df is not None:
            x=norm_cols(df)
            if x is not None and len(x):
                parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["date","code","close"])

def load_all():
    parts=[]
    assets=wanted_assets(release_assets())
    print("assets",len(assets),flush=True)
    for i,a in enumerate(assets,1):
        print(i,a["name"],flush=True)
        parts.append(load_zip(a["browser_download_url"]))
    d=pd.concat(parts,ignore_index=True)
    d=d[(d.date>=START)&(d.date<=END)].sort_values(["date","code"])
    d=d.drop_duplicates(["date","code"],keep="last")
    return d

def build_monthly_candidate_schedule(px, ret):
    """
    True walk-forward candidate generation:
    At each month-end t, candidates are determined using ONLY the prior FORM returns
    up to t. These candidates are then eligible only for the NEXT calendar month.
    """
    month_last = pd.Series(np.arange(len(px)), index=px.index).groupby(px.index.to_period("M")).last()
    schedule={}
    for period, t in month_last.items():
        if t < FORM:
            continue
        w = ret.iloc[t-FORM+1:t+1]
        good = w.notna().sum() >= MIN_OBS
        cols = list(w.columns[good])
        pairs=set()
        if len(cols)>=2:
            corr = w[cols].corr(min_periods=MIN_OBS).to_numpy()
            ii,jj=np.where(np.triu(corr,1) >= CORR_MIN)
            pairs={(cols[i],cols[j]) for i,j in zip(ii,jj)}
        next_period = period + 1
        schedule[next_period]=pairs
        print("schedule", str(next_period), "pairs", len(pairs), flush=True)
    return schedule

def pair_metrics(px, ret, a, b):
    ratio=(px[a]/px[b])
    corr=ret[a].rolling(FORM,min_periods=MIN_OBS).corr(ret[b])
    mean=ratio.shift(1).rolling(FORM,min_periods=MIN_OBS).mean()
    std=ratio.shift(1).rolling(FORM,min_periods=MIN_OBS).std(ddof=1)
    dev=ratio/mean-1
    z=(ratio-mean)/std
    return ratio,corr,mean,std,dev,z

def eval_signal(dates, ratio, base_mean, entry_dev, sign, t, horizon):
    end=min(len(dates)-1,t+horizon)
    rec={.03:None,.02:None,.01:None,0.0:None}
    mae=0.0
    mfe=0.0
    for j in range(t+1,end+1):
        rj=ratio[j]
        if not np.isfinite(rj) or not np.isfinite(base_mean) or base_mean==0:
            continue
        dj=rj/base_mean-1
        adverse=max(0.0, sign*dj - sign*entry_dev)
        favorable=max(0.0, abs(entry_dev)-abs(dj))
        mae=max(mae,adverse)
        mfe=max(mfe,favorable)
        for th in rec:
            if rec[th] is not None:
                continue
            if th==0.0:
                if sign*dj <= 0:
                    rec[th]=j-t
            else:
                if abs(dj) <= th:
                    rec[th]=j-t
    return rec,mae,mfe,end

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-stocks",type=int,default=0)
    args=ap.parse_args()

    d=load_all()
    px=d.pivot(index="date",columns="code",values="close").sort_index()
    valid=px.notna().sum()
    codes=valid[valid>=FORM+HORIZON+20].sort_values(ascending=False).index.tolist()
    if args.max_stocks:
        codes=codes[:args.max_stocks]
    px=px[codes]
    ret=px.pct_change(fill_method=None)
    dates=px.index.to_numpy()

    schedule=build_monthly_candidate_schedule(px,ret)

    # Cache metrics per pair, computed lazily only if a pair becomes eligible.
    cache={}
    rows=[]
    active_until={}  # non-overlap per pair
    start_ts=pd.Timestamp(START)

    for t,dt in enumerate(px.index):
        if dt < start_ts or t < FORM:
            continue
        period=dt.to_period("M")
        pairs=schedule.get(period,set())
        if not pairs:
            continue

        for a,b in pairs:
            key=(a,b)
            if active_until.get(key,-1) >= t:
                continue

            if key not in cache:
                ratio,corr,mean,std,dev,z=pair_metrics(px,ret,a,b)
                cache[key]=(
                    ratio.to_numpy(),
                    corr.to_numpy(),
                    mean.to_numpy(),
                    std.to_numpy(),
                    dev.to_numpy(),
                    z.to_numpy()
                )
            ratio,corr,mean,std,dev,z=cache[key]

            if not np.isfinite(corr[t]) or corr[t] < CORR_MIN:
                continue
            if not np.isfinite(dev[t]) or abs(dev[t]) < ENTRY:
                continue

            # First crossing only, relative to immediately previous trading day.
            if t>0 and np.isfinite(dev[t-1]) and abs(dev[t-1]) >= ENTRY:
                continue

            sign=1 if dev[t]>0 else -1
            rec,mae,mfe,end_idx=eval_signal(dates,ratio,mean[t],dev[t],sign,t,HORIZON)
            active_until[key]=end_idx

            rows.append({
                "date":str(dt.date()),
                "year":int(dt.year),
                "month":int(dt.month),
                "a":a,"b":b,
                "corr":float(corr[t]),
                "entry_dev":float(dev[t]),
                "abs_entry_dev":float(abs(dev[t])),
                "direction":"positive" if sign>0 else "negative",
                "z":float(z[t]) if np.isfinite(z[t]) else None,
                "mae":float(mae),
                "mfe":float(mfe),
                "d3":rec[.03],"d2":rec[.02],"d1":rec[.01],"d0":rec[0.0]
            })

    ev=pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    ev.to_csv("data/validation_events_walkforward.csv",index=False)

    summary={
        "period":[START,END],
        "formation_days":FORM,
        "corr_min":CORR_MIN,
        "entry":ENTRY,
        "horizon":HORIZON,
        "candidate_selection":"month-end prior 60d correlation; eligible next calendar month only",
        "stocks":len(codes),
        "events":len(ev),
        "unique_pairs":int(ev[["a","b"]].drop_duplicates().shape[0]) if len(ev) else 0
    }

    if len(ev):
        for th,col in [("3pct","d3"),("2pct","d2"),("1pct","d1"),("mean","d0")]:
            for h in (5,10,20,30):
                summary[f"conv_{th}_{h}d"]=float((ev[col].fillna(999)<=h).mean())
        summary["avg_mae"]=float(ev.mae.mean())
        summary["median_mae"]=float(ev.mae.median())
        summary["median_abs_z"]=float(ev.z.abs().median())
        summary["year_counts"]={str(k):int(v) for k,v in ev.groupby("year").size().items()}
        summary["direction_counts"]={str(k):int(v) for k,v in ev.groupby("direction").size().items()}

    Path("data/validation_walkforward.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
