import argparse, io, json, re, zipfile, requests
from pathlib import Path
import numpy as np, pandas as pd

EVENTS="data/validation_events_walkforward.csv"
OUTCSV="data/disposal_ab_events.csv"
OUTJSON="data/disposal_ab_summary.json"
TWSE_URL="https://www.twse.com.tw/rwd/zh/announcement/punish"
TPEX_URL="https://www.tpex.org.tw/www/zh-tw/announce/market/disposal"

def rocdate(s):
    s=str(s).strip()
    nums=re.findall(r"\d+",s)
    if len(nums)>=3:
        y=int(nums[0]); y=y+1911 if y<1911 else y
        return pd.Timestamp(y,int(nums[1]),int(nums[2]))
    return pd.NaT

def fetch_json(url, params):
    r=requests.get(url,params=params,timeout=60,headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def parse_tables(j, market):
    rows=[]
    tables=j.get("tables") or ([j] if j.get("data") else [])
    for tb in tables:
        fields=tb.get("fields") or []
        data=tb.get("data") or []
        for x in data:
            rec=dict(zip(fields,x))
            txt=" ".join(map(str,x))
            code=None
            for v in x:
                m=re.fullmatch(r"\s*(\d{4})\s*",str(v))
                if m: code=m.group(1); break
            if not code: continue
            # Parse all date-like strings; disposal period usually contains start/end.
            ds=[]
            for v in x:
                for m in re.finditer(r"(\d{2,4})[./年-](\d{1,2})[./月-](\d{1,2})",str(v)):
                    y=int(m.group(1)); y=y+1911 if y<1911 else y
                    try: ds.append(pd.Timestamp(y,int(m.group(2)),int(m.group(3))))
                    except: pass
            if not ds: continue
            rows.append({"code":code,"start":min(ds),"end":max(ds),"market":market,"raw":txt})
    return rows

def fetch_disposals(start="2016-08-01", end="2026-08-31"):
    rows=[]
    # Official historical pages. Query in yearly chunks to reduce response size.
    for y in range(2016,2027):
        s=max(pd.Timestamp(start),pd.Timestamp(y,1,1))
        e=min(pd.Timestamp(end),pd.Timestamp(y,12,31))
        if s>e: continue
        # TWSE
        candidates=[
          (TWSE_URL,{"response":"json","startDate":s.strftime("%Y%m%d"),"endDate":e.strftime("%Y%m%d"),"stockNo":""}),
          (TPEX_URL,{"response":"json","startDate":s.strftime("%Y/%m/%d"),"endDate":e.strftime("%Y/%m/%d"),"code":""}),
        ]
        for market,(url,p) in zip(("TWSE","TPEX"),candidates):
            try:
                j=fetch_json(url,p); got=parse_tables(j,market); rows.extend(got)
                print(y,market,len(got),flush=True)
            except Exception as ex:
                print("WARN",y,market,repr(ex),flush=True)
    d=pd.DataFrame(rows)
    if len(d):
        d=d.drop_duplicates(["code","start","end","market"]).sort_values(["code","start"])
    return d

def add_flags(ev, disp):
    ev=ev.copy()
    ev["date"]=pd.to_datetime(ev["date"])
    by={}
    for code,g in disp.groupby("code"):
        by[code]=list(zip(g.start,g.end))
    for side in ("a","b"):
        cur=[]; pre5=[]; pre10=[]; pre20=[]; days_to=[]
        for code,dt in zip(ev[side].astype(str),ev.date):
            spans=by.get(code,[])
            iscur=any(s<=dt<=e for s,e in spans)
            future=[(s-dt).days for s,e in spans if s>dt]
            nxt=min(future) if future else 99999
            cur.append(iscur); pre5.append(0<nxt<=7); pre10.append(0<nxt<=14); pre20.append(0<nxt<=28); days_to.append(nxt if nxt<99999 else np.nan)
        ev[f"{side}_in_disposal"]=cur
        ev[f"{side}_pre5"]=pre5; ev[f"{side}_pre10"]=pre10; ev[f"{side}_pre20"]=pre20
        ev[f"{side}_days_to_disposal"]=days_to
    ev["any_in_disposal"]=ev.a_in_disposal|ev.b_in_disposal
    ev["any_pre5"]=ev.a_pre5|ev.b_pre5
    ev["any_pre10"]=ev.a_pre10|ev.b_pre10
    ev["any_pre20"]=ev.a_pre20|ev.b_pre20
    ev["success20"]=pd.to_numeric(ev.d2,errors="coerce").le(20)
    return ev

def stat(ev, mask):
    x=ev[mask]
    n=len(x); w=int(x.success20.sum())
    base=float(ev.success20.mean())
    raw=w/n if n else None
    # Beta-binomial shrinkage toward global base, prior strength 100 events.
    shr=(w+100*base)/(n+100) if n else None
    return {"n":n,"wins":w,"raw_success20":raw,"shrunk_confidence":shr,
            "lift_pp":(raw-base)*100 if n else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--events",default=EVENTS); args=ap.parse_args()
    ev=pd.read_csv(args.events,dtype={"a":str,"b":str})
    disp=fetch_disposals()
    Path("data").mkdir(exist_ok=True)
    disp.to_csv("data/disposal_history.csv",index=False)
    z=add_flags(ev,disp); z.to_csv(OUTCSV,index=False)
    groups={
      "baseline":pd.Series(True,index=z.index),
      "in_disposal":z.any_in_disposal,
      "pre5":z.any_pre5,
      "pre10":z.any_pre10,
      "pre20":z.any_pre20,
      "clean_no_pre20_no_current":~(z.any_pre20|z.any_in_disposal),
      "positive_pre20":z.any_pre20 & (z.entry_dev>0),
      "negative_pre20":z.any_pre20 & (z.entry_dev<0),
    }
    out={"disposal_records":len(disp),"groups":{k:stat(z,m) for k,m in groups.items()}}
    Path(OUTJSON).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
