import argparse, json, re, requests
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS="data/validation_events_walkforward.csv"
OUTCSV="data/disposal_ab_events_v2.csv"
OUTJSON="data/disposal_ab_summary_v2.json"

TWSE="https://www.twse.com.tw/rwd/zh/announcement/punish"
TPEX_PAGE="https://www.tpex.org.tw/www/zh-tw/announce/market/disposal"
TPEX_API="https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"
ORD=re.compile(r"^[1-9][0-9]{3}$")

def roc_to_ts(x):
    if x is None: return pd.NaT
    s=str(x).strip()
    m=re.search(r"(\d{2,4})\s*[/.\-年]\s*(\d{1,2})\s*[/.\-月]\s*(\d{1,2})",s)
    if not m: return pd.NaT
    y,mo,d=map(int,m.groups())
    if y<1911: y+=1911
    try: return pd.Timestamp(y,mo,d)
    except: return pd.NaT

def all_dates(x):
    s=str(x)
    out=[]
    for m in re.finditer(r"(\d{2,4})\s*[/.\-年]\s*(\d{1,2})\s*[/.\-月]\s*(\d{1,2})",s):
        y,mo,d=map(int,m.groups())
        if y<1911: y+=1911
        try: out.append(pd.Timestamp(y,mo,d))
        except: pass
    return out

def getj(url,params=None):
    r=requests.get(url,params=params,timeout=60,headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def parse_twse(j):
    rows=[]
    tables=j.get("tables") or ([j] if j.get("data") else [])
    for tb in tables:
        fields=[str(x).strip() for x in (tb.get("fields") or [])]
        for vals in tb.get("data") or []:
            rec=dict(zip(fields,vals))
            code=None
            for k,v in rec.items():
                if "代號" in k:
                    m=re.fullmatch(r"\s*(\d{4})\s*",str(v))
                    if m: code=m.group(1)
            if not code:
                for v in vals:
                    m=re.fullmatch(r"\s*(\d{4})\s*",str(v))
                    if m: code=m.group(1); break
            if not code or not ORD.fullmatch(code): continue
            announce=pd.NaT; start=pd.NaT; end=pd.NaT
            for k,v in rec.items():
                if "公布日期" in k: announce=roc_to_ts(v)
                if "起迄" in k or "處置期間" in k:
                    ds=all_dates(v)
                    if len(ds)>=2: start,end=ds[0],ds[1]
            if pd.isna(start) or pd.isna(end):
                # Search row text, but keep announcement separate.
                ds=all_dates(" ".join(map(str,vals)))
                ds=[d for d in ds if pd.isna(announce) or d!=announce]
                if len(ds)>=2: start,end=ds[0],ds[1]
            if pd.notna(start) and pd.notna(end):
                rows.append({"code":code,"announce":announce,"start":start,"end":end,"market":"TWSE"})
    return rows

def fetch_twse(start,end):
    rows=[]
    for y in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
        s=max(pd.Timestamp(start),pd.Timestamp(y,1,1)); e=min(pd.Timestamp(end),pd.Timestamp(y,12,31))
        p={"response":"json","startDate":s.strftime("%Y%m%d"),"endDate":e.strftime("%Y%m%d"),
           "stockNo":"","selectType":"","proceType":"","remarkType":"","sortKind":"DATE"}
        try:
            got=parse_twse(getj(TWSE,p)); rows+=got; print(y,"TWSE",len(got),flush=True)
        except Exception as ex: print("WARN TWSE",y,repr(ex),flush=True)
    return rows

def normalize_tpex_item(rec):
    # OpenAPI field names can be Chinese or English; inspect values robustly.
    code=None
    for k,v in rec.items():
        ks=str(k).lower()
        if "代號" in str(k) or "code" in ks:
            m=re.fullmatch(r"\s*(\d{4})\s*",str(v))
            if m: code=m.group(1); break
    if not code:
        for v in rec.values():
            m=re.fullmatch(r"\s*(\d{4})\s*",str(v))
            if m: code=m.group(1); break
    if not code or not ORD.fullmatch(code): return None
    announce=start=end=pd.NaT
    for k,v in rec.items():
        ks=str(k)
        if "公布" in ks and "日" in ks: announce=roc_to_ts(v)
        if ("開始" in ks or "起始" in ks) and "日" in ks: start=roc_to_ts(v)
        if ("結束" in ks or "截止" in ks or "迄" in ks) and "日" in ks: end=roc_to_ts(v)
        if ("處置起迄" in ks or "處置期間" in ks):
            ds=all_dates(v)
            if len(ds)>=2: start,end=ds[0],ds[1]
    if pd.isna(start) or pd.isna(end):
        ds=[]
        for v in rec.values(): ds += all_dates(v)
        if pd.notna(announce): ds=[d for d in ds if d!=announce]
        if len(ds)>=2: start,end=ds[0],ds[1]
    if pd.isna(start) or pd.isna(end): return None
    return {"code":code,"announce":announce,"start":start,"end":end,"market":"TPEX"}

def fetch_tpex(start,end):
    rows=[]
    # 1) Official OpenAPI. It may expose a broad/current set depending on server version.
    try:
        j=getj(TPEX_API)
        items=j if isinstance(j,list) else j.get("data",[])
        for rec in items:
            x=normalize_tpex_item(rec)
            if x and x["end"]>=pd.Timestamp(start) and x["start"]<=pd.Timestamp(end): rows.append(x)
        print("TPEX OpenAPI",len(rows),flush=True)
    except Exception as ex: print("WARN TPEX API",repr(ex),flush=True)

    # 2) Official historical page, yearly chunks. This is the authoritative backfill path.
    for y in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
        s=max(pd.Timestamp(start),pd.Timestamp(y,1,1)); e=min(pd.Timestamp(end),pd.Timestamp(y,12,31))
        param_sets=[
          {"response":"json","startDate":s.strftime("%Y/%m/%d"),"endDate":e.strftime("%Y/%m/%d"),"code":"","sort":"date"},
          {"response":"json","startDate":s.strftime("%Y%m%d"),"endDate":e.strftime("%Y%m%d"),"code":""},
          {"response":"json","dateStart":s.strftime("%Y/%m/%d"),"dateEnd":e.strftime("%Y/%m/%d"),"code":""},
        ]
        best=[]
        for p in param_sets:
            try:
                j=getj(TPEX_PAGE,p)
                items=[]
                if isinstance(j,list): items=j
                elif isinstance(j,dict):
                    if isinstance(j.get("data"),list) and j["data"] and isinstance(j["data"][0],dict):
                        items=j["data"]
                    else:
                        for tb in j.get("tables",[]) or []:
                            fs=tb.get("fields",[]); items += [dict(zip(fs,x)) for x in tb.get("data",[]) or []]
                got=[normalize_tpex_item(r) for r in items]
                got=[x for x in got if x]
                if len(got)>len(best): best=got
            except: pass
        rows+=best; print(y,"TPEX history",len(best),flush=True)
    return rows

def add_flags(ev,disp):
    ev=ev.copy(); ev["date"]=pd.to_datetime(ev.date)
    by={c:list(zip(g.announce,g.start,g.end)) for c,g in disp.groupby("code")}
    for side in ("a","b"):
        cur=[]; before_ann5=[]; before_ann10=[]; before_ann20=[]; after_ann_before_start=[]
        for code,dt in zip(ev[side].astype(str),ev.date):
            spans=by.get(code,[])
            cur.append(any(s<=dt<=e for a,s,e in spans))
            future_ann=sorted([(a-dt).days for a,s,e in spans if pd.notna(a) and a>dt])
            n=future_ann[0] if future_ann else 99999
            before_ann5.append(0<n<=7); before_ann10.append(0<n<=14); before_ann20.append(0<n<=28)
            after_ann_before_start.append(any(pd.notna(a) and a<=dt<s for a,s,e in spans))
        ev[f"{side}_in_disposal"]=cur
        ev[f"{side}_pre_announce5"]=before_ann5
        ev[f"{side}_pre_announce10"]=before_ann10
        ev[f"{side}_pre_announce20"]=before_ann20
        ev[f"{side}_announced_not_started"]=after_ann_before_start
    for x in ("in_disposal","pre_announce5","pre_announce10","pre_announce20","announced_not_started"):
        ev["any_"+x]=ev["a_"+x]|ev["b_"+x]
    ev["success20"]=pd.to_numeric(ev.d2,errors="coerce").le(20)
    return ev

def stat(z,m):
    x=z[m]; n=len(x); w=int(x.success20.sum()); base=float(z.success20.mean())
    raw=w/n if n else None; shr=(w+100*base)/(n+100) if n else None
    return {"n":n,"wins":w,"raw_success20":raw,"shrunk_confidence":shr,
            "lift_pp":(raw-base)*100 if n else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--events",default=EVENTS); args=ap.parse_args()
    ev=pd.read_csv(args.events,dtype={"a":str,"b":str})
    start=str(pd.to_datetime(ev.date).min().date()); end="2026-08-31"
    disp=pd.DataFrame(fetch_twse(start,end)+fetch_tpex(start,end))
    if len(disp):
        disp=disp.drop_duplicates(["code","announce","start","end","market"]).sort_values(["code","start"])
    Path("data").mkdir(exist_ok=True); disp.to_csv("data/disposal_history_v2.csv",index=False)
    z=add_flags(ev,disp); z.to_csv(OUTCSV,index=False)
    groups={
      "baseline":pd.Series(True,index=z.index),
      "in_disposal":z.any_in_disposal,
      "announced_not_started":z.any_announced_not_started,
      "pre_announce5":z.any_pre_announce5,
      "pre_announce10":z.any_pre_announce10,
      "pre_announce20":z.any_pre_announce20,
      "positive_pre20":z.any_pre_announce20 & (z.entry_dev>0),
      "negative_pre20":z.any_pre_announce20 & (z.entry_dev<0),
      "clean":~(z.any_pre_announce20|z.any_announced_not_started|z.any_in_disposal),
    }
    out={"records":len(disp),
         "records_by_market":disp.groupby("market").size().to_dict() if len(disp) else {},
         "groups":{k:stat(z,m) for k,m in groups.items()}}
    Path(OUTJSON).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
