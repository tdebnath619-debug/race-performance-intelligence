"""telemetry/delta.py — Distance-aligned lap delta analysis."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
GRID = 5000
MAX_MISMATCH = 200.0

@dataclass
class CornerDelta:
    corner_id: int; corner_type: str; apex_dist_m: float
    time_delta_s: float; entry_speed_delta: float; min_speed_delta: float
    exit_speed_delta: float; brake_point_delta_m: float; throttle_point_delta_m: float
    ers_delta_kj: float = 0.0; lateral_g_delta: float = 0.0
    primary_cause: str = ""; narrative: str = ""
    def __post_init__(self):
        if not self.primary_cause: self.primary_cause, self.narrative = _narrative(self)
    def to_dict(self): return asdict(self)
    def __repr__(self):
        return (f"T{self.corner_id} [{self.corner_type}]  Δt={self.time_delta_s:+.3f}s  "
                f"ΔvEntry={self.entry_speed_delta:+.1f}  ΔvApex={self.min_speed_delta:+.1f}  "
                f"ΔvExit={self.exit_speed_delta:+.1f} km/h")

@dataclass
class ComparisonReport:
    driver_a: str; driver_b: str; lap_a: int; lap_b: int; session: str
    total_delta_s: float
    corner_deltas: list = field(default_factory=list)
    key_findings: list  = field(default_factory=list)
    unmatched_a: list   = field(default_factory=list)
    unmatched_b: list   = field(default_factory=list)
    _common_dist: object = field(default=None, repr=False)
    _cum_delta: object   = field(default=None, repr=False)

    def summary(self):
        faster = self.driver_a if self.total_delta_s>0 else self.driver_b
        return (f"{faster} faster by {abs(self.total_delta_s):.3f} s  |  "
                f"{len(self.corner_deltas)} corners  |  {self.session}")

    def to_dict(self):
        d = asdict(self)
        d.pop("_common_dist",None); d.pop("_cum_delta",None)
        d["corner_deltas"] = [c.to_dict() for c in self.corner_deltas]
        return d

    def to_json(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        class E(json.JSONEncoder):
            def default(self,o):
                if isinstance(o,(np.integer,)): return int(o)
                if isinstance(o,(np.floating,)): return float(o)
                if isinstance(o,np.ndarray): return o.tolist()
                if isinstance(o,(np.bool_,)): return bool(o)
                return super().default(o)
        path.write_text(json.dumps(self.to_dict(),indent=2,cls=E))
        log.info("Delta report → %s", path); return path

    def print_report(self):
        sep = "═"*72
        print(f"\n{sep}")
        print(f"  DELTA: {self.driver_a} (A) vs {self.driver_b} (B)")
        print(f"  {self.summary()}")
        print(f"{sep}")
        if self.key_findings:
            print("\n  KEY FINDINGS"); print("  "+"─"*60)
            for f in self.key_findings: print(f"  • {f}")
        print("\n  CORNER-BY-CORNER"); print("  "+"─"*60)
        print(f"  {'T':<4} {'Type':<8} {'Δt(s)':>8} {'ΔvEntry':>9} {'ΔvApex':>9} {'ΔvExit':>9}")
        print("  "+"─"*60)
        for cd in self.corner_deltas:
            print(f"  T{cd.corner_id:<3} {cd.corner_type:<8} {cd.time_delta_s:>+8.3f} "
                  f"{cd.entry_speed_delta:>+9.1f} {cd.min_speed_delta:>+9.1f} {cd.exit_speed_delta:>+9.1f}")
            print(f"          → {cd.narrative}")
        if self.unmatched_a: print(f"\n  Unmatched in {self.driver_a}: {self.unmatched_a}")
        if self.unmatched_b: print(f"\n  Unmatched in {self.driver_b}: {self.unmatched_b}")
        print(sep+"\n")

def compare(lap_a, lap_b, seg_a, seg_b):
    df_a, df_b = lap_a.data, lap_b.data
    cd, sa, sb = _align(df_a, df_b)
    cum = _cum_delta(df_a, df_b, cd)
    total = round(float(cum[-1]), 4)
    matched, um_a, um_b = _match(seg_a.corners, seg_b.corners)
    deltas = []
    for ca,cb in matched:
        d = _corner_delta(df_a,df_b,ca,cb,cd,cum)
        deltas.append(d); log.info("%s  →  %s", d, d.narrative)
    findings = _findings(deltas, total, lap_a.driver, lap_b.driver)
    report = ComparisonReport(driver_a=lap_a.driver, driver_b=lap_b.driver,
        lap_a=lap_a.lap_number, lap_b=lap_b.lap_number, session=lap_a.session,
        total_delta_s=total, corner_deltas=deltas, key_findings=findings,
        unmatched_a=[c.corner_id for c in um_a], unmatched_b=[c.corner_id for c in um_b],
        _common_dist=cd, _cum_delta=cum)
    log.info("Delta complete: %s", report.summary())
    return report

def _align(df_a, df_b, n=GRID):
    d_min=max(df_a["distance"].min(),df_b["distance"].min())
    d_max=min(df_a["distance"].max(),df_b["distance"].max())
    if d_max<=d_min: raise ValueError("No overlapping distance range.")
    cd=np.linspace(d_min,d_max,n)
    def _i(df):
        return {col:np.interp(cd,df["distance"].values,df[col].values)
                for col in ("speed","throttle","brake","time","ers_deployment_kw","lateral_g") if col in df.columns}
    return cd, _i(df_a), _i(df_b)

def _cum_delta(df_a, df_b, cd):
    va=np.interp(cd,df_a["distance"].values,df_a["speed"].values)/3.6
    vb=np.interp(cd,df_b["distance"].values,df_b["speed"].values)/3.6
    va=np.where(va>0,va,np.nan); vb=np.where(vb>0,vb,np.nan)
    dd=np.diff(cd,prepend=cd[0])
    return np.nancumsum(np.where(~np.isnan(vb),dd/vb,0)-np.where(~np.isnan(va),dd/va,0))

def _match(ca_list, cb_list):
    matched,used,um_a=[],set(),[]
    for ca in ca_list:
        best,bd=None,np.inf
        for cb in cb_list:
            if id(cb) in used: continue
            d=abs(ca.apex_dist-cb.apex_dist)
            if d<bd and d<MAX_MISMATCH: bd,best=d,cb
        if best: matched.append((ca,best)); used.add(id(best))
        else: um_a.append(ca)
    um_b=[c for c in cb_list if id(c) not in used]
    return matched,um_a,um_b

def _spd(df,idx): return round(float(df.iloc[max(0,min(idx,len(df)-1))]["speed"]),2)
def _ers(df,i0,i1):
    if "ers_deployment_kw" not in df.columns: return 0.0
    seg=df.iloc[i0:i1]; return float((seg["ers_deployment_kw"]*seg["time"].diff().fillna(0)).sum())/1000.0
def _latg(df,i0,i1):
    if "lateral_g" not in df.columns: return 0.0
    return float(df.iloc[i0:i1]["lateral_g"].abs().max())

def _corner_delta(df_a,df_b,ca,cb,cd,cum):
    exit_d=(ca.exit_dist+cb.exit_dist)/2
    dt_val=float(np.interp(exit_d,cd,cum))
    return CornerDelta(
        corner_id=ca.corner_id, corner_type=ca.corner_type,
        apex_dist_m=round((ca.apex_dist+cb.apex_dist)/2,1),
        time_delta_s=round(dt_val,4),
        entry_speed_delta=round(_spd(df_a,ca.entry_idx)-_spd(df_b,cb.entry_idx),2),
        min_speed_delta=round(_spd(df_a,ca.apex_idx)-_spd(df_b,cb.apex_idx),2),
        exit_speed_delta=round(_spd(df_a,ca.exit_idx)-_spd(df_b,cb.exit_idx),2),
        brake_point_delta_m=round(ca.entry_dist-cb.entry_dist,1),
        throttle_point_delta_m=round(ca.apex_dist-cb.apex_dist,1),
        ers_delta_kj=round(_ers(df_a,ca.entry_idx,ca.exit_idx)-_ers(df_b,cb.entry_idx,cb.exit_idx),3),
        lateral_g_delta=round(_latg(df_a,ca.entry_idx,ca.exit_idx)-_latg(df_b,cb.entry_idx,cb.exit_idx),3))

def _narrative(cd):
    faster="A" if cd.time_delta_s>0 else "B"
    if abs(cd.time_delta_s)<0.005: return "none","No significant delta."
    if abs(cd.entry_speed_delta)>=5:
        high="A" if cd.entry_speed_delta>0 else "B"
        return "braking_zone",(f"Driver {high} carries {abs(cd.entry_speed_delta):.1f} km/h more entry speed "
               f"(brakes {abs(cd.brake_point_delta_m):.0f} m later). Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.min_speed_delta)>=3:
        high="A" if cd.min_speed_delta>0 else "B"
        return "corner_speed",(f"Driver {high} achieves {abs(cd.min_speed_delta):.1f} km/h higher minimum speed. "
               f"Likely: mechanical grip, aero balance, or line. Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.throttle_point_delta_m)>=10:
        early="A" if cd.throttle_point_delta_m>0 else "B"
        return "throttle_application",(f"Driver {early} picks up throttle {abs(cd.throttle_point_delta_m):.0f} m earlier. "
               f"Suggests higher traction confidence. Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.exit_speed_delta)>=5:
        high="A" if cd.exit_speed_delta>0 else "B"
        return "corner_exit",(f"Driver {high} exits {abs(cd.exit_speed_delta):.1f} km/h faster. "
               f"Check ERS deployment and rear downforce. Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    return "unclassified",(f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s. No single dominant cause.")

def _findings(deltas,total,da,db):
    if not deltas: return []
    findings=[]; faster=da if total>0 else db
    worst=max(deltas,key=lambda d:abs(d.time_delta_s))
    findings.append(f"Largest delta: T{worst.corner_id} ({worst.corner_type}) — {abs(worst.time_delta_s):.3f} s — {worst.primary_cause.replace('_',' ')}.")
    late=[d for d in deltas if d.brake_point_delta_m<-10]
    if late: findings.append(f"{da} brakes later at {len(late)} corner(s): {[d.corner_id for d in late]}.")
    early=[d for d in deltas if d.throttle_point_delta_m>10]
    if early: findings.append(f"{da} picks up throttle earlier at {len(early)} corner(s): {[d.corner_id for d in early]}.")
    apex=[d for d in deltas if d.min_speed_delta<-3]
    if apex: findings.append(f"{da} has lower apex speed at {len(apex)} corner(s) — check mechanical balance.")
    return findings
