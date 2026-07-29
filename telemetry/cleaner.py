"""telemetry/cleaner.py — Signal processing pipeline."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CLIP = {"speed":(0,380),"throttle":(0,1),"brake":(0,1),"steering":(-1,1),
         "gear":(0,9),"rpm":(0,20000),"lateral_g":(-7,7),"longitudinal_g":(-7,5),
         "tyre_temp_fl":(0,200),"tyre_temp_fr":(0,200),"tyre_temp_rl":(0,200),"tyre_temp_rr":(0,200),
         "brake_temp_fl":(0,1200),"brake_temp_fr":(0,1200),"brake_temp_rl":(0,1200),"brake_temp_rr":(0,1200),
         "fuel_kg":(0,110),"ers_deployment_kw":(0,120),"ers_harvesting_kw":(0,120)}

_SMOOTH = {"speed":5,"throttle":3,"brake":3,"steering":7,"rpm":5,
           "lateral_g":5,"longitudinal_g":5,"tyre_temp_fl":11,"tyre_temp_fr":11,
           "tyre_temp_rl":11,"tyre_temp_rr":11,"brake_temp_fl":9,"brake_temp_fr":9,
           "brake_temp_rl":9,"brake_temp_rr":9,"ers_deployment_kw":7,"ers_harvesting_kw":7}

@dataclass
class ProcessingRecord:
    step: str
    signals_affected: list
    n_samples_modified: int
    notes: str = ""

@dataclass
class CleaningReport:
    driver: str
    lap: int
    steps: list = field(default_factory=list)
    def summary(self):
        lines = [f"Cleaning report — {self.driver} Lap {self.lap}"]
        for s in self.steps:
            lines.append(f"  [{s.step}] {s.signals_affected} — {s.n_samples_modified} samples — {s.notes}")
        return "\n".join(lines)

def clean(lap, smooth=True, spike_threshold=4.0, interpolation="linear"):
    from loader import TelemetryLap
    df = lap.data.copy()
    report = CleaningReport(driver=lap.driver, lap=lap.lap_number)

    df, rec = _clip(df);               report.steps.append(rec)
    df, rec = _fill_nans(df, interpolation); report.steps.append(rec)
    df, rec = _despike(df, spike_threshold, interpolation); report.steps.append(rec)
    if smooth:
        df, rec = _smooth(df);         report.steps.append(rec)
    df = _derivatives(df)

    cleaned = TelemetryLap(data=df, lap_number=lap.lap_number, driver=lap.driver,
                           session=lap.session, source_file=lap.source_file,
                           sample_rate_hz=lap.sample_rate_hz,
                           signals_present=lap.signals_present, warnings=lap.warnings.copy())
    log.info(report.summary())
    return cleaned, report

def _clip(df):
    affected, total = [], 0
    for col,(lo,hi) in _CLIP.items():
        if col not in df.columns: continue
        mask = (df[col]<lo)|(df[col]>hi); n = int(mask.sum())
        if n: df[col]=df[col].clip(lo,hi); affected.append(col); total+=n
    return df, ProcessingRecord("clip_bounds", affected, total, "Hard clip to physical bounds.")

def _fill_nans(df, method):
    num = df.select_dtypes(include=np.number).columns.tolist()
    before = int(df[num].isna().sum().sum())
    if before > 0:
        df[num] = df[num].interpolate(method=method,limit_direction="both").ffill().bfill()
    after = int(df[num].isna().sum().sum())
    return df, ProcessingRecord("fill_nans", num if before>0 else [], before-after,
                                f"method='{method}'. {before} NaN before, {after} after.")

def _despike(df, threshold, method):
    targets = [c for c in ["speed","throttle","brake","rpm","lateral_g","longitudinal_g","ers_deployment_kw"] if c in df.columns]
    total = 0
    for col in targets:
        s=df[col].copy(); rm=s.rolling(30,center=True,min_periods=1).mean()
        rs=s.rolling(30,center=True,min_periods=1).std().replace(0,np.nan)
        mask=(s-rm).abs()/rs>threshold; n=int(mask.sum())
        if n: df.loc[mask,col]=np.nan; total+=n
    if total>0:
        num=df.select_dtypes(include=np.number).columns
        df[num]=df[num].interpolate(method=method,limit_direction="both").ffill().bfill()
    return df, ProcessingRecord("remove_spikes", targets, total, f"window=30, threshold={threshold}.")

def _smooth(df):
    affected = []
    for col,win in _SMOOTH.items():
        if col not in df.columns: continue
        df[col]=df[col].rolling(win,center=True,min_periods=1).mean(); affected.append(f"{col}(w={win})")
    return df, ProcessingRecord("smooth", affected, len(df)*len(affected), "Centred rolling mean.")

def _derivatives(df):
    dt = df["time"].diff().replace(0,np.nan)
    for col in ("speed","throttle","brake","steering"):
        if col not in df.columns: continue
        deriv = df[col].diff()/dt
        if col=="speed": deriv = deriv/3.6
        df[f"d_{col}"] = deriv.round(5)
    return df
