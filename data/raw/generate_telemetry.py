"""
data/raw/generate_telemetry.py
================================
Synthetic telemetry generator — Bahrain-inspired 15-corner circuit.
Produces a 2-lap CSV with all standard channels for pipeline testing.
"""
import numpy as np, pandas as pd
from pathlib import Path

RNG   = np.random.default_rng(2024)
HZ    = 100
LAP_M = 5_412

CORNERS = [
    (310,80,120,150),(550,115,148,80),(750,75,110,130),
    (1050,155,190,60),(1350,120,158,90),(1580,145,178,70),
    (1820,95,130,110),(2100,65,100,150),(2380,110,145,90),
    (2650,165,205,50),(3000,85,125,120),(3280,130,168,75),
    (3500,70,108,140),(3780,95,132,110),(4100,155,200,55),
]
DRS_ZONES = [(4300,5100),(200,700)]

def _speed_profile(noise=0.4):
    d = np.linspace(0,LAP_M,LAP_M*3)
    v = np.full_like(d,310.0)
    for entry,apex_v,exit_v,brk in CORNERS:
        b_end=entry+brk
        mb=(d>=entry)&(d<b_end)
        if mb.sum():
            t=(d[mb]-entry)/brk; v[mb]=np.minimum(v[mb],310*(1-t)+apex_v*t)
        ma=(d>=b_end)&(d<b_end+40); v[ma]=np.minimum(v[ma],apex_v)
        es,ee=b_end+40,min(b_end+200,LAP_M); me=(d>=es)&(d<ee)
        if me.sum():
            t=(d[me]-es)/max(ee-es,1); v[me]=np.minimum(v[me],apex_v*(1-t)+exit_v*t)
    return d,np.clip(v,60,340)+RNG.normal(0,noise,len(d))

def _build_lap(dists,speeds,lap_num,fuel_start=90.0,noise=0.4):
    n=len(dists); dt=np.gradient(dists)/np.clip(speeds/3.6,5,400); ts=np.cumsum(dt)
    dv=np.gradient(speeds,ts)
    throttle=np.clip(dv/80+0.5+RNG.normal(0,0.015,n),0,1)
    brake=np.clip(np.where(dv<-10,np.clip(-dv/100,0,1),0.0)+RNG.normal(0,0.005,n),0,1)
    gear=np.where(speeds<80,1,np.where(speeds<120,2,np.where(speeds<160,3,
         np.where(speeds<200,4,np.where(speeds<240,5,np.where(speeds<280,6,
         np.where(speeds<315,7,8)))))))
    rpm=np.clip(gear*2200+speeds*18+RNG.normal(0,80,n),4000,18000)
    steering=np.zeros(n)
    for entry,*_ in CORNERS:
        mask=(dists>=entry)&(dists<entry+120)
        if mask.sum():
            steering[mask]=RNG.choice([-1,1])*np.clip(
                np.linspace(0,0.6,mask.sum())+RNG.normal(0,0.02,mask.sum()),-1,1)
    ers_dep=np.clip(np.where(throttle>0.7,speeds/300*120,0)+RNG.normal(0,2,n),0,120)
    ers_har=np.clip(np.where(brake>0.1,brake*80,0)+RNG.normal(0,1,n),0,120)
    fuel_flow=np.clip(np.where(throttle>0.9,100,throttle*80+10)+RNG.normal(0,1,n),0,100)
    fuel_kg=np.clip(fuel_start-np.cumsum(fuel_flow/3600*dt),0,fuel_start)
    drs=np.zeros(n,dtype=int)
    for a,e in DRS_ZONES: drs[(dists>=a)&(dists<=e)&(speeds>210)]=1
    g_lon=np.clip(dv/(3.6*9.81),-5.5,3.0)+RNG.normal(0,0.05,n)
    g_lat=np.clip(steering*(speeds/200)**2*3.5+RNG.normal(0,0.1,n),-6.5,6.5)
    bt=np.clip(200+lap_num*30+brake*800*dt+RNG.normal(0,10,n),50,1100)
    tt=np.clip(85+min(lap_num*5,20)+np.abs(g_lat)*8+RNG.normal(0,3,n),60,150)
    return pd.DataFrame({
        "time":ts.round(4),"distance":dists.round(2),"speed":speeds.round(2),
        "throttle":throttle.round(4),"brake":brake.round(4),
        "gear":gear.astype(int),"rpm":rpm.round(0).astype(int),"steering":steering.round(4),
        "drs_state":drs,"ers_deployment_kw":ers_dep.round(2),"ers_harvesting_kw":ers_har.round(2),
        "fuel_flow_kgh":fuel_flow.round(2),"fuel_kg":fuel_kg.round(3),
        "lateral_g":g_lat.round(4),"longitudinal_g":g_lon.round(4),
        "brake_temp_fl":bt.round(1),"brake_temp_fr":(bt+RNG.normal(0,20,n)).round(1),
        "brake_temp_rl":(bt*0.7).round(1),"brake_temp_rr":(bt*0.7+RNG.normal(0,12,n)).round(1),
        "tyre_temp_fl":tt.round(1),"tyre_temp_fr":(tt+RNG.normal(0,4,n)).round(1),
        "tyre_temp_rl":(tt*0.95).round(1),"tyre_temp_rr":(tt*0.95+RNG.normal(0,4,n)).round(1),
    })

def generate(out_path="data/raw/telemetry.csv",n_laps=2):
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
    frames,t_off,fuel=[],0.0,90.0
    for lap in range(n_laps):
        noise=0.4 if lap==0 else 0.9
        dists,speeds=_speed_profile(noise)
        df=_build_lap(dists,speeds,lap,fuel,noise)
        df["time"]+=t_off; df["lap_number"]=lap; df["lap_time"]=df["time"]-t_off
        t_off=float(df["time"].iloc[-1])+0.01; fuel=float(df["fuel_kg"].iloc[-1])
        frames.append(df)
    out=pd.concat(frames,ignore_index=True); out.to_csv(out_path,index=False)
    print(f"Telemetry → {out_path}  ({len(out):,} rows, {n_laps} laps, {len(out.columns)} channels)")
    return out_path

if __name__=="__main__": generate()
