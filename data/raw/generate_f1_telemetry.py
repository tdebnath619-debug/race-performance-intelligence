"""
F1 Performance Intelligence System
data/raw/generate_f1_telemetry.py

Generates realistic 2-lap F1 telemetry for a Bahrain-like 15-corner circuit.
Includes all F1-specific channels: ERS, DRS, fuel, g-forces, brake temps, tyre temps.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG    = np.random.default_rng(2024)
HZ     = 100          # 100 Hz — realistic F1 data rate
LAP_M  = 5_412        # Bahrain circuit length (m)

# 15 corners: (entry_dist, apex_speed_kmh, exit_speed_kmh, brake_dist_m, corner_type)
CORNERS = [
    (310,  80,  120, 150, "slow"),    # T1 braking zone
    (550, 115,  148,  80, "medium"),  # T2
    (750,  75,  110, 130, "slow"),    # T3 (hairpin)
    (1050,155,  190,  60, "fast"),    # T4
    (1350,120,  158,  90, "medium"),  # T5
    (1580,145,  178,  70, "fast"),    # T6
    (1820, 95,  130, 110, "medium"),  # T7
    (2100, 65,  100, 150, "slow"),    # T8 hairpin
    (2380,110,  145,  90, "medium"),  # T9
    (2650,165,  205,  50, "fast"),    # T10 DRS zone
    (3000, 85,  125, 120, "medium"),  # T11
    (3280,130,  168,  75, "fast"),    # T12
    (3500, 70,  108, 140, "slow"),    # T13 chicane
    (3780, 95,  132, 110, "medium"),  # T14
    (4100,155,  200,  55, "fast"),    # T15 → DRS detection
]

# DRS zones (activation_dist, end_dist)
DRS_ZONES = [(4300, 5100), (200, 700)]


def _speed_profile(noise: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    dists  = np.linspace(0, LAP_M, LAP_M * 3)
    speeds = np.full_like(dists, 320.0)   # base straight speed (F1 on Bahrain)

    for entry_dist, apex_speed, exit_speed, brake_dist, _ in CORNERS:
        # Braking
        b_end = entry_dist + brake_dist
        mask_b = (dists >= entry_dist) & (dists < b_end)
        t_b = (dists[mask_b] - entry_dist) / brake_dist
        speeds[mask_b] = np.minimum(speeds[mask_b], 320 * (1-t_b) + apex_speed * t_b)
        # Apex
        mask_a = (dists >= b_end) & (dists < b_end + 40)
        speeds[mask_a] = np.minimum(speeds[mask_a], apex_speed)
        # Exit
        ex_s, ex_e = b_end + 40, min(b_end + 200, LAP_M)
        mask_e = (dists >= ex_s) & (dists < ex_e)
        t_e = (dists[mask_e] - ex_s) / max(ex_e - ex_s, 1)
        speeds[mask_e] = np.minimum(speeds[mask_e], apex_speed * (1-t_e) + exit_speed * t_e)

    speeds = np.clip(speeds, 60, 355) + RNG.normal(0, noise, len(dists))
    return dists, speeds


def _derive_all(dists, speeds, lap_num, fuel_start_kg=90.0, noise=0.3):
    n  = len(dists)
    dt = np.gradient(dists) / np.clip(speeds / 3.6, 5, 400)
    ts = np.cumsum(dt)
    d_spd = np.gradient(speeds, ts)

    # ── Controls ──────────────────────────────────────────────────────────
    throttle = np.clip(d_spd / 80 + 0.5 + RNG.normal(0, 0.015, n), 0, 1)
    brake    = np.where(d_spd < -10, np.clip(-d_spd / 100, 0, 1), 0.0)
    brake    = np.clip(brake + RNG.normal(0, 0.005, n), 0, 1)

    # Gear from speed
    gear = np.where(speeds < 80,  1,
           np.where(speeds < 120, 2,
           np.where(speeds < 160, 3,
           np.where(speeds < 200, 4,
           np.where(speeds < 240, 5,
           np.where(speeds < 280, 6,
           np.where(speeds < 315, 7, 8)))))))
    rpm = gear * 2200 + speeds * 18 + RNG.normal(0, 80, n)
    rpm = np.clip(rpm, 4000, 18000)

    # Steering
    steering = np.zeros(n)
    for entry_dist, *_ in CORNERS:
        mask = (dists >= entry_dist) & (dists < entry_dist + 120)
        if mask.sum() > 0:
            steering[mask] = RNG.choice([-1, 1]) * np.clip(
                np.linspace(0, 0.6, mask.sum()) + RNG.normal(0, 0.02, mask.sum()), -1, 1)

    # ── F1 Power Unit ──────────────────────────────────────────────────────
    # ERS deployment: max on straights (120 kW limit), harvest in braking
    ers_dep = np.where(throttle > 0.7, np.clip(speeds / 300 * 120, 0, 120), 0.0)
    ers_dep += RNG.normal(0, 2, n)
    ers_dep  = np.clip(ers_dep, 0, 120)

    ers_har = np.where(brake > 0.1, np.clip(brake * 80, 0, 80), 0.0)
    ers_har += RNG.normal(0, 1, n)
    ers_har  = np.clip(ers_har, 0, 120)

    mguk_kw = ers_dep - ers_har
    mguh_kw = RNG.uniform(10, 40, n)  # always harvesting from turbo

    # Fuel flow (FIA limit 100 kg/h; ~65 at cruise)
    fuel_flow = np.where(throttle > 0.9, 100, throttle * 80 + 10)
    fuel_flow = np.clip(fuel_flow + RNG.normal(0, 1, n), 0, 100)

    fuel_consumed = np.cumsum(fuel_flow / 3600 * dt)   # kg consumed per sample
    fuel_kg = np.clip(fuel_start_kg - fuel_consumed, 0, fuel_start_kg)

    # ── DRS ───────────────────────────────────────────────────────────────
    drs_state = np.zeros(n, dtype=int)
    for (act_d, end_d) in DRS_ZONES:
        mask = (dists >= act_d) & (dists <= end_d) & (speeds > 210)
        drs_state[mask] = 1

    # ── G-forces ──────────────────────────────────────────────────────────
    g_lon = np.clip(d_spd / (3.6 * 9.81), -5.5, 3.0) + RNG.normal(0, 0.05, n)
    # Lateral G: approximated from steering + speed
    g_lat = steering * (speeds / 200) ** 2 * 3.5 + RNG.normal(0, 0.1, n)
    g_lat = np.clip(g_lat, -6.5, 6.5)
    g_vert= 1.0 + np.abs(g_lat) * 0.05 + RNG.normal(0, 0.02, n)

    # ── Temperatures ──────────────────────────────────────────────────────
    # Brake temps: rise with braking, cool on straights
    brake_base = 200 + lap_num * 30
    def _brake_temp(corner_mask_fn):
        t = np.full(n, brake_base, dtype=float)
        for i in range(1, n):
            heat  = brake[i] * 800
            cool  = (t[i-1] - 80) * 0.003
            t[i]  = t[i-1] + heat * dt[i] - cool * dt[i]
        return np.clip(t + RNG.normal(0, 10, n), 50, 1100)

    brake_fl = _brake_temp(None)
    brake_fr = brake_fl + RNG.normal(0, 20, n)   # slight imbalance
    brake_rl = brake_fl * 0.7 + RNG.normal(0, 15, n)
    brake_rr = brake_rl + RNG.normal(0, 12, n)

    # Tyre surface temps: warm up, then stable
    tyre_base = 85 + min(lap_num * 5, 20)
    tyre_surf_fl = np.clip(tyre_base + np.abs(g_lat) * 8 + RNG.normal(0, 3, n), 60, 150)
    tyre_surf_fr = tyre_surf_fl + RNG.normal(0, 4, n)
    tyre_surf_rl = tyre_base + np.abs(g_lat) * 6 + RNG.normal(0, 3, n)
    tyre_surf_rr = tyre_surf_rl + RNG.normal(0, 4, n)

    # Suspension travel (mm)
    susp_fl = -20 + np.abs(g_lat) * 5 + RNG.normal(0, 1, n)
    susp_fr = -20 - np.abs(g_lat) * 5 + RNG.normal(0, 1, n)
    susp_rl = -15 + np.abs(g_lat) * 4 + RNG.normal(0, 1, n)
    susp_rr = -15 - np.abs(g_lat) * 4 + RNG.normal(0, 1, n)

    return pd.DataFrame({
        "time":              ts.round(4),
        "distance":          dists.round(2),
        "speed":             speeds.round(2),
        "throttle":          throttle.round(4),
        "brake":             brake.round(4),
        "gear":              gear.astype(int),
        "rpm":               rpm.round(0).astype(int),
        "steering":          steering.round(4),
        "drs_state":         drs_state,
        "ers_deployment_kw": ers_dep.round(2),
        "ers_harvesting_kw": ers_har.round(2),
        "mguk_kw":           mguk_kw.round(2),
        "mguh_kw":           mguh_kw.round(2),
        "fuel_flow_kgh":     fuel_flow.round(2),
        "fuel_kg":           fuel_kg.round(3),
        "g_lateral":         g_lat.round(4),
        "g_longitudinal":    g_lon.round(4),
        "g_vertical":        g_vert.round(4),
        "brake_temp_fl":     brake_fl.round(1),
        "brake_temp_fr":     brake_fr.round(1),
        "brake_temp_rl":     brake_rl.round(1),
        "brake_temp_rr":     brake_rr.round(1),
        "tyre_surf_fl":      tyre_surf_fl.round(1),
        "tyre_surf_fr":      tyre_surf_fr.round(1),
        "tyre_surf_rl":      tyre_surf_rl.round(1),
        "tyre_surf_rr":      tyre_surf_rr.round(1),
        "susp_travel_fl":    susp_fl.round(2),
        "susp_travel_fr":    susp_fr.round(2),
        "susp_travel_rl":    susp_rl.round(2),
        "susp_travel_rr":    susp_rr.round(2),
    })


def generate(out_path="data/raw/f1_telemetry.csv", n_laps=2) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames  = []
    t_off   = 0.0
    fuel_kg = 90.0   # full tank at race start

    for lap in range(n_laps):
        noise = 0.4 if lap == 0 else 0.8   # lap 1 cleaner baseline
        dists, speeds = _speed_profile(noise)
        df_lap = _derive_all(dists, speeds, lap_num=lap, fuel_start_kg=fuel_kg, noise=noise)
        df_lap["time"]       += t_off
        df_lap["lap_number"]  = lap
        df_lap["lap_time"]    = df_lap["time"] - t_off
        t_off   = float(df_lap["time"].iloc[-1]) + 0.01
        fuel_kg = float(df_lap["fuel_kg"].iloc[-1])
        frames.append(df_lap)

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(out_path, index=False)
    print(f"F1 telemetry generated → {out_path}  ({len(df_all):,} rows, {n_laps} laps, "
          f"{len(df_all.columns)} channels)")
    return out_path


if __name__ == "__main__":
    generate()
