from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 6000
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy()/60
    dow = ts.dayofweek.to_numpy()
    daily = 0.55 + 0.24*np.sin((hour-7)/24*2*np.pi) + 0.12*np.sin((hour-17)/24*2*np.pi)
    weekend = np.where(dow >= 5, 0.82, 1.0)
    load = np.clip(daily*weekend + rng.normal(0, 0.10, n), 0.15, 1.0)
    voltage = np.clip(rng.normal(230, 4.0, n), 215, 245)
    current = np.clip(1.0 + load*13 + rng.normal(0, 1.1, n), 0.8, 16.0)
    typ = rng.choice(["resistive","mixed","inductive"], n, p=[0.24,0.43,0.33])
    base = np.select([typ=="resistive",typ=="mixed",typ=="inductive"],
        [rng.uniform(0.93,0.995,n),rng.uniform(0.78,0.96,n),rng.uniform(0.52,0.84,n)])
    pf = np.clip(base + 0.025*(load-0.5) + rng.normal(0,0.012,n),0.45,0.999)
    S = voltage*current
    P = S*pf
    Q = np.sqrt(np.maximum(S*S-P*P,0))
    vm = voltage+rng.normal(0,0.35,n)
    im = np.maximum(current+rng.normal(0,0.08,n),0.1)
    pm = np.maximum(P+rng.normal(0,12,n),0)
    qm = np.maximum(Q+rng.normal(0,12,n),0)
    Sm = vm*im
    pfc = np.clip(pm/np.maximum(Sm,1e-6),0,1)
    cls = np.where(pfc<0.80,"Low",np.where(pfc<0.90,"Moderate","Good"))
    df = pd.DataFrame({"timestamp":ts.astype(str),"voltage_V":vm.round(2),"current_A":im.round(3),
      "active_power_W":pm.round(2),"reactive_power_var":qm.round(2),"load_level":load.round(3),
      "hour":hour.round(2),"day_of_week":dow,"simulated_load_type":typ,"apparent_power_VA":Sm.round(2),
      "power_factor":pfc.round(4),"pf_class":cls})
    for c in ["voltage_V","current_A","reactive_power_var"]:
        df.loc[rng.choice(n,12,replace=False),c]=np.nan
    df=pd.concat([df,df.iloc[[10,250,1000]]],ignore_index=True)
    out=Path("data"); out.mkdir(exist_ok=True)
    df.to_csv(out/"simulated_power_factor_readings.csv",index=False)
    print(f"Wrote {len(df)} rows to data/simulated_power_factor_readings.csv")
if __name__=="__main__": main()
