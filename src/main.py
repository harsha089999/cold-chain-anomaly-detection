"""Cold chain anomaly detection: rolling features + Isolation Forest + rules -> incidents."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SPEC = {"chiller": (0.0, 5.0), "freezer": (-25.0, -15.0)}


def make_data(seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-09-01", periods=30 * 288, freq="5min")
    frames = []
    for r in range(12):
        kind = "chiller" if r < 7 else "freezer"
        setpoint = 3.0 if kind == "chiller" else -20.0
        temp = setpoint + rng.normal(0, 0.3, len(ts))
        # defrost cycles every 6h for freezers (normal behaviour)
        if kind == "freezer":
            temp += np.where((ts.hour % 6 == 0) & (ts.minute < 20), 4.0, 0)
        truth = np.zeros(len(ts), dtype=bool)
        for _ in range(rng.integers(1, 3)):  # door left open: fast spike
            s = rng.integers(0, len(ts) - 30)
            temp[s:s + 12] += np.linspace(0, 8, 12)
            truth[s:s + 12] = True
        if r in (2, 9):  # slow compressor failure
            s = rng.integers(len(ts) // 2, len(ts) - 400)
            temp[s:s + 300] += np.linspace(0, 9, 300)
            truth[s:s + 300] = True
        frames.append(pd.DataFrame({"ts": ts, "room": f"R{r:02d}", "kind": kind,
                                    "setpoint": setpoint, "temp_c": temp.round(2), "true_event": truth}))
    return pd.concat(frames, ignore_index=True)


def add_features(df):
    g = df.groupby("room")["temp_c"]
    df["dev"] = df.temp_c - df.setpoint
    df["roll_mean_1h"] = g.transform(lambda s: s.rolling(12, min_periods=1).mean())
    df["roll_std_1h"] = g.transform(lambda s: s.rolling(12, min_periods=2).std()).fillna(0)
    df["delta_15m"] = g.diff(3).fillna(0)
    df["dev_mean_1h"] = df.roll_mean_1h - df.setpoint
    return df


def main():
    df = add_features(make_data())
    feats = ["dev", "roll_std_1h", "delta_15m", "dev_mean_1h"]
    df["ml_flag"] = False
    for kind, g in df.groupby("kind"):
        iso = IsolationForest(contamination=0.02, random_state=0).fit(g[feats])
        df.loc[g.index, "ml_flag"] = iso.predict(g[feats]) == -1
    lo = df.kind.map(lambda k: SPEC[k][0])
    hi = df.kind.map(lambda k: SPEC[k][1])
    out_spec = (df.temp_c < lo) | (df.temp_c > hi)
    df["rule_flag"] = out_spec.groupby(df.room).transform(lambda s: s.rolling(6, min_periods=6).sum() >= 6)
    # require ML flag to persist 3 readings to cut noise from defrost and one-off blips
    df["alert"] = df.rule_flag | df.ml_flag.groupby(df.room).transform(
        lambda s: s.rolling(3, min_periods=3).sum() >= 3)

    # group alerts into incidents
    df["block"] = (df.alert != df.groupby("room").alert.shift()).cumsum()
    inc = (df[df.alert].groupby(["room", "block"])
           .agg(start=("ts", "min"), end=("ts", "max"), peak_temp=("temp_c", "max"),
                readings=("ts", "size"), overlaps_true=("true_event", "any"))
           .reset_index().drop(columns="block"))
    inc["duration_min"] = inc.readings * 5
    inc.to_csv(OUT / "incidents.csv", index=False)
    df.drop(columns="block").to_csv(OUT / "readings_scored.csv", index=False)

    true_blocks = df[df.true_event].assign(b=(df.true_event != df.groupby("room").true_event.shift()).cumsum())
    caught = true_blocks.groupby(["room", "b"]).alert.any()
    print(f"Incidents raised: {len(inc)}  precision: {inc.overlaps_true.mean():.0%}  "
          f"recall: {caught.mean():.0%} of {len(caught)} real events")


if __name__ == "__main__":
    main()
