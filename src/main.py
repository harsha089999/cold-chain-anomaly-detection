"""Temperature anomaly detection on a real industrial sensor with labelled failures.

Data: Numenta Anomaly Benchmark (NAB), realKnownCause/machine_temperature_system_failure.csv. A temperature
sensor on a large industrial machine, 5-minute readings from Dec 2013 to Feb 2014, with four anomaly windows
labelled by NAB (NAB notes a planned shutdown and a catastrophic failure among them).
Also on Kaggle as boltzmannbrain/nab. Stored in data/.

The same approach (rolling features + Isolation Forest + a persistence rule) is what you would use on cold room
or freezer sensors; the units here are the machine's own temperature readings.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TRAIN_END = "2013-12-08"  # first ~6 days, before any labelled event, used to learn "normal"


def load_windows():
    import gzip
    with gzip.open(DATA / "nab_windows.json.gz", "rt") as f:
        w = json.load(f)["realKnownCause/machine_temperature_system_failure.csv"]
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in w]


def add_features(df):
    s = df.temp
    df["roll_mean_1h"] = s.rolling(12, min_periods=1).mean()
    df["roll_std_1h"] = s.rolling(12, min_periods=2).std().fillna(0)
    df["delta_15m"] = s.diff(3).fillna(0)
    df["dev_from_6h"] = s - s.rolling(72, min_periods=1).mean()
    return df


def main():
    df = pd.read_csv(DATA / "nab_machine_temp.csv.gz", parse_dates=["timestamp"]).rename(
        columns={"timestamp": "ts", "value": "temp"})
    windows = load_windows()
    df["true_window"] = False
    for a, b in windows:
        df.loc[df.ts.between(a, b), "true_window"] = True
    df = add_features(df)
    print(f"{len(df):,} readings from {df.ts.min():%Y-%m-%d} to {df.ts.max():%Y-%m-%d}, "
          f"{len(windows)} labelled anomaly windows")

    feats = ["temp", "roll_std_1h", "delta_15m", "dev_from_6h"]
    train = df[df.ts < TRAIN_END]
    iso = IsolationForest(contamination=0.01, random_state=0).fit(train[feats])
    df["score"] = -iso.score_samples(df[feats])
    cut = np.quantile(-iso.score_samples(train[feats]), 0.995)  # top 0.5% of "normal" behaviour
    df["ml_flag"] = df.score > cut

    # rule: temperature below the 0.5th percentile of the training week for 30 minutes (machine running cold/stopped)
    low = train.temp.quantile(0.005)
    df["rule_flag"] = (df.temp < low).rolling(6, min_periods=6).sum().ge(6)
    # ML flags must persist for 30 minutes to cut one-off blips
    df["alert"] = df.rule_flag | df.ml_flag.rolling(6, min_periods=6).sum().ge(6)

    df["block"] = (df.alert != df.alert.shift()).cumsum()
    inc = (df[df.alert].groupby("block")
           .agg(start=("ts", "min"), end=("ts", "max"), min_temp=("temp", "min"), max_temp=("temp", "max"),
                readings=("ts", "size"), in_labelled_window=("true_window", "any"))
           .reset_index(drop=True))
    # merge alerts less than 2 hours apart into one incident
    inc["incident"] = (inc.start - inc.end.shift() > pd.Timedelta("2h")).cumsum()
    inc = (inc.groupby("incident").agg(start=("start", "min"), end=("end", "max"), min_temp=("min_temp", "min"),
                                        max_temp=("max_temp", "max"), readings=("readings", "sum"),
                                        in_labelled_window=("in_labelled_window", "any")).reset_index(drop=True))
    inc["duration_min"] = inc.readings * 5
    inc.round({"min_temp": 2, "max_temp": 2}).to_csv(OUT / "incidents.csv", index=False)
    df.drop(columns="block").round({f: 3 for f in feats + ["score"]}).to_csv(OUT / "readings_scored.csv.gz", index=False)

    rows = []
    for a, b in windows:
        hits = inc[(inc.end >= a) & (inc.start <= b)]
        first = hits.start.min() if len(hits) else pd.NaT
        rows.append({"window_start": a, "window_end": b, "caught": len(hits) > 0, "first_alert": first,
                     "hours_after_window_start": round((first - a) / pd.Timedelta("1h"), 1) if len(hits) else None})
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "window_recall.csv", index=False)
    print(f"\nIncidents raised: {len(inc)}  | in a labelled window: {inc.in_labelled_window.sum()} "
          f"(precision {inc.in_labelled_window.mean():.0%})  | windows caught: {ev.caught.sum()} of {len(ev)}")
    print("\n" + ev.to_string(index=False))
    print("\nIncidents:\n" + inc[["start", "end", "min_temp", "max_temp", "duration_min", "in_labelled_window"]]
          .round({"min_temp": 1, "max_temp": 1}).to_string(index=False))


if __name__ == "__main__":
    main()
