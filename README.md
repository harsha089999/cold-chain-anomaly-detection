# Temperature Anomaly Detection (Cold Chain Method)

[![Run project](https://github.com/harsha089999/cold-chain-anomaly-detection/actions/workflows/run.yml/badge.svg)](https://github.com/harsha089999/cold-chain-anomaly-detection/actions/workflows/run.yml)

**Catch equipment failures early from a real temperature sensor, using rolling features, Isolation Forest and a persistence rule, scored against labelled failures.**

> **About the data:** this project runs on a real public dataset, **Numenta Anomaly Benchmark (NAB), realKnownCause/machine_temperature_system_failure**, included in `data/` so it runs straight after cloning. Sources and licence are listed below and in [`data/README.md`](data/README.md).

## The business question

Can we raise an alert early enough to act when a machine or cold room starts to fail, without flooding the team with false alarms?

## What this project does

Cold rooms, freezers and industrial machines all fail the same way in sensor data: either a sudden jump or a slow drift away from normal. This project applies the detection method a cold chain team would use to a real temperature sensor that has independently labelled failure windows, so you can actually measure precision and recall instead of guessing.

## The data

A real temperature sensor on a large industrial machine, read every 5 minutes for about 11 weeks. NAB's notes describe a planned shutdown, then a subtle anomaly that led directly to a catastrophic failure of the machine, and its labels file marks four anomaly windows in this series. The labels come from knowledge of what actually happened, not from this model.

**Why this data for a cold chain project.** Public cold room sensor data with confirmed failures is very hard to find. The detection method doesn't care whether the sensor sits in a freezer or on a machine, so this tests it on real failures. To use it on cold rooms, swap in your own readings and set the rule threshold to your spec limits.

**Source:**
- [NAB on GitHub](https://github.com/numenta/NAB)
- [Kaggle: Numenta Anomaly Benchmark (boltzmannbrain/nab)](https://www.kaggle.com/datasets/boltzmannbrain/nab)

**Licence:** NAB is released under the AGPL-3.0 licence by Numenta.

## How it works

1. Learn 'normal' from the first 6 days only (before any labelled event), so the model never trains on the failures it's meant to catch.
2. Features: temperature, 1-hour rolling standard deviation, 15-minute change, and deviation from the 6-hour mean.
3. Isolation Forest scores every reading. A reading is flagged above the 99.5th percentile of normal scores, and the flag must last 30 minutes.
4. A simple rule runs alongside: temperature below the training week's 0.5th percentile for 30 minutes (machine stopped or running cold).
5. Merge alerts less than 2 hours apart into incidents, then score against NAB's windows: windows caught, incident precision, and hours from window start to first alert.

## Run it

```bash
git clone https://github.com/harsha089999/cold-chain-anomaly-detection
cd cold-chain-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Outputs are written to `outputs/`. Python 3.10 or newer is recommended.

## Results from the included run

This is the actual console output when the script is run as shipped:

```text
22,695 readings from 2013-12-02 to 2014-02-19, 4 labelled anomaly windows

Incidents raised: 16  | in a labelled window: 5 (precision 31%)  | windows caught: 4 of 4

       window_start          window_end  caught         first_alert  hours_after_window_start
2013-12-10 06:25:00 2013-12-12 05:35:00    True 2013-12-10 07:45:00                       1.3
2013-12-15 17:50:00 2013-12-17 17:00:00    True 2013-12-16 06:45:00                      12.9
2014-01-27 14:20:00 2014-01-29 13:30:00    True 2014-01-28 14:00:00                      23.7
2014-02-07 14:55:00 2014-02-09 14:05:00    True 2014-02-07 18:05:00                       3.2

Incidents:
              start                 end  min_temp  max_temp  duration_min  in_labelled_window
2013-12-10 07:45:00 2013-12-10 17:05:00      48.4      54.0           565                True
2013-12-10 22:55:00 2013-12-10 23:10:00      76.5      78.1            20                True
2013-12-16 06:45:00 2013-12-16 19:45:00       2.1      99.0           765                True
2013-12-28 04:35:00 2013-12-28 04:35:00      86.3      86.3             5               False
2013-12-28 07:00:00 2013-12-28 07:20:00      87.4      91.1            25               False
2014-01-05 16:35:00 2014-01-05 18:30:00      52.5      75.2            35               False
2014-01-13 21:50:00 2014-01-13 21:50:00      79.9      79.9             5               False
2014-01-16 13:10:00 2014-01-16 13:10:00      81.0      81.0             5               False
2014-01-24 12:30:00 2014-01-24 13:40:00      51.3      84.3            50               False
2014-01-28 14:00:00 2014-01-28 17:25:00      51.2      54.2           135                True
2014-01-29 14:20:00 2014-01-29 16:15:00      48.9      54.0           120               False
2014-01-29 19:05:00 2014-01-29 19:05:00      74.9      74.9             5               False
2014-01-30 17:50:00 2014-01-30 20:05:00      46.6      54.0           140               False
2014-02-03 00:30:00 2014-02-03 00:35:00      69.8      70.1            10               False
2014-02-03 07:05:00 2014-02-03 13:00:00      43.9      94.9           335               False
2014-02-07 18:05:00 2014-02-09 13:00:00      25.9      85.6          2565                True
```

All 4 labelled windows were caught, the earliest 1.3 hours after the window opened and the slowest 23.7 hours in. Out of 16 incidents, 5 fell inside a labelled window, so incident precision is 31%. That's the honest trade-off: catching every real event means some false alarms. Several of those 'false' incidents (late January and early February) are large temperature drops that sit just outside NAB's windows, so a person would probably still want to see them. A site would tune the persistence rule and score cut-off against how much a missed failure costs compared with a wasted call-out.

## Output files

| File | What it contains |
|---|---|
| `outputs/incidents.csv` | Each incident: start, end, min and max temperature, duration, and whether it overlaps a labelled window. |
| `outputs/window_recall.csv` | Each labelled window: caught or not, first alert time, and hours after the window opened. |
| `outputs/readings_scored.csv.gz` | Every reading with features, anomaly score and flags (for plotting). |

## Limitations and honest caveats

- This is one sensor with 4 events, so the precision and recall figures are indicative, not statistically solid.
- NAB's windows are deliberately wide, so 'hours after window start' is not the same as lead time before failure.
- Temperature units aren't documented by NAB. The method works on relative change, so that doesn't affect detection.
- Real cold rooms have defrost cycles and door openings that need their own suppression rules.

## Turning it into a dashboard

A line chart of temperature with incidents shaded and labelled windows outlined, an incident table with acknowledge status, and a precision and recall card per month.

## Ideas to extend it

- Run the same code on other NAB sensors and report results across all of them.
- Add a seasonal baseline for equipment with daily cycles (defrost, shifts).
- Compare with a forecasting-based detector that alerts on large residuals.

## Skills demonstrated

Time series feature engineering, unsupervised anomaly detection, evaluation against ground truth, alert design, explaining the precision vs recall trade-off to operations.

**Tools:** Python, pandas, scikit-learn

## Project structure

```text
cold-chain-anomaly-detection/
├── src/main.py                 # the whole pipeline
├── data/                       # real source data (gzipped) + data/README.md with source and licence
├── outputs/                    # files from the run shown above
├── requirements.txt
├── .github/workflows/run.yml   # CI: installs and runs the project on every push
└── LICENSE
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The data in `data/` keeps its original licence (see above).
