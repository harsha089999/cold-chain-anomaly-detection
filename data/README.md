# Data

Real data, not simulated. Files are the original CSVs, gzip-compressed to keep the repo small. pandas and DuckDB read
`.csv.gz` directly, so nothing needs unzipping.

**Dataset:** Numenta Anomaly Benchmark (NAB), realKnownCause/machine_temperature_system_failure

- [NAB on GitHub](https://github.com/numenta/NAB)
- [Kaggle: Numenta Anomaly Benchmark (boltzmannbrain/nab)](https://www.kaggle.com/datasets/boltzmannbrain/nab)

**Licence:** NAB is released under the AGPL-3.0 licence by Numenta.

| File | Contents |
|---|---|
| `nab_machine_temp.csv.gz` | 22,695 five-minute temperature readings, 2 Dec 2013 to 19 Feb 2014 |
| `nab_windows.json.gz` | NAB's labelled anomaly windows for every NAB file (4 for this sensor) |

To refresh from Kaggle instead, download the dataset with the Kaggle CLI (`kaggle datasets download <owner/dataset>`),
then gzip the needed files into this folder with the same names.
