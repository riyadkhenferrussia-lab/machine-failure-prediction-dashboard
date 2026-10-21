# Data

`raw/ai4i2020.csv` is the **AI4I 2020 Predictive Maintenance Dataset**.

- Author: Stephan Matzka. Source: UCI Machine Learning Repository, dataset 601,
  <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>, DOI 10.24432/C5HS5C.
- Licence: CC BY 4.0. Redistributed unchanged with attribution.
- Synthetic data that the author generated to reflect real predictive-maintenance data; 10,000 rows, 14 columns.
- SHA-256 of the file in this repository: `59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183` (512,047 bytes).

**Provenance of this copy.** The UCI download could not be reached from the build environment, so the
file was taken from a public GitHub mirror
(<https://github.com/lingchenlijing/AI4I-2020-Predictive-Maintenance-Dataset>). It matches the published
dataset description on every checked statistic (row count, 339 failures, the five failure-mode counts,
Type split 6000/2997/1003) and the documented PWF/HDF/OSF criteria reproduce those labels for 100% of rows,
but it has **not been byte-compared with the official UCI file**. To do that:

```bash
python -m src.data --verify-official   # downloads the UCI ZIP in memory and compares hashes
python -m src.data --download --force  # or replace the local file with the official one
```

Citation: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
