# MoS2 Memristor ML Analysis

Machine learning and statistical analysis of printed MoS2/Graphene memristor
devices fabricated and measured at Imperial College London (2024), Felice Torrisi
Lab. Raw IV curves collected using a Keithley 2634B SourceMeter across 577 device
measurements spanning multiple chips, electrode geometries, and measurement
conditions.

## Key Findings

| Analysis | Finding |
|---|---|
| Layer count vs on/off ratio | r < 0.25 — no predictive relationship |
| Random Forest (layer count) | R² = −0.09 — layer count insufficient |
| Electroforming (Chip#14) | R² = 0.48, p<0.001 — confirmed kinetic trend |
| Electrode width sweep | ON current spans 6 decades: 2μm→18μm |
| Minimum viable width | 2μm devices show pA leakage only — no switching |

## Notebooks

| Notebook | Description |
|---|---|
| 01_eda | Layer distribution, IV curve visualisation |
| 02_random_forest | ML prediction from layer count |
| 03_stability_analysis | Electroforming kinetics, Chip#14 |
| 04_electrode_width | Contact geometry → switching performance |

## Data Pipeline

```
Raw Keithley 2634B CSVs (577 files)
↓
Feature extraction: v_set, v_reset, i_on, i_off per sweep
↓
data/processed/   ← cleaned feature tables (read-only)
↓
data/derived/     ← position-parsed, enriched tables
↓
notebooks/        ← analysis and visualisation
```

## Project Structure

```
mos2-memristor-ml/
├── data/
│   ├── processed/     ← extracted features (read-only)
│   └── derived/       ← enriched, position-parsed tables
├── notebooks/         ← analysis notebooks
├── scripts/           ← extraction and parsing scripts
├── results/figures/   ← all output figures
└── docs/              ← analysis log and personal notes
```

## Author

Won Jun Lee (이원준)
MRes Soft Electronics, Imperial College London
Keithley 2634B measurements, Torrisi Lab / 2DWeb Group, 2024
[github.com/wjlee619](https://github.com/wjlee619)
