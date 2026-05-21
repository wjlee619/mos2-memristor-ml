# MoS₂/Graphene Printed Memristor — Electrical Characterisation & ML Analysis

> Data pipeline and ML analysis for inkjet-printed MoS₂/Graphene resistive switching devices.  
> Characterised on Keithley 2634B SMU · Imperial College London (2DWeb Group, Torrisi Lab) · 2023–2024

**Won Jun Lee (이원준)**  
MRes Soft Electronics, Imperial College London  
[github.com/wjlee619](https://github.com/wjlee619)

---

## Motivation

Resistive switching in 2D transition metal dichalcogenides (TMDs) is a candidate mechanism for non-volatile memory and neuromorphic compute elements. Understanding **what controls switching variability** — electrode geometry, layer count, measurement history — is directly relevant to process integration of MoS₂ as a channel or switching layer in scaled devices.

This project answers three questions from real experimental data:

1. Does layer count predict switching performance? (**No — R² = −0.09**)
2. Is there a systematic drift in device behaviour across repeated cycles? (**Yes — electroforming confirmed, R² = 0.48, p < 0.001**)
3. Does optical illumination affect ON-state stability? (**No — p = 0.25**)

---

## Device Summary

| Parameter | Value |
|---|---|
| Material system | MoS₂ / Graphene (inkjet-printed) |
| Device type | Bipolar memristor |
| Measurement instrument | Keithley 2634B SourceMeter |
| Total raw files processed | 577 CSV files |
| ON/OFF ratio (best devices) | ~10⁷ |
| SET voltage range | −12.8 V to +19.4 V |
| Electroforming signature | R² = 0.48, p < 0.001 (Chip#14) |

---

## Key Results

| Analysis | Method | Finding |
|---|---|---|
| Layer count vs ON/OFF ratio | Pearson r, Random Forest | r < 0.25; R² = −0.09 — layer count is not predictive |
| ON-state stability (Chip#14) | OLS regression, run index | R² = 0.48, p < 0.001 — progressive electroforming confirmed |
| Light vs dark condition | Mann-Whitney U | p = 0.25 — no photoconductive effect on stable ON state |
| Noise floor artefact | EDA / correlation audit | i_on ↔ on/off ratio (r = 0.96) is instrument-limited, not physical |

---

## Signal Processing: SET/RESET Detection

Raw IV sweeps span ~10⁷ dynamic range (OFF: ~10⁻¹² A → ON: ~10⁻⁵ A).  
Standard dI/dV fails near the ON state due to noise saturation.

**Algorithm: d(log₁₀|I|)/dV**

```python
log_i      = np.log10(np.abs(current))
grad_logI  = np.gradient(log_i) / np.gradient(np.abs(voltage))  # decades/V
smoothed   = np.convolve(grad_logI, np.ones(20)/20, mode='same')

set_idx    = np.argmax(smoothed)   # steepest positive slope → SET
reset_idx  = np.argmin(smoothed)   # steepest negative slope → RESET
```

This approach compresses the 10⁷ dynamic range into 7 log-decades, making both transitions equally detectable regardless of absolute current level.

---

## Repository Structure

```
mos2-memristor-ml/
├── data/processed/
│   ├── layer_sweep.csv               # Gate sweep features (73 files)
│   ├── memeffect_sweep.csv           # IV sweep features — SET/RESET (39 files)
│   └── memeffect_sweep_aug30.csv     # Aug 2024 batch (85 files)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Feature distributions, correlation audit, data quality
│   ├── 02_random_forest.ipynb        # Layer count predictability (R² = −0.09)
│   └── 03_stability_analysis.ipynb   # Electroforming trend, light condition test
│
├── scripts/
│   ├── process_layer_sweep.py        # Gate sweep feature extraction
│   └── extract_memeffect_iv.py       # IV sweep SET/RESET detection algorithm
│
├── results/figures/                  # All output figures
├── docs/mos2_project_notes.md        # Analysis log and methodology notes
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/wjlee619/mos2-memristor-ml.git
cd mos2-memristor-ml
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

---

## Data Schema

**`layer_sweep.csv`** — Gate sweep characterisation

| Column | Description |
|---|---|
| `layers` | MoS₂ layer count (10–60) |
| `id_on_A` | ON-state drain current (A) |
| `id_off_A` | OFF-state drain current (A) — noise floor ~3.66×10⁻⁶ A |
| `on_off_ratio` | id_on / id_off |
| `vgs_min_V`, `vgs_max_V` | Gate sweep voltage range (V) |

**`memeffect_sweep.csv`** — IV switching characterisation

| Column | Description |
|---|---|
| `switching_state` | `switched` / `already_on` / `low_voltage_sweep` |
| `v_set_V` | SET voltage (V) |
| `v_reset_V` | RESET voltage (V) |
| `i_on_A` | ON-state current (A) |
| `on_off_ratio` | i_on / i_off |
| `hysteresis_window_V` | \|v_set – v_reset\| (V) |

---

## Limitations

- Dataset originates from a single lab run; electrode geometry was not systematically varied across all chips
- Noise floor at ~3.66×10⁻⁶ A constrains OFF-state measurement resolution for high-resistance devices
- Electroforming analysis (Chip#14) is single-chip; cross-chip generalisation requires further data
- Raw CSV files not included (institutional data — available on request)

---

## Research Context

Data collected at **Imperial College London** (2DWeb Group, Torrisi Lab, MSRH building) using a Keithley 2634B SourceMeter. Fabrication and initial characterisation in collaboration with PhD researcher Shanglong (2024).

MoS₂ as a switching material is under active investigation for integration into back-end-of-line (BEOL) compatible memory and in-memory compute architectures, relevant to sub-10nm node scaling constraints where silicon-based memory faces tunnelling and variability limits.

---

## License

MIT — see `LICENSE`
