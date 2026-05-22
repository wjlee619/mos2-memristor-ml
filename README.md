# MoS₂/Graphene Printed Memristor — Electrical Characterisation & ML Analysis

> Data pipeline and ML analysis for inkjet-printed MoS₂/Graphene resistive switching devices.  
> Characterised on Keithley 2634B SMU · Imperial College London (2DWeb Group, Torrisi Lab) · 2023–2024

**Won Jun Lee (이원준)** · MRes Soft Electronics, Imperial College London · [github.com/wjlee619](https://github.com/wjlee619)

---

## Hero Figure: Bipolar IV Hysteresis

![IV Hysteresis](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/00_iv_hysteresis_loops.png)

---

## Motivation

Resistive switching in 2D transition metal dichalcogenides (TMDs) is a candidate mechanism for non-volatile memory and neuromorphic compute. Understanding **what controls switching variability** — electrode geometry, layer count, measurement history — is directly relevant to process integration of MoS₂ at scaled nodes.

This project answers three questions from real experimental data:

| Question | Answer |
|---|---|
| Does layer count predict switching performance? | **No — R² = −0.09** |
| Is there systematic drift across repeated cycles? | **Yes — electroforming confirmed (R² = 0.48, p < 0.001)** |
| Does optical illumination affect ON-state stability? | **No — p = 0.25** |

---

## Device Summary

| Parameter | Value |
|---|---|
| Material system | MoS₂ / Graphene (inkjet-printed) |
| Device type | Bipolar memristor |
| Instrument | Keithley 2634B SourceMeter |
| Raw files processed | 577 CSV files |
| ON/OFF ratio (best devices) | ~10⁷ |
| SET voltage range | −12.8 V to +19.4 V |
| Electroforming signature | R² = 0.48, p < 0.001 (Chip#14) |

---

## Results

### 1 — Layer Count Does Not Predict Switching Performance

![Layer Distribution](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s1_layer_distribution.png)

Boxplot of ON/OFF ratio across 6 layer groups (10–60 layers). No monotonic trend — Pearson r < 0.25, Random Forest R² = −0.09. Layer count alone is insufficient as a process control parameter for switching yield.

---

### 2 — IV Curves: SET and RESET Detection

![IV Curves](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s2_iv_curves.png)

Log-scale IV curves for Run 33 (negative polarity, −12.8 V SET) and Run 35 (positive polarity, +19.4 V SET). Classic bipolar memristive hysteresis. ON/OFF ratio ~10⁷ in best devices.

---

### 3 — Correlation Audit: Noise Floor Artefact

![Correlation Heatmap](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s3_correlation.png)

Feature correlation matrix for `layer_sweep` data. Strong i_on ↔ on/off ratio correlation (r = 0.96) is an **instrument noise floor artefact**, not a physical relationship — OFF current is clamped at ~3.66×10⁻⁶ A by the measurement system.

---

### 4 — Chip-Level ON-State Distribution

![Chip Ion Boxplot](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_chip_ion_boxplot.png)

ON-state current by chip batch. Chip#14 and Chip#1 show stable mA-range ON states. Chip#6 (pA range) indicates incomplete SET — a yield failure mode consistent with insufficient electroforming.

---

### 5 — Electroforming Effect (Key Finding)

![Electroforming Trend](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_run_ion_trend.png)

ON-state current vs run number for Chip#14. Significant upward trend — **R² = 0.48, p < 0.001** — consistent with progressive conductive filament formation across repeated voltage sweeps. This is the primary source of device-to-device variability in this dataset.

---

### 6 — Light Condition: No Effect on Stable ON State

![Light Condition](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_lightcond_ion_boxplot.png)

ON-state current under Dark vs Dark-ThenLight conditions. Mann-Whitney U: **p = 0.25** — no significant photoconductive effect once the device is in a stable ON state.

---

### 7 — Electrode Width: Process Window Analysis (Key Finding)

![Electrode Width vs ON Current](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/04_electrode_width_ion.png)

ON-state current scales monotonically with electrode width. Sharp yield cliff below 12 µm — minimum reliable electrode CD = **12 µm**.

| Electrode Width | Mean i_ON | Yield |
|---|---|---|
| 18 µm | 9.44 mA | Reliable |
| 12 µm | 5.89 mA | Reliable |
| 6 µm | 2.22 mA* | Marginal — high variability |
| 2 µm | ~2.55 pA | Fail — leakage only |

*6 µm mean is bimodal — misleading. FC4 yield = 92%, FC1 yield = 9%.

![Spatial Map](https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/04_spatial_map.png)

Spatial yield map across Chip#14. Column 4 devices switch reliably (92%); column 1 devices almost never switch (9%) — consistent with spray-coating thickness gradient across the substrate.

---

## Signal Processing: SET/RESET Detection Algorithm

Raw IV sweeps span ~10⁷ dynamic range (OFF: ~10⁻¹² A → ON: ~10⁻⁵ A). Standard dI/dV fails near the ON state due to noise saturation.

**Solution: d(log₁₀|I|)/dV** — compresses 10⁷ range into 7 log-decades, making both transitions equally detectable.

```python
log_i      = np.log10(np.abs(current))
grad_logI  = np.gradient(log_i) / np.gradient(np.abs(voltage))  # decades/V
smoothed   = np.convolve(grad_logI, np.ones(20)/20, mode='same')

set_idx    = np.argmax(smoothed)   # steepest positive slope → SET
reset_idx  = np.argmin(smoothed)   # steepest negative slope → RESET
```

---

## Repository Structure

```
mos2-memristor-ml/
├── data/processed/
│   ├── layer_sweep.csv               # Gate sweep features (73 files)
│   ├── memeffect_sweep.csv           # IV sweep SET/RESET features (39 files)
│   └── memeffect_sweep_aug30.csv     # Aug 2024 batch (85 files)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Distributions, correlation audit, data quality
│   ├── 02_random_forest.ipynb        # Layer count predictability (R² = −0.09)
│   ├── 03_stability_analysis.ipynb   # Electroforming trend, light condition test
│   ├── 04_electrode_width_analysis.ipynb  # CD process window, spatial yield map
│   ├── 05_data_quality_audit.ipynb   # Distribution audit, mean vs spread
│   └── 06_drilldown_analysis.ipynb   # T-code / position yield drilldown
│
├── scripts/
│   ├── process_layer_sweep.py        # Gate sweep feature extraction
│   ├── extract_memeffect_iv.py       # SET/RESET detection algorithm
│   └── parse_position_aug30.py       # Electrode width + position parsing
│
├── results/figures/                  # All output figures
├── docs/
│   ├── device_physics_notes.md       # Device physics and mechanism notes
│   ├── switching_variability_note.md # Technical note — process implications
│   └── analysis_log.md              # Per-notebook analysis log
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

## Limitations

- Single lab run — electrode geometry not systematically varied across all chips in a controlled DOE
- Noise floor at ~3.66×10⁻⁶ A limits OFF-state resolution for high-resistance devices
- Electroforming analysis is single-chip (Chip#14); cross-chip generalisation requires further data
- T-code fully confounded with column position in the 6 µm yield analysis — cannot separate channel vs spatial effects without follow-on experiment
- Raw CSV files not included (institutional data — available on request)

---

## Research Context

Data collected at **Imperial College London** (2DWeb Group, Torrisi Lab) using a Keithley 2634B SourceMeter. Fabrication in collaboration with PhD researcher Shanglong (2024).

MoS₂ is under active investigation for BEOL-compatible memory and in-memory compute, relevant to sub-10nm node scaling where silicon-based memory faces hard tunnelling and variability limits.

---

## License

MIT — see `LICENSE`
