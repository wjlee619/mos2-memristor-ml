# MoS₂/Graphene Printed Memristor — Electrical Characterisation and Data Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Institution](https://img.shields.io/badge/Imperial_College_London-Torrisi_Lab-navy?style=flat-square)

**Si/SiO₂/Au/Cr/MoS₂/Graphene vertical memristor stack · Keithley 2634B SMU · Imperial College London, Torrisi Lab (2DWeb Group) · 2023–2024**

Won Jun Lee (이원준) · MRes Soft Electronics, Imperial College London · [github.com/wjlee619](https://github.com/wjlee619)

---

## Device Architecture

The device is a solution-processed bipolar memristor fabricated on a Si/SiO₂ substrate. The layer stack from bottom to top:

- **Si/SiO₂** — substrate and electrical isolation layer
- **Au/Cr** — bottom electrodes, inkjet-printed in a four-contact cross geometry (Au for low-resistance ohmic contact; Cr adhesion layer)
- **MoS₂** — switching layer, spray-coated from solution over the bottom electrodes using an automated spray coater
- **Graphene** — top electrode, wet-transferred over the MoS₂ switching layer

Electrode linewidth was varied by design across the chip: **2, 6, 12, and 18 µm** (encoded as position codes A–D in measurement filenames). All electrical measurements were performed on a Keithley 2634B dual-channel SourceMeter in dual-sweep IV mode (4,000 points per sweep).

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/device_stack_fabrication.png" width="700"/>
</p>

---

## Key Specifications

| Parameter | Value |
|:---|:---|
| Device type | Bipolar memristor (filamentary switching) |
| Layer stack | Si/SiO₂ / Au/Cr / MoS₂ (spray-coated) / Graphene (wet transfer) |
| Measurement instrument | Keithley 2634B SourceMeter |
| Total IV sweeps processed | 577 raw CSV files |
| ON/OFF ratio (best devices) | ~10⁷ |
| SET voltage range | −12.8 V to +19.4 V |
| Electroforming trend | R² = 0.48, p < 0.001 (Chip#14, n = 85 sweeps) |
| Minimum reliable electrode CD | 12 µm |

---

## Summary of Findings

Four process-relevant findings are reported from real experimental data:

**1. MoS₂ layer count does not predict switching performance.** Random Forest regression yields R² = −0.09 across 73 gate-sweep measurements. Layer count as measured by optical contrast is too coarse a metric to predict ON/OFF ratio in this system — local interface quality and defect density are the controlling variables.

**2. Repeated measurement cycles produce systematic drift in device behaviour.** ON-state current increases monotonically with run number on Chip#14 (R² = 0.48, p < 0.001), consistent with progressive conductive filament widening — the electroforming signature. Devices in the Aug 2024 batch were permanently in the ON state by the measurement session, indicating over-electroforming prior to characterisation.

**3. Electrode geometry is the primary process-control variable for switching yield.** ON-state current scales monotonically with electrode width. Yield drops sharply below 12 µm: 6 µm devices show bimodal behaviour (FC4 column yield 92%, FC1 column yield 9%); 2 µm devices produce leakage current only with no observable switching events.

**4. Resistance scales with electrode width following a near-ideal sheet resistance law.** Fitted resistance across 52 functional devices follows R ∝ W⁻¹·¹⁵ (Pearson r = −0.9999, log–log), with an extracted sheet resistance of approximately 1800–2000 Ω/□ consistent across all geometry groups. Yield at 2 µm is 50% compared to 71–75% at wider geometries, with two physically distinct failure modes identified: open-circuit dead devices and anomalous high-resistance devices.

---

## Results

### 1 — Resistance Scaling and Yield — Chip#1 (Notebook 07)

![R5 scaling log-log](results/figures/fig_07b_r5_scaling_loglog.png)

![Yield by geometry](results/figures/fig_07a_yield_by_geometry.png)

Fitted resistance (R5) across 52 functional devices follows R ∝ W⁻¹·¹⁵ (Pearson r = −0.9999, log–log), confirming sheet resistance-dominated transport with an extracted R_sheet of approximately 1800–2000 Ω/□. Device yield decreases from 71–75% at wider geometries to 50% at 2 µm, consistent with contact formation failure on the chemically inert MoS₂ basal plane. Two failure modes are identified: dead open-circuit devices (16.2%) attributable to nucleation gaps, and anomalous high-resistance devices (13.5%) consistent with grain boundary barriers. 86.5% of functional devices achieve average fit quality above 0.999, validating the R5 extraction methodology. IA3 (18 µm, R5 = 31 kΩ, CV = 318%) is excluded from annotation and discussed separately as a switching candidate.

---

### 2 — Layer Count vs Switching Performance

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s1_layer_distribution.png" width="700"/>
</p>

ON/OFF ratio across 6 MoS₂ layer groups (10–60 layers). No monotonic trend observed — Pearson r < 0.25, Random Forest R² = −0.09. Layer count is not a viable process control parameter for switching yield in this material system.

---

### 3 — Bipolar IV Hysteresis: SET and RESET

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/00_iv_hysteresis_loops.png" width="700"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s2_iv_curves.png" width="700"/>
</p>

Log-scale IV curves for Run 33 (negative polarity, V_SET = −12.8 V) and Run 35 (positive polarity, V_SET = +19.4 V) — same device, same contact pair (Chip#1, CC1-T14), consecutive measurements. V_SET asymmetry of 32.2 V between polarities reflects the stochastic nature of filament nucleation and asymmetric defect distribution within the MoS₂ switching layer.

---

### 4 — Feature Correlation Audit

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/fig_s3_correlation.png" width="700"/>
</p>

Pearson correlation matrix for gate-sweep features. The strong i_ON ↔ ON/OFF ratio correlation (r = 0.96) is an instrument noise floor artefact: OFF-state current is clamped at ~3.66×10⁻⁶ A by the Keithley measurement floor, compressing the denominator of the ON/OFF ratio. This correlation is not a physical device property.

---

### 5 — Chip-Level ON-State Distribution

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_chip_ion_boxplot.png" width="700"/>
</p>

ON-state current distribution by chip. Chip#14 and Chip#1 show stable mA-range ON states. Chip#6 devices remain in the pA range — consistent with incomplete electroforming or insufficient field to nucleate a stable filament.

---

### 6 — Electroforming Trend

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_run_ion_trend.png" width="700"/>
</p>

ON-state current vs measurement run number for Chip#14 (n = 85 sweeps). Statistically significant positive trend — R² = 0.48, p < 0.001 — consistent with progressive conductive filament stabilisation across repeated voltage cycles. Electroforming accounts for approximately half of the observed ON-state variance in this chip.

---

### 7 — Optical Illumination: No Effect on Stable ON State

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/03_lightcond_ion_boxplot.png" width="700"/>
</p>

ON-state current under Dark vs Dark-ThenLight measurement conditions. Mann-Whitney U test: p = 0.25. No statistically significant photoconductive effect on devices in a stable ON state.

---

### 8 — Electrode Width: Process Window Analysis

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/04_electrode_width_ion.png" width="700"/>
</p>

<p align="center">

| Electrode Width | Mean i_ON | n | Result |
|:---:|:---:|:---:|:---|
| 18 µm | 9.44 mA | 30 | Reliable — tight unimodal distribution |
| 12 µm | 5.89 mA | 18 | Reliable — tight unimodal distribution |
| 6 µm | 2.22 mA* | 35 | Marginal — bimodal distribution |
| 2 µm | ~2.55 pA | 2 | Fail — leakage only, no switching |

</p>

*The 6 µm mean averages two distinct populations: ~26 devices at 1–4 mA (filament formed) and ~9 devices at pA-level (no switching). Spatial yield breakdown: FC4 column yield = 92%; FC1 column yield = 9% — consistent with a spray-coating thickness gradient across the substrate.

**Minimum reliable electrode CD: 12 µm.**

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/electrode_width_optical.png" width="700"/>
</p>

Optical microscope images of all four electrode geometries (18, 12, 6, 2 µm) fabricated on the same chip using an automated spray coater.

<p align="center">
  <img src="https://raw.githubusercontent.com/wjlee619/mos2-memristor-ml/main/results/figures/04_spatial_map.png" width="700"/>
</p>

Spatial yield map across Chip#14. Column 4 devices switch reliably; column 1 devices show near-zero yield — consistent with spray-coating non-uniformity producing a MoS₂ thickness gradient across the substrate.

---

## Signal Processing: Log-Domain SET/RESET Detection

Raw IV sweeps span approximately 10⁷ dynamic range (OFF state: ~10⁻¹² A; ON state: ~10⁻⁵ A). Standard dI/dV differentiation fails across this range due to noise saturation near the ON state.

**Algorithm: d(log₁₀|I|)/dV**

```python
log_i     = np.log10(np.abs(current))
grad_logI = np.gradient(log_i) / np.gradient(np.abs(voltage))  # decades/V
smoothed  = np.convolve(grad_logI, np.ones(20)/20, mode='same')

set_idx   = np.argmax(smoothed)   # maximum positive gradient → SET transition
reset_idx = np.argmin(smoothed)   # maximum negative gradient → RESET transition
```

Operating in log-space compresses the 10⁷ dynamic range into 7 log-decades, making SET and RESET transitions equally detectable regardless of absolute current magnitude.

---

## Analysis Pipeline

<p align="center">
  <img src="results/figures/pipeline.png" width="520" alt="Analysis Pipeline"/>
</p>

---

## Repository Structure

```
mos2-memristor-ml/
├── data/
│   ├── processed/
│   │   ├── layer_sweep.csv                   # Gate sweep features (73 files)
│   │   ├── memeffect_sweep.csv               # IV sweep SET/RESET features (39 files)
│   │   └── memeffect_sweep_aug30.csv         # Aug 2024 batch (85 files)
│   ├── derived/
│   │   ├── memeffect_sweep_aug30_parsed.csv  # Electrode width + position parsed
│   │   └── df6_enriched.csv                  # 6 µm device yield drilldown
│   └── processedTable.csv                    # Chip#1 R5 resistance metrics (74 devices)
│
├── notebooks/
│   ├── 01_eda.ipynb                          # Feature distributions, correlation audit
│   ├── 02_random_forest.ipynb                # Layer count predictability (R² = −0.09)
│   ├── 03_stability_analysis.ipynb           # Electroforming trend, light condition test
│   ├── 04_electrode_width_analysis.ipynb     # CD process window, spatial yield map
│   ├── 05_data_quality_audit.ipynb           # Distribution audit, mean vs spread
│   ├── 06_drilldown_analysis.ipynb           # T-code / position yield drilldown
│   └── 07_chip1_resistance_scaling.ipynb     # Chip#1 resistance scaling — R ∝ W⁻¹·¹⁵, yield by geometry, sheet resistance extraction
│
├── scripts/
│   ├── extract_memeffect_iv.py               # Core IV feature extractor
│   ├── process_layer_sweep.py                # Gate sweep feature extraction
│   └── parse_position_aug30.py              # Electrode width + position parsing
│
├── results/figures/                          # All output figures
├── docs/
│   ├── device_physics_notes.md              # Device physics and mechanism notes
│   ├── switching_variability_note.md        # Technical note — process implications
│   └── analysis_log.md                      # Per-notebook analysis log
└── requirements.txt
```

The [`docs/device_physics_notes.md`](docs/device_physics_notes.md) file contains a detailed reference on device physics, switching mechanisms, and measurement methodology developed during the project.

---

## Quickstart

```bash
git clone https://github.com/wjlee619/mos2-memristor-ml.git
cd mos2-memristor-ml
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

---

## Scope and Constraints

- Dataset covers a single fabrication run; a controlled DOE varying electrode geometry, MoS₂ deposition passes, and measurement sequence independently was not performed
- Instrument noise floor (~3.66×10⁻⁶ A) limits OFF-state resolution; ON/OFF ratios for low-current devices are partially instrument-limited
- Only two switching events were captured from the OFF state (Chip#1, Runs 33 and 35); V_SET distribution statistics require a larger cycling dataset
- Column position and T-code (measurement channel) are confounded in the 6 µm yield analysis — a follow-on experiment measuring T12 and T24 at both FC1 and FC4 positions is required to separate spatial from channel effects
- Raw IV trace files are not included (institutional data; available on request)

---

## Research Context

Electrical characterisation performed at **Imperial College London** (2DWeb Group, Torrisi Lab, MSRH) using a Keithley 2634B SourceMeter. Device fabrication and initial characterisation in collaboration with PhD researcher Shanglong (2024).

MoS₂-based resistive switching is under active investigation for back-end-of-line (BEOL) compatible non-volatile memory integration at sub-10nm nodes, where silicon floating-gate memory is constrained by tunnelling oxide scaling limits. The variability mechanisms characterised in this work — stochastic filament nucleation, electroforming, and electrode CD dependence — are directly relevant to process development and yield engineering for 2D TMD memory integration.

---

## License

MIT — see `LICENSE`
