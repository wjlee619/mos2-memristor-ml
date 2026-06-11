# Switching Variability in MoS₂/Graphene Bipolar Memristors: Process Implications for 2D Device Integration

**Won Jun Lee (이원준)**  
MRes Soft Electronics, Imperial College London (2DWeb Group, Torrisi Lab)  
[github.com/wjlee619](https://github.com/wjlee619)

---

## Abstract

Resistive switching variability is the central reliability challenge for emerging non-volatile memory based on 2D transition metal dichalcogenides (TMDs). This note summarises electrical characterisation of MoS₂/Graphene bipolar memristors (n = 577 IV sweeps), identifying three independent sources of switching variability: stochastic filament nucleation, progressive electroforming, and electrode geometry. The findings have direct implications for process integration of 2D materials at scaled nodes, where variability in SET voltage, ON/OFF ratio, and retention directly determine memory window yield.

---

## 1. Device Architecture and Measurement Method

Devices are fabricated on SiO₂/Si substrates with bottom electrodes of evaporated Cr/Au, photolithographically patterned in a four-contact cross geometry; a spray-coated MoS₂ switching layer; and a CVD-grown graphene top electrode applied by wet transfer. All electrical measurements were performed on a Keithley 2634B SourceMeter in dual-sweep IV mode (4000 points per sweep, current compliance 10–100 mA).

The switching layer is MoS₂ — a semiconducting TMD with bandgap ~1.8 eV (monolayer) to ~1.2 eV (bulk). Resistive switching in this system is filamentary: conductive paths form and rupture within the MoS₂ under applied electric field, driven by ion migration and local Joule heating.

**Key device numbers:**

| Parameter | Value |
|---|---|
| ON/OFF ratio range | 10² – 10⁷ |
| Best ON/OFF ratio | ~10⁷ (Run 33, Chip#1) |
| SET voltage range | −12.8 V to +19.4 V |
| RESET voltage range | −3.4 V to +1.4 V |
| Hysteresis window | 9.5 V – 18.0 V |
| Noise floor (OFF state) | ~3.66×10⁻⁶ A (instrument limit) |

---

## 2. Bipolar Filamentary Switching Mechanism

The devices exhibit bipolar switching: SET and RESET occur at opposite voltage polarities. This is the expected signature of ionic filament formation and dissolution, distinct from threshold switching or phase-change mechanisms.

**Run 33** (negative polarity): SET at −12.8 V, RESET at −3.4 V, ON/OFF ~1.5×10⁷  
**Run 35** (positive polarity): SET at +19.4 V, RESET at +1.4 V, ON/OFF ~2.6×10⁷

The large SET voltage asymmetry between runs (ΔV_SET = 32.2 V) reflects the stochastic nature of filament nucleation — the critical nucleus forms at a random site within the MoS₂ switching layer, with nucleation probability governed by local field enhancement rather than bulk material properties. This is consistent with the Weibull statistics observed in SiO₂ and HfO₂ RRAM, and represents the same fundamental variability mechanism that limits memory window yield in production RRAM.

**Implication:** V_SET spread of 32 V across two devices from the same chip means that a fixed write voltage cannot reliably SET all devices. In a production memory array, this would require adaptive write-verify schemes — the same challenge faced by NAND flash and RRAM vendors at sub-20nm nodes.

---

## 3. Stochastic Filament Nucleation Window

The IV hysteresis curve contains a diagnostic feature: a region of zero current gradient immediately before the SET transition, where current is rising but switching has not yet occurred. This "nucleation window" — the voltage span between filament initiation and full SET — varies between runs and reflects the stochastic energy barrier for filament formation.

In Run 33, the nucleation window spans approximately 2–3 V before the abrupt SET transition at −12.8 V. In Run 35, the window is broader before SET at +19.4 V. The width of this window is a direct measure of switching stochasticity — devices with tight nucleation windows show more reproducible SET voltages, an important process metric for memory array integration.

---

## 4. Electroforming Effect: The Primary Source of Device Drift

The most significant finding in this dataset is a **progressive increase in ON-state current across repeated measurement runs on a single chip (Chip#14)**.

OLS regression of i_ON vs run number:

- **R² = 0.48, p < 0.001**
- Positive slope: ON-state current increases systematically with each successive voltage sweep

This is the electroforming signature — repeated high-field sweeps progressively enlarge or stabilise the conductive filament, increasing ON-state conductance. Electroforming is well-documented in HfO₂ and TaOₓ RRAM but less studied in 2D TMD systems.

**Process implication:** Electroforming means that device properties measured early in a test sequence are not representative of steady-state device behaviour. In a production flow, this creates a burn-in requirement — devices must be pre-formed before memory window characterisation. The R² = 0.48 means electroforming accounts for ~half the observed ON-state variance in this chip; the remainder is attributed to local filament geometry variation.

---

## 5. Electrode Width Effect on Yield

Electrode width was extracted from device position codes encoded in measurement filenames. The Aug 2024 dataset (85 files) contains devices with electrode widths ranging from 2 µm to 10+ µm.

**Findings:**

| Electrode Width | Switching Outcome |
|---|---|
| 2 µm | Complete failure — no switching observed |
| 6 µm | Marginal — switching present but inconsistent |
| 10+ µm | Reliable switching |

The complete failure at 2 µm is consistent with a **critical dimension (CD) threshold** for filament formation: below a minimum electrode area, the current density required to nucleate a filament cannot be sustained without exceeding the compliance limit or causing device destruction.

**Spatial mapping** of the Aug 2024 chip shows a non-uniform distribution of switching outcomes across the substrate, consistent with MoS₂ spray-coating thickness gradients across the substrate. Devices at substrate edges show systematically lower ON/OFF ratios than centre devices.

**Metrology implication:** This is a direct demonstration that optical or AFM measurement of electrode CD predicts switching yield — a finding directly relevant to in-line metrology at the lithography step. The 2 µm failure threshold defines the process window lower bound for this material system.

---

## 6. Dataset Limitations and Scope

This dataset is a research characterisation dataset, not a process qualification dataset. Key limitations:

- **Single lab run** — electrode geometry not systematically varied across all chips in a controlled DOE
- **Noise floor** at ~3.66×10⁻⁶ A limits OFF-state resolution; reported ON/OFF ratios for low-current devices are instrument-limited, not physical
- **Electroforming analysis** is single-chip (Chip#14); cross-chip generalisation requires further data
- **Layer count** (10–60 layers, ±10 V sub-threshold two-terminal sweep data) shows no predictive power for low-bias conduction ratio (Random Forest R² = −0.09) — these sweeps stay below the SET threshold and do not probe switching; layer count is too coarse a metric to predict sub-threshold conduction in this system
- **Raw IV traces** not included due to institutional data restrictions; processed features available in `data/processed/`

---

## 7. Relevance to Semiconductor Process Integration

MoS₂ and related TMDs (WSe₂, MoTe₂) are under active investigation for integration into back-end-of-line (BEOL) compatible memory at sub-10nm nodes, where silicon-based floating-gate memory faces fundamental tunnelling and variability limits. The IEDM 2025 programme included multiple papers on TMD RRAM integration from TSMC, Intel, and imec.

The variability sources identified in this work — stochastic filament nucleation, electroforming, and electrode CD dependence — are the same variability sources that limit production RRAM yield at 22nm and below. Characterisation methods developed here (log-domain SET/RESET detection, spatial yield mapping, run-to-run trend analysis) are directly applicable to process development and yield engineering workflows for 2D memory integration.

---

## 8. Measurement Algorithm: Log-Domain SET/RESET Detection

Standard dI/dV fails for memristor IV data due to the 10⁷ dynamic range between OFF (~10⁻¹² A) and ON (~10⁻⁵ A) states. The detection algorithm used here operates in log-space:

```python
log_i      = np.log10(np.abs(current))
grad_logI  = np.gradient(log_i) / np.gradient(np.abs(voltage))  # decades/V
smoothed   = np.convolve(grad_logI, np.ones(20)/20, mode='same')

set_idx    = np.argmax(smoothed)   # maximum positive slope → SET
reset_idx  = np.argmin(smoothed)   # maximum negative slope → RESET
```

This compresses the 10⁷ dynamic range into 7 log-decades, making SET and RESET transitions equally detectable regardless of absolute current magnitude. The 20-point moving average suppresses instrument noise without smearing the transition location.

---

*Data collected at Imperial College London, 2DWeb Group, Torrisi Lab (2023–2024). Measurement instrument: Keithley 2634B SourceMeter. Fabrication in collaboration with PhD researcher Shanglong.*

*Full analysis code and processed datasets: [github.com/wjlee619/mos2-memristor-ml](https://github.com/wjlee619/mos2-memristor-ml)*
