# Device Physics Notes — MoS₂ Printed Memristor
**Won Jun Lee | Imperial College London | Torrisi Lab 2024**

---

## 1. Device Architecture

The device is a printed MoS₂/Graphene bipolar memristor fabricated on a SiO₂ substrate.

**Layer stack (bottom to top):**
- **Substrate:** SiO₂ (thermally grown on Si, provides electrical isolation)
- **Bottom electrode:** Au/Cr, deposited by evaporation and photolithographically patterned. Four contacts per device arranged in a cross geometry, numbered 1–4 clockwise from top. The Au provides low-resistance ohmic contact; Cr is the adhesion layer.
- **Switching layer:** MoS₂ ink spray-coated over the bottom electrodes. The MoS₂ is solution-processed, which means it contains a mixture of single- and few-layer flakes with a disordered grain boundary network, residual solvent, and a non-uniform distribution of sulfur vacancies. Same nominal layer count across all Chip#14 devices (same spray pass), but local thickness varies due to spray pattern geometry (see Section 7).
- **Top electrode:** CVD-grown graphene, wet-transferred. Provides the counter-electrode for current injection into the MoS₂.

**Measurement instrument:** Keithley 2634B dual-channel SMU. Dual-sweep mode enabled — each measurement file contains a full out-and-back voltage sweep in a single CSV. Header is ~62 rows of instrument metadata; data begins at the `"Index"` row. The last 5 rows of each file are summary statistics (Min, Max, Mean, StdDev, CV) — these must be dropped before analysis.

**Channel configurations (T-codes):**
- **T12:** Probes contacts 1 → 2. Short channel — may remain largely within the graphene electrode region rather than fully crossing the MoS₂ switching layer. This means T12 might measure primarily graphene sheet resistance plus contact resistance, with the MoS₂ playing a secondary role.
- **T14:** Probes contacts 1 → 4 — the long diagonal. This crosses the full device width, traversing the MoS₂ switching layer between the two non-adjacent contacts. **T14 is the most physically meaningful measurement** for memristive switching, as confirmed by Run 33 and Run 35 (the only two switching events captured).
- **T24:** Probes contacts 2 → 4. Crosses the full device similarly to T14 but between a different contact pair.

**Critical note on T-code interpretation:** T-code comparisons must account for whether the measurement path actually traverses the switching material. High i_on in T12 could reflect graphene conductance rather than MoS₂ filament conduction. Until T12 is confirmed to show hysteresis loops (switching), its current values are not evidence of memristive behaviour.

**Electrode widths** (encoded in the position code second letter):
- A = 18 μm
- B = 12 μm
- C = 6 μm
- D = 2 μm

The width refers to the electrode linewidth at the point where it contacts the MoS₂ switching layer. Smaller widths reduce contact area and increase the current density required for filament nucleation.

---

## 2. Switching Mechanism — Bipolar Filamentary

### What "bipolar" means

The device switches under opposite voltage polarities: SET (transition from high-resistance to low-resistance state) occurs under one polarity, RESET (transition back) under the opposite polarity. This is confirmed by Run 33 (SET at −12.83 V) and Run 35 (SET at +19.41 V) — the same device, same contact pair (CC1, T14), measured sequentially.

Bipolar operation rules out trap-filling mechanisms, which are inherently unipolar (they rely on filling available trap states and reset when the field is removed, with no polarity dependence). Bipolar switching is the signature of **ion migration**: a mobile species drifts along the electric field to form or dissolve a conductive path.

### The filament

A conductive path — the filament — forms through the MoS₂ switching layer between the two probed contacts. The most likely mechanism is **sulfur vacancy drift** under the applied electric field.

Sulfur vacancies (V_S) are the dominant point defect in solution-processed MoS₂. They arise naturally during spray-coating because the solvent evaporation process is rapid and non-equilibrium, leaving behind a defect-rich lattice. Under a high electric field (~1 MV/cm in a 10–20 nm thick MoS₂ layer), V_S are mobile enough to drift and accumulate, forming a locally metallic region connecting the two electrodes.

The filament is metallic because a high concentration of sulfur vacancies dopes the MoS₂ heavily n-type (each V_S donates two electrons), pushing the local carrier density above the metal-insulator transition threshold. Current through the filament is ohmic (I ∝ V) rather than space-charge-limited.

### Why the ON/OFF ratio is ~10⁶

- **OFF state:** Current flows only through leakage paths across the disordered MoS₂ film — through tunnelling between flakes, along grain boundaries, and through surface-adsorbed moisture. Limited by the defect density and film morphology. Measured at ~10–15 pA near 0 V.
- **ON state:** Current flows through the metallic filament — orders of magnitude more conductive than the surrounding semiconductor matrix. Measured at 9–16 μA for Chip#1, CC1-T14.
- **Ratio:** ~10⁶. This is a lower bound — it is limited by instrument noise floor (Keithley 2634B minimum current range is ~1 fA, but at long cables and ambient conditions the practical floor is ~10 pA).

### Ohmic vs. memristive conduction

A device showing only high current with no hysteresis is resistive (ohmic) — it is simply a resistor, and current is determined entirely by the instantaneous applied voltage and the (fixed) resistance. A memristor shows **history-dependent resistance**: the current at a given voltage depends on the previous state of the device — specifically, whether a conductive filament has been formed.

The proof of memristive behaviour is the **hysteresis loop**: the sweep-out IV curve differs from the sweep-back curve. In the forward direction, the device is in the OFF state (low current); after the SET event, the return sweep shows high current at voltages below V_SET. The area enclosed by the loop represents the energy stored in the ionic configuration of the device.

**All 85 measurements in the Aug 30 Chip#14 batch show `switching_state = already_on`** — no hysteresis was captured because no sweep started from the OFF state. The only direct evidence of memristive switching in this dataset is from Chip#1, CC1-T14, Runs 33 and 35 (Aug 15 batch).

---

## 3. Stochastic Filament Nucleation — Run 33 Detail

This is the most physically important finding from the raw IV data. The raw CSV was inspected row-by-row to characterise the switching transition at the point-by-point level.

### What was observed

During Run 33 (negative dual sweep, 0 → −20 V → 0 V, step size −0.01 V, 3999 data points), the switching transition was not a single abrupt step. Instead:

| Voltage range | Current | Physical process |
|---|---|---|
| 0 to −12.0 V | ~1–6 pA flat | Leakage only — field too low to drive ion migration |
| −12.0 to −12.87 V | 5–30 nA (rising) | Precursor current — carriers filling shallow trap states; field approaching nucleation threshold |
| −12.86 V (row 1286) | Spike: pA → 352 nA | **First switching burst — filament nucleates** |
| −12.88 to −13.06 V | Drops back to ~15 nA | **Filament collapses** — nucleus too small to sustain, thermal fluctuations break it |
| −13.07 to −13.51 V | Repeated bursts: 280–880 nA | **Renucleation ×3** — filament tries multiple competing nucleation sites |
| −14.0 to −18.0 V | 0.4–6.2 μA (rising) | Filament growing along stabilised path, widening |
| −19.0 to −20.0 V | 11–18 μA | Fully formed, stable filament |

### Physical interpretation

This is **stochastic filament nucleation**. The conductive path forms, collapses, and reforms multiple times before finding the lowest-energy stable configuration. The mechanism:

1. The electric field at −12.5 V is sufficient to begin drifting sulfur vacancies toward the nucleation zone.
2. A locally high-density V_S cluster nucleates a small conducting patch — current jumps.
3. Joule heating from the transient current pulse is insufficient to sustain the filament. The thermal energy at room temperature breaks the nascent nucleus — current drops.
4. The field continues to increase. Three more nucleation-collapse cycles occur at slightly different defect clusters in the −13 V range.
5. At −14 V, sufficient vacancy accumulation has occurred that the filament length is long enough to be thermally stable. It then grows monotonically as the voltage increases.

This stochastic behaviour arises because **solution-processed MoS₂ has a disordered defect landscape** — sulfur vacancies are distributed non-uniformly across the film, clustered near grain boundaries and flake edges. Multiple vacancy clusters compete as nucleation sites. The filament attempts each candidate site in sequence until it finds a path with sufficient vacancy density to sustain ohmic conduction.

### Why this matters for device engineering

Stochastic nucleation causes **cycle-to-cycle variability in V_SET**. Each switching event may nucleate at a different defect cluster, giving a different threshold voltage. If the device is cycled 100 times, V_SET will scatter over a distribution — the width of that distribution is a direct measure of the disorder in the MoS₂ defect landscape.

This is one of the core reliability challenges in memristor development for neuromorphic computing. A synaptic weight stored as a resistance state must be programmable to a specific value reproducibly — stochastic nucleation undermines this.

### The ALD connection

A uniform, chemically controlled interface — achievable via surface functionalisation of the MoS₂ before ALD deposition of a conformal dielectric — would introduce **controlled nucleation sites** rather than relying on the random defect distribution from spray-coating. With deliberate nucleation site engineering, the filament would always form at the same location and through the same defect pathway, making V_SET reproducible.

The Run 33 measurement quantifies the problem that interface engineering solves. The stochastic nucleation window (−12.0 V to −13.5 V, ~1.5 V width) is the experimental target for ALD-enabled V_SET stabilisation.

---

## 4. V_SET Asymmetry — Run 33 vs Run 35

### Observation

- **Run 33 (negative sweep):** V_SET = −12.83 V
- **Run 35 (positive sweep):** V_SET = +19.41 V
- Same device (Chip#1), same contact pair (CC1, T14), same ambient conditions, consecutive measurements

The voltage required to switch the device under positive bias is **6.6 V higher** than under negative bias. This is a large asymmetry.

### Physical interpretation

The filament nucleation site is **asymmetrically located** between contacts 1 and 4. Under negative bias (field directed from contact 4 toward contact 1), the electric field drives sulfur vacancies toward a nucleation-favourable region efficiently — the threshold is lower. Under positive bias, vacancies must drift in the opposite direction, either encountering a higher-energy nucleation path or a region with lower V_S density.

This asymmetry is a **fabrication fingerprint**. In a printed device, the ink deposition is never perfectly symmetric about the channel midpoint:
- MoS₂ flake orientation and stacking varies across the channel
- The spray-coating trajectory deposits more material in one half of the channel depending on the nozzle sweep direction
- Au/Cr contact morphology may differ slightly between contacts 1 and 4 due to non-uniformity in the evaporation or photolithographic patterning

These asymmetries create a **preferred polarity for filament growth**: lower V_SET under the polarity that aligns the field with the dominant V_S drift direction.

### Switching character difference

Beyond the threshold asymmetry, the two runs differ qualitatively:

- **Run 33 (negative):** Stochastic nucleation over a 1.5 V window (−12.0 to −13.5 V), three nucleation-collapse cycles before stabilisation. This suggests the field is marginal and multiple sites compete.
- **Run 35 (positive):** Clean single-step SET — current jumps ~3× in one data point (row 1941: 2.93 μA → row 1942: 9.23 μA) and immediately stabilises. This suggests a strongly preferred nucleation site that triggers cleanly once the threshold is reached.

The cleaner switching under positive bias — despite requiring higher voltage — may reflect that the positive-polarity filament path runs through a region of higher V_S density that, once threshold is exceeded, forms rapidly and stably.

### Practical implications

For a neuromorphic computing application:
- Programming under negative bias requires more voltage cycles to reach a stable state (stochastic nucleation).
- Programming under positive bias is more deterministic but needs higher voltage.
- The device is not symmetric in its programming characteristics — directional asymmetry must be accounted for in the circuit design.
- Asymmetric switching has been proposed as a feature for directional synaptic plasticity (spike-timing-dependent plasticity) models where pre-synaptic and post-synaptic signals are distinguishable.

---

## 5. Electroforming Effect — Chip#14

### What electroforming is

The progressive improvement of switching performance with repeated cycling. Each switching event slightly modifies the defect microstructure of the MoS₂:
- Sulfur vacancies migrate and accumulate along the filament path
- The preferred conduction pathway becomes lower-resistance with each pass
- V_SET decreases, I_ON increases, cycle-to-cycle variability decreases

This is a common observation in oxide-based memristors and reflects the device "finding" its optimal filament configuration over the first N cycles (typically N = 10–100 for solution-processed materials).

### What the data shows

The Aug 30 batch (Chip#14, 85 measurements) shows:
- R² = 0.48 on i_on vs run number trend (p < 0.001) — statistically significant positive correlation
- All 85 measurements carry `switching_state = already_on` — by the time of this measurement session, the device was sitting in its ON state permanently
- The upward trend in i_on confirms the filament is widening and becoming more conductive with cumulative cycling

### Why all Aug 30 devices were `already_on`

By the time the Aug 30 measurement session began, Chip#14 had been cycled enough times that devices were sitting in their ON state permanently — the filament had stabilised to the point where thermal fluctuations at room temperature (k_B T ≈ 26 meV) were insufficient to disrupt it. This is **over-electroforming**: the device has passed through the useful electroforming window and is now stuck in a permanent ON state.

Consequence for the dataset: the Aug 30 batch is useful for studying ON-state properties (I_ON magnitude, electrode width scaling, spatial uniformity) but cannot be used to study switching dynamics (V_SET, V_RESET, hysteresis). Those quantities require measurements starting from a reset (OFF) state, which was only achieved in the Aug 15 batch (Runs 33 and 35).

---

## 6. Electrode Width Effect

### Finding (from data/derived/memeffect_sweep_aug30_parsed.csv, n=85)

| Electrode width | n | Mean I_ON | Distribution |
|---|---|---|---|
| 18 μm | 30 | 9.44 mA¹ | Tight, unimodal |
| 12 μm | 18 | 5.89 mA | Tight, unimodal |
| 6 μm | 35 | 2.22 mA | **Bimodal — misleading mean** |
| 2 μm | 2 | 2.55 pA | Leakage only, never switched |

¹ **18 μm mean distinction:** 9.44 mA is the grand mean across all 30 devices (full spatial spread). The NB04 cell text reports 10.49 mA, which is the mean for Row F devices only — a spatial subset that may be concentrated in a higher-coverage region of the spray-coated film. The two values are not contradictory; they reflect different population scopes. Use 9.44 mA for chip-level comparisons and 10.49 mA only when discussing Row F specifically.

### Physical mechanism

**Wider electrode → larger contact area → lower contact resistance → higher current through filament once formed.**

The ON-state current is determined by the series resistance of: (filament resistance) + (MoS₂-electrode contact resistance) + (metal contact resistance). As electrode width increases, the contact area increases proportionally, reducing the contact resistance contribution. The filament resistance itself is geometry-independent (it is a nanoscale path), so the width scaling reflects primarily the contact resistance term.

**At 2 μm:** The electrode width is comparable to or smaller than individual MoS₂ flake sizes in the printed film (flakes from solution processing are typically 100 nm to 5 μm lateral size). The contact area is so small that:
- **Edge effects dominate:** the proportion of the electrode perimeter (where defects and grain boundaries concentrate) relative to total area is maximised at small widths.
- **Field non-uniformity:** the electric field at a narrow contact tip is highly non-uniform, concentrating near the corners. This disrupts the uniform vacancy drift needed for filament nucleation along a well-defined path.
- **Result:** zero switching events at 2 μm across all measured devices.

### The 6 μm bimodality

The 6 μm electrode width sits at a critical threshold. Two populations coexist:
- **~26 devices at mA range** (1–4 mA): electrode width is sufficient to support filament formation. These devices are in their ON state.
- **~9 devices at pA range** (~2–3 pA): electrode too narrow to support switching **or** positioned over a low-defect region of the MoS₂ film where V_S density is insufficient for nucleation.

The mean (2.22 mA) is not physically meaningful — it averages over two fundamentally different populations. The bimodal distribution is the key finding, not the mean.

**Minimum reliable electrode width: 12 μm.** Below this, device yield drops significantly. The 12 μm devices show tight, unimodal I_ON distribution, indicating consistent filament formation across the measured population.

---

## 7. Spray Coating Non-uniformity Hypothesis

### The fabrication process

MoS₂ ink was spray-coated in a top-left to bottom-right zigzag pattern across the chip. The nozzle traverses the chip multiple times; central positions receive more coating passes than edge positions:
- **Central positions:** thicker MoS₂ layer, higher V_S density, more nucleation sites available
- **Edge positions:** thinner MoS₂ layer, fewer nucleation sites

### What the data suggests

For 6 μm devices (the most sensitive to position effects, being at the reliability threshold):

| Position | Switched | Not switched | Yield |
|---|---|---|---|
| FC4 (column 4) | 22 | 2 | **92%** |
| FC1 (column 1) | 1 | 10 | **9%** |

The yield difference is dramatic: 92% vs 9%. Column 4 devices switch reliably; column 1 devices almost never switch.

### The confound

The T-code was not controlled independently of position in this dataset:
- Most T12 measurements were on FC4 devices
- Most T24 measurements were on FC1 devices
- Every T24 measurement is at column 1; column 4 is entirely T12

This means the **column position effect and the T-code (channel configuration) effect are fully confounded**. Both hypotheses are consistent with the data:

1. **Spray coating hypothesis:** Column 1 has thinner MoS₂ → insufficient V_S → no switching, regardless of T-code.
2. **Channel configuration hypothesis:** T24 probes a path that does not efficiently traverse the switching region → no switching, regardless of position.

The existing dataset cannot distinguish between these two causes.

### What the next experiment would be

Measure T12 and T24 on **both** FC1 and FC4 devices systematically:
- If FC1 devices fail regardless of T-code → **position (spray coating)** is the cause
- If T24 devices fail regardless of position → **channel configuration** is the cause
- If FC1-T12 switches but FC1-T24 does not → **both effects are real**

Specifically: measure FC4-T24 and FC1-T12. These are the missing cells in the experimental matrix. Until this is done, the yield difference cannot be attributed to a single physical cause.

This is the fundamental limitation of the current dataset — the experimental design did not control for both variables simultaneously.

---

## 8. What the MATLAB File Likely Contains

A PhD researcher measured devices in a closed vacuum environment. Vacuum measurements differ from ambient-air measurements in physically important ways:

**No surface moisture:**
At ambient conditions, MoS₂ surfaces adsorb a monolayer of water from the atmosphere. This water layer creates parasitic conduction paths along the MoS₂ surface in parallel with the through-film filament. In vacuum, this layer is removed by pumping, giving cleaner device characteristics: lower leakage current, better-defined I_OFF, higher ON/OFF ratio.

**No oxygen:**
Atmospheric oxygen can partially oxidise sulfur vacancies (V_S + O → MoO_x at the defect site), effectively pinning them and reducing their mobility under applied field. In vacuum, V_S remain unpassivated and more mobile, potentially lowering V_SET.

**Expected result:** Vacuum measurements likely show:
- Lower V_SET (more mobile vacancies)
- Lower I_OFF (no moisture leakage)
- Better ON/OFF ratio (both effects combine)
- Less stochastic nucleation (fewer competing surface pathways)
- Potentially more reproducible V_SET distribution (lower cycle-to-cycle variability)

The MATLAB file likely contains full IV curves from vacuum measurements, possibly at multiple pressures or temperatures. Opening it: `scipy.io.loadmat('filename.mat')` — the returned dict will have variable names as keys. Common structure is nested MATLAB structs accessible as numpy void arrays.

---

## 9. What This Dataset Cannot Tell You

**No controlled reset measurements in Aug 30 batch:**
All 85 Aug 30 measurements started from the ON state (`already_on`). Cannot extract V_RESET, hysteresis window, or switching dynamics for Chip#14.

**Only 2 switching events total (Run 33, Run 35):**
Statistically insufficient for a V_SET distribution. Cannot compute mean V_SET, standard deviation, or cycle-to-cycle variability from two data points. The stochastic nucleation observed in Run 33 suggests high variability, but this is a qualitative inference, not a measured distribution.

**T-code fully confounded with position:**
Cannot separate channel length / configuration effects from spatial position (spray coating) effects in the 6 μm failure analysis. The next experiment is specified in Section 7.

**No Raman at device positions:**
Cannot correlate material quality (measured by Raman E₂g and A₁g peak widths, which track disorder and doping) with electrical performance at the individual device level. Raman mapping across the chip would directly test the spray coating hypothesis.

**Spray coating non-uniformity is a hypothesis:**
The column-position yield difference is strongly suggestive, but has not been confirmed by direct thickness measurement (e.g. AFM cross-section or ellipsometry) at FC1 vs FC4 positions.

**No cross-section TEM:**
Cannot confirm whether the filament is truly a nanoscale metallic path through the MoS₂ or whether switching occurs at the MoS₂/electrode interface (interface-type switching vs. bulk-filament switching). TEM on a switched device would resolve this.

---

## 10. Key Numbers to Remember

All values confirmed against source data files unless marked with *.

| Parameter | Value | Source |
|---|---|---|
| ON/OFF ratio (Run 33) | ~10⁶ | data/processed/memeffect_sweep.csv |
| ON/OFF ratio (Run 35) | ~8.7×10⁵ | data/processed/memeffect_sweep.csv |
| V_SET (Run 33) | −12.83 V | data/processed/memeffect_sweep.csv |
| V_SET (Run 35) | +19.41 V | data/processed/memeffect_sweep.csv |
| V_RESET (Run 33) | −3.36 V | data/processed/memeffect_sweep.csv |
| V_RESET (Run 35) | +1.42 V | data/processed/memeffect_sweep.csv |
| Hysteresis window (Run 33) | 9.46 V | data/processed/memeffect_sweep.csv |
| Hysteresis window (Run 35) | 17.99 V | data/processed/memeffect_sweep.csv |
| Stochastic nucleation window | −12.0 to −13.5 V | Raw CSV, Run 33, rows 1256–1350 |
| Nucleation-collapse cycles | 3 | Raw CSV, Run 33, rows 1286–1313 |
| 18 μm mean I_ON | 9.44 mA | data/derived/memeffect_sweep_aug30_parsed.csv (n=30) |
| 12 μm mean I_ON | 5.89 mA | data/derived/memeffect_sweep_aug30_parsed.csv (n=18) |
| 6 μm mean I_ON | 2.22 mA | data/derived/memeffect_sweep_aug30_parsed.csv (n=35) — bimodal, mean is misleading |
| 2 μm mean I_ON | 2.55 pA | data/derived/memeffect_sweep_aug30_parsed.csv (n=2) |
| Minimum reliable electrode width | 12 μm | Electrode width analysis (NB04) |
| 6 μm FC4 yield | 92% (22/24) | data/derived/df6_enriched.csv |
| 6 μm FC1 yield | 9% (1/11) | data/derived/df6_enriched.csv |
| Electroforming R² | 0.48, p<0.001* | Aug 30 batch trend |
| Total aug30 devices | 85 | data/processed/memeffect_sweep_aug30.csv |
| Sensors (SECOM fab-drill) | 590 total* | fab-drill project |
| Zero-variance sensors | 116 (20%)* | fab-drill project |

*Not directly verified in this session's analysis scripts. Treat as provisional until confirmed against source.

---

*Last updated: 2026-05-04*
