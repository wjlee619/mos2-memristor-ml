# Project Audit & Remediation Design
**Date:** 2026-06-02  
**Project:** mos2-memristor-ml — Electrical Characterisation and Data Analysis of Printed MoS₂/Graphene Devices  
**Audiences:** MRes examiner (B), peer reviewer (C), portfolio/recruiter  
**Audit method:** Multi-lens structured review — scientific rigour, narrative coherence, code reproducibility, per-audience readability, scope and claims alignment  

---

## Summary Counts

| Severity | Count | Must fix before submission |
|---|---|---|
| 🔴 CRITICAL | 10 | Yes — all |
| 🟡 MAJOR | 17 | Yes — all |
| 🟢 MINOR | 6 | Recommended |
| **Total** | **33** | |

---

## Section 1 — Scientific Rigour

### S1-01 🔴 CRITICAL — Single fabrication run, no replication; declarative conclusions overstated

**Finding:** Every result in this project comes from one chip or one batch per experiment. No chip-to-chip or batch-to-batch replication exists. The Summary of Findings and notebook conclusions use declarative language ("resistance scales", "yield drops", "electroforming confirmed") without consistently attaching this caveat. The R ∝ W⁻¹·¹⁵ result has n=52 devices but they are all from one chip on one deposition run — that is not the same as n=52 independent experiments. A peer reviewer will flag this immediately.

**Affected locations:**
- README: Summary of Findings (all four points)
- Notebook 04: Results table and "Key finding" cell
- Notebook 07: Key Findings table, Physical interpretation cell

**Fix:** Add a single consistent caveat sentence to the Summary of Findings preamble in the README: *"All results are derived from a single fabrication run unless otherwise stated; reproducibility across runs has not been demonstrated."* Add identical language as the opening line of the Key Findings section in notebooks 04 and 07.

---

### S1-02 🔴 CRITICAL — Random Forest model uses experimental parameters as features, undermining the layer count conclusion

**Finding:** Notebook 02 uses Random Forest regression with four features: `layers`, `vgs_min_V`, `vgs_max_V`, `n_points`. Of these, `vgs_min_V`, `vgs_max_V`, and `n_points` are experimenter-set sweep parameters — they are not device material properties. Including them in a model that claims to test whether layer count predicts switching performance is a methodological error. The feature importance chart will show these parameters dominating, which means the model is measuring sensitivity to sweep configuration, not to material properties. The conclusion "layer count does not predict switching performance" is only defensible if the non-material features are excluded or their role is explicitly addressed.

**Affected locations:**
- Notebook 02: Section 1 (Feature Matrix), Section 4 (Feature Importance), Summary cell
- README: Summary of Findings Finding 1

**Fix:**
1. Add a markdown cell in Notebook 02 Section 1 explicitly noting: *"Note: vgs_min_V, vgs_max_V, and n_points are experimental sweep parameters set by the operator, not intrinsic device properties. Their presence in the feature matrix means the model measures sensitivity to sweep configuration as well as material properties. The key test is whether `layers` carries any importance when these parameters are held fixed."*
2. Add a secondary analysis: rerun the RF with only `layers` as the feature and report that R² separately. This directly tests the layer count hypothesis in isolation.
3. Restate the conclusion in the README as: *"Within the range of experimental conditions tested, and accounting for the presence of sweep configuration parameters in the feature set, layer count does not independently predict ON/OFF ratio."*

---

### S1-03 🟡 MAJOR — Power law exponent from 4-point fit; r = −0.9999 overstated as evidence

**Finding:** The power law exponent −1.15 is derived from a log-log linear fit through 4 median points (one per geometry group). A linear fit through 4 points will achieve r ≈ −0.9999 by near-construction if the data is monotonically consistent — the near-perfect correlation is not surprising and does not independently validate the power law. The exponent uncertainty (standard error from `linregress`) is never reported. The result "Pearson r = −0.9999" from a 4-point fit will trigger scepticism from a peer reviewer, not confidence.

**Affected locations:**
- Notebook 07: Cell 8 (regression cell), Cell 18 (Key Findings table)
- README: Summary of Findings Finding 4, Results Section 1 caption

**Fix:**
1. In Notebook 07 Cell 8, after the `linregress` call, add: `print(f'Slope SE: {se:.4f}')` and `print(f'95% CI on exponent: [{slope - 1.96*se:.3f}, {slope + 1.96*se:.3f}]')`.
2. Report the exponent as `−1.15 ± 0.xx (95% CI)` rather than just `−1.15` in the Key Findings table.
3. Add a note: *"Note: the Pearson r is computed across n=4 geometry-group medians, not n=52 individual devices. Near-perfect r from a 4-point fit is expected for any monotonic dataset and should not be interpreted as strong statistical evidence independent of the physical interpretation."*
4. Update the README caption to include the SE.

---

### S1-04 🟡 MAJOR — Instrument noise floor artefact correctly identified but ON/OFF ratio still reported as device property

**Finding:** Notebook 01 correctly identifies that the i_ON ↔ ON/OFF ratio correlation (r = 0.96) is a noise floor artefact: the OFF-state current is clamped at ~3.66×10⁻⁶ A by the instrument. But the Key Specifications table in the README reports "ON/OFF ratio (best devices): ~10⁷" as if it were a device characteristic. Every ON/OFF ratio reported in this project is partially or wholly an instrument floor artefact. This is the most prominent claim in the README and the least defensible one.

**Affected locations:**
- README: Key Specifications table
- Notebook 01: Section 1 markdown (if it reports ON/OFF ratio as a result)

**Fix:**
1. Remove the "ON/OFF ratio (best devices): ~10⁷" row from the Key Specifications table.
2. Replace with: "ON-state current (best devices): ~10–20 mA at ±20 V sweep".
3. Add a new row: "OFF-state measurement floor: ~3.66×10⁻⁶ A (Keithley instrument limit); intrinsic OFF-state and true ON/OFF ratio not determinable from this dataset."

---

### S1-05 🟡 MAJOR — Confound in 6 µm yield analysis stated in Notebook 06 but not propagated to README and Notebook 04

**Finding:** Notebook 06 correctly identifies that T-code and column position are fully confounded (every T24 device is at column 1, every T12 device is at column 4). The conclusion is "neither hypothesis can be confirmed." But the README Section 8 and Notebook 04 state the spatial yield gradient as if it is a confirmed result: *"FC4 column yield = 92%; FC1 column yield = 9% — consistent with a spray-coating thickness gradient across the substrate."* This is a conclusion drawn from confounded data presented as confirmed. An examiner or reviewer will catch this contradiction between Notebook 06 and the README.

**Affected locations:**
- README: Results Section 8 caption
- Notebook 04: Results table footnote, "Spatial yield breakdown" sentence

**Fix:**
1. In README Section 8: change *"consistent with a spray-coating thickness gradient across the substrate"* to *"consistent with either a spray-coating thickness gradient or a channel-length effect (T-code); these variables are confounded in this dataset and cannot be separated without a designed follow-on experiment."*
2. In Notebook 04: add a sentence after the spatial yield breakdown: *"Caution: column position (FC1 vs FC4) is completely confounded with T-code (T12 vs T24) in this dataset — see Notebook 06 for the full confound analysis. The spatial gradient hypothesis is consistent with the data but unconfirmed."*

---

### S1-06 🟢 MINOR — Electroforming regression slope reported without confidence interval

**Finding:** R² = 0.48, p < 0.001 is correctly included for the electroforming trend (Notebook 03), but the regression slope (the rate of ON-current increase per measurement run) and its 95% CI are never reported. For a paper, this is incomplete — p < 0.001 establishes significance but the slope and CI establish effect size.

**Affected locations:**
- Notebook 03: stability analysis regression cell, summary output
- README: Results Section 6 caption

**Fix:** After the regression in Notebook 03, add: `print(f'Slope: {slope:.4e} A/run (95% CI: {slope-1.96*se:.4e} to {slope+1.96*se:.4e})')`. Update the README caption to include the slope.

---

## Section 2 — Narrative Coherence

### S2-01 🔴 CRITICAL — Notebooks 01, 02, and 03 contain Korean-language section headers, markdown cells, and captions

**Finding:** Notebook 01 contains Korean in section titles (e.g., *"레이어 수(10–60L)에 따른 on/off ratio 및 I_ON 전류 분포 비교"*). Notebook 02 is almost entirely in Korean — every section header, every markdown explanation (e.g., *"Section 1 — Feature Matrix 구성"*, *"각 feature가 on/off ratio 예측에 기여하는 비중"*). Notebook 03 is entirely in Korean (e.g., *"섹션 1 — 데이터 합치기"*, *"섹션 2 — Chip별 i_on 분포"*). Notebooks 04–08 are fully in English. This creates an immediate barrier for any non-Korean reader and makes the project appear half-finished. The README presents a polished English-language narrative but the underlying notebooks — the evidence — are inaccessible.

**Affected locations:**
- notebooks/01_eda.ipynb: all markdown cells
- notebooks/02_random_forest.ipynb: all markdown cells
- notebooks/03_stability_analysis.ipynb: all markdown cells

**Fix:** Translate all Korean markdown cells in notebooks 01, 02, and 03 into English. Preserve the scientific content exactly — section structure, analytical reasoning, interpretation — but in English. Do not translate variable names, column names, or inline code. See Translation Reference below for full cell-by-cell target text.

---

### S2-02 🔴 CRITICAL — No overarching research question stated anywhere in the project

**Finding:** There is no notebook 00, no introductory cell in Notebook 01, and no framing paragraph in the README that states the research question in one sentence. The README begins immediately with Device Architecture. A reader opening the repo cold does not know: *"What scientific problem are we solving? What do we not yet understand? What decision does this analysis inform?"* The Research Context section at the bottom of the README describes BEOL integration relevance but is buried after 8 results sections and does not state the specific question this dataset addresses.

**Affected locations:**
- README: immediately after badges

**Fix:** Add the following paragraph immediately after the badges and before the bold subtitle line:

> This project characterises how electrode geometry controls resistance and switching yield in printed MoS₂/Graphene two-terminal devices fabricated at Imperial College London. The central question: does electrode width determine whether a device can form a conductive filament, and if so, what is the minimum viable geometry for reliable switching? A secondary question: does MoS₂ layer count predict switching performance, or is local interface quality the controlling variable? A third question: does the fitted resistance of functional (non-switching) devices follow the sheet resistance scaling law expected from the MoS₂ film geometry?

---

### S2-03 🟡 MAJOR — Notebook sequence does not follow the scientific argument; cross-references missing

**Finding:** The current order is: EDA → ML → Stability → Electrode Width → Data Quality → Drilldown → Resistance Scaling → IV Curves. This reflects the chronological order of analysis, not the logical order of a scientific argument. Notebooks 05–08 have good linking sentences; notebooks 01–04 do not explain where they sit in the overall argument.

**Affected locations:**
- Notebook 01: opening markdown cell
- Notebook 02: opening markdown cell
- Notebook 03: opening markdown cell
- Notebook 04: opening markdown cell

**Fix:** Add a one-sentence "position in argument" statement to the opening cell of notebooks 01–04. Examples:
- NB01: *"This notebook provides exploratory analysis of gate-sweep and IV-sweep features, establishing the dataset structure before the targeted analyses in Notebooks 02–06."*
- NB02: *"This notebook tests whether MoS₂ layer count predicts switching performance (ON/OFF ratio), using the gate-sweep features prepared in Notebook 01."*
- NB03: *"This notebook analyses temporal stability of the ON-state current across repeated measurement cycles, using devices from Notebooks 01–02's memeffect sweep dataset."*
- NB04: *"This notebook identifies the minimum viable electrode width for reliable switching — the primary process-control finding of this project — using the August 2024 measurement batch."*

---

### S2-04 🟡 MAJOR — Notebook 05 reads as a statistics tutorial rather than a research notebook

**Finding:** The Anscombe's Quartet reference and the "fab practice" explanation (L1 reporting, L3 binning, SPC control charts) in Notebook 05 make the notebook feel like educational content inserted into a research project. An examiner would ask: *"Is this a research finding or a statistics lecture?"* The Anscombe's Quartet analogy is also imprecise — Anscombe's datasets have identical means AND variances; this dataset does not. The notebook does produce a genuine finding (three distinct device populations), but the framing buries it.

**Affected locations:**
- Notebook 05: title cell, Anscombe paragraph, "Connection to Fab Practice" cell

**Fix:**
1. Retitle the notebook opening: change *"Why Aggregate Statistics Hide Device Failures"* to *"Distribution Audit: Electrode Geometry Groups Contain Distinct Device Populations"*.
2. Trim the Anscombe paragraph to one sentence: *"The effect is analogous to Anscombe's Quartet (1973) — datasets with identical means can contain fundamentally different distributions."*
3. Trim the "Connection to Fab Practice" cell to remove the L1/L3/SPC explanation entirely, replacing with: *"Reporting mean ON-current per geometry group, as is common in aggregate device characterisation, obscures the bimodal population structure found at 6 µm. Notebook 06 investigates the root cause."*

---

### S2-05 🟡 MAJOR — Two different chips used for different analyses with no explicit justification or cross-validation

**Finding:** Notebooks 03–04 use Chip#14 (August 2024 batch, 85 sweeps) for stability and electrode width analysis. Notebooks 07–08 use Chip#1 (July 2024, 74 devices) for resistance scaling and IV curves. The README mentions both but never explains why different chips are used, whether they were fabricated under identical conditions, or whether findings transfer between them. The Summary of Findings presents results from both chips as if they are parts of the same experiment.

**Affected locations:**
- README: after Summary of Findings, before Results
- Notebook 07: opening cell

**Fix:** Add a "Dataset Overview" subsection to the README immediately after the Summary of Findings:

> **Dataset Overview**
>
> This project uses two measurement datasets from different chips and campaigns:
>
> | Dataset | Chip | Date | Devices | Used in |
> |---|---|---|---|---|
> | Gate sweep (73 files) | Multiple chips | 2023–2024 | 73 gate measurements | Notebooks 01–02 |
> | Memeffect sweep (124 files) | Chip#14 (primary) | Aug 2024 | 85 IV sweeps | Notebooks 03–04 |
> | R5 resistance table | Chip#1 | Jul 2024 | 74 devices | Notebooks 07–08 |
>
> Findings from Chip#14 (switching yield, stability) and Chip#1 (resistance scaling) have not been cross-validated. Both chips use the same MoS₂/Graphene stack and electrode geometry design, but were fabricated and measured independently.

---

### S2-06 🟡 MAJOR — "already_on" device state used throughout but never defined for the reader

**Finding:** Notebooks 03–06 filter on `switching_state == "already_on"` — a term that appears in the README implicitly through stability analysis results. A reader who hasn't seen the raw data doesn't know what this means: does it mean the device was found conducting before any measurement? Does it mean it switched during a prior session? This distinction is scientifically important — it determines whether the electroforming trend measures post-forming stability or the forming process itself.

**Affected locations:**
- README: Scope and Constraints section
- Notebook 03: opening cell (before data loading)

**Fix:**
1. Add to README Scope and Constraints: *"Devices classified as 'already_on' were found conducting at mA-range current at the start of the measurement session, indicating that conductive filament formation (electroforming) had already occurred in a prior measurement. The stability analysis (Notebook 03) therefore measures post-forming filament evolution, not the initial forming event."*
2. Add identical definition as the first paragraph of Notebook 03's Section 1 markdown cell.

---

### S2-07 🟢 MINOR — Pipeline diagram shows linear flow; actual data has two parallel input tracks

**Finding:** The pipeline image shows a single linear flow from raw CSVs to five analysis phases. The actual data flow branches: gate sweep track (layer_sweep.csv → notebooks 01–02) and memeffect sweep track (memeffect_sweep.csv → notebooks 03–06), plus a separate Chip#1 track (processedTable.csv → notebook 07). The current diagram implies one linear story when the data branches.

**Affected locations:**
- results/figures/pipeline.png (the source diagram, not the PNG)

**Fix:** Update the pipeline diagram to show the branching structure — two input datasets (gate sweeps and memeffect sweeps) feeding different notebook groups, with Chip#1 processedTable as a separate third track. This requires updating the pipeline image source (FigJam or equivalent) and re-exporting. Note in the README: *"The pipeline branches at extraction: gate sweep features feed Notebooks 01–02; IV sweep features feed Notebooks 03–06; Chip#1 resistance metrics feed Notebooks 07–08."*

---

## Section 3 — Code Reproducibility

### S3-01 🔴 CRITICAL — L_ASSUMED = 20 µm is hardcoded and flagged as uncertain; all R_sheet values depend on it

**Finding:** Notebook 07 Cell 10 contains: `L_ASSUMED = 20  # µm — UPDATE THIS from your fabrication notes`. Every R_sheet value in the project (the "~1800–2000 Ω/□" result in the README Summary of Findings, Key Findings table, and Key Specifications) is derived from this single unverified constant. If the actual channel length differs from 20 µm, all R_sheet values are proportionally wrong. The comment "UPDATE THIS" remains in the publicly committed notebook.

**Affected locations:**
- Notebook 07: Cell 10
- README: Summary of Findings Finding 4, Key Specifications (if R_sheet is listed)

**Fix:**
1. **If channel length is confirmed as 20 µm from fabrication records:** Replace the comment with: `L_ASSUMED = 20  # µm — confirmed from lithography mask design (Torrisi Lab, 2024)`.
2. **Regardless:** Add a sensitivity analysis cell immediately after the R_sheet calculation:
```python
print('R_sheet sensitivity to channel length assumption:')
for L_test in [15, 18, 20, 22, 25]:
    r_sheet_test = df_r['R_sheet'].median() * 20 / L_test
    print(f'  L = {L_test} µm → median R_sheet ≈ {r_sheet_test:.0f} Ω/□')
```
3. In the README, report R_sheet as: *"~1800–2000 Ω/□ (assuming L = 20 µm; see Notebook 07 for sensitivity analysis)."*

---

### S3-02 🔴 CRITICAL — table.csv not in repo; Notebook 08 cannot be reproduced without undocumented setup

**Finding:** Notebook 08 loads `table.csv` (42.8 MB) via path detection that falls back to `~/Downloads/`. Anyone cloning the repo cannot run Notebook 08 without separately obtaining the file. The README Quickstart section gives no instruction about this. `table.csv` is not listed in `.gitignore` explicitly, creating risk of accidental future commit.

**Affected locations:**
- README: Quickstart section, Scope and Constraints section
- Notebook 08: opening cell
- .gitignore

**Fix:**
1. Add `data/table.csv` to `.gitignore` explicitly with a comment.
2. Add to README Quickstart:
> **Data availability:** Notebooks 01–06 run fully from data included in this repository. Notebook 07 requires `data/processedTable.csv` (included). Notebook 08 requires `table.csv` (42.8 MB raw sweep data; not committed). Place it in `data/` or `~/Downloads/` — the notebook will find it automatically. Contact [your email] to request access.
3. Add to Notebook 08 Cell 0 (the path detection cell): a clearer error message: `raise FileNotFoundError('table.csv not found. Place in data/ or ~/Downloads/. See README for access instructions.')`.

---

### S3-03 🟡 MAJOR — requirements.txt has no version pinning; results not reproducible across package updates

**Finding:** `requirements.txt` lists bare package names with no version constraints. Scientific computing results can change between package versions — particularly sklearn model behaviour, scipy statistics functions, and matplotlib rendering defaults.

**Affected locations:**
- requirements.txt

**Fix:** Replace current requirements.txt with pinned versions from the project venv:
```
pandas==3.0.2
numpy==2.4.4
matplotlib==3.10.9
scipy==1.17.1
scikit-learn==1.8.0
jupyter>=1.0.0
nbformat==5.10.4
nbconvert==7.17.1
nbclient==0.10.4
ipython==9.13.0
ipykernel==7.2.0
```

---

### S3-04 🟡 MAJOR — Notebooks 04, 05, 06 repeat full import blocks in every code cell

**Finding:** Notebook 04 has `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt` repeated in at least 5 separate code cells. Notebooks 05 and 06 follow the same pattern. This signals the notebooks were developed as standalone scripts and assembled rather than designed as coherent documents. It also means cells can be run out of order, making the execution dependency non-linear.

**Affected locations:**
- notebooks/04_electrode_width_analysis.ipynb: cells 1, 3, 5, 6, 7
- notebooks/05_data_quality_audit.ipynb: cells 1, 3, 5, 7
- notebooks/06_drilldown_analysis.ipynb: cells 1, 3, 5, 7

**Fix:** Restructure each notebook to have a single consolidated import/setup cell (Cell 0 or Cell 1) containing all imports, path setup, and data loading. All subsequent cells should reference variables from that cell without re-importing. This requires merging the repeated import blocks and consolidating data loading.

---

### S3-05 🟡 MAJOR — Measurement conditions (Keithley settings) not documented anywhere

**Finding:** The project analyses data from a Keithley 2634B SMU, but no measurement parameters are recorded: compliance current limit, voltage sweep range, sweep rate, number of measurement points per sweep, integration time, settling time, or measurement channel configuration. These parameters directly affect the data — the noise floor (~3.66×10⁻⁶ A) is a consequence of specific instrument settings. A reader trying to reproduce the measurement has no information.

**Affected locations:**
- README: Key Specifications table (partial — instrument listed but settings missing)
- docs/device_physics_notes.md (should contain this)

**Fix:** Add a "Measurement Conditions" subsection to the README Key Specifications section, or to `docs/device_physics_notes.md` with a link:

| Parameter | Value |
|---|---|
| Instrument | Keithley 2634B dual-channel SourceMeter |
| Sweep mode | Dual-sweep IV (forward + reverse) |
| Voltage range | [value] V to [value] V |
| Points per sweep | 1500 (half-sweep) / 4000 total per file |
| Compliance current | [value] A |
| Integration time | [value] s/point |
| Measurement floor | ~3.66×10⁻⁶ A (OFF-state clamped) |

*(Values marked [value] require confirmation from lab records.)*

---

### S3-06 🟡 MAJOR — Notebook 07 never explains how R5 was extracted; processedTable.csv origin undocumented

**Finding:** The entire resistance scaling analysis depends on `processedTable.csv` containing R5 fitted resistance values, but Notebook 07 never explains or validates how R5 was computed. `R5_AvgFitQuality` is used to assess reliability, but the fitting algorithm is treated as a black box. Which script generated processedTable.csv? What model was fitted? What does "R5" stand for?

**Affected locations:**
- Notebook 07: opening cell

**Fix:** Add to Notebook 07 Cell 0 (the existing markdown header):

> **Data provenance:** `processedTable.csv` was generated by [script name / external pipeline] and contains the R5 resistance metric extracted from raw IV sweeps in `table.csv`. R5 is defined as [the fitted resistance from a linear ohmic model / R = V/I at V=5V / etc.]. `R5_AvgFitQuality` is the mean R² of the linear fit across all sweeps for each device. Devices with R5_AvgFitQuality < 0.99 are flagged as unreliable. The extraction pipeline is described in `docs/device_physics_notes.md`.

---

### S3-07 🟢 MINOR — processedTable.csv in data/ root rather than data/processed/

**Finding:** The three feature CSV files are in `data/processed/`, but `processedTable.csv` is in `data/` directly. The README repository structure documents this accurately but the inconsistency is jarring.

**Affected locations:**
- data/processedTable.csv
- Notebook 07: data loading path

**Fix:** Move `processedTable.csv` to `data/processed/processedTable.csv`. Update the path reference in Notebook 07 from `'../data/processedTable.csv'` to `'../data/processed/processedTable.csv'`. Update the README repository structure tree.

---

## Section 4 — Per-Audience Readability

### S4-01 🔴 CRITICAL — No "contribution to knowledge" statement for examiner audience

**Finding:** An MRes examiner's primary question is: *"What did you learn that wasn't known before?"* The project never answers this directly. The Summary of Findings lists four results but doesn't distinguish what is novel to this device system versus what is already established. Is the R ∝ W⁻¹·¹⁵ scaling result new? Is the 12 µm minimum viable CD known from prior work?

**Affected locations:**
- README: after Summary of Findings

**Fix:** Add a "Contribution" paragraph immediately after the Summary of Findings:

> **Contribution:** These results provide the first systematic CD-dependent yield and resistance characterisation for this specific printed MoS₂/Graphene two-terminal stack fabricated at Imperial College London (Torrisi Lab). The process window bound of ≥12 µm minimum electrode width, the sheet resistance value of ~1800–2000 Ω/□, and the identification of two physically distinct failure modes at 2 µm geometry establish quantitative process targets for future device optimisation in the group. The confound identified in the 6 µm yield analysis defines the specific experiment required to separate spatial from channel-length effects.

---

### S4-02 🟡 MAJOR — Scope and Constraints section placed too late; examiners see confident claims before caveats

**Finding:** The Scope and Constraints section is at the bottom of the README, after 8 results sections, the pipeline, and the repository structure. An examiner reading linearly encounters all confident declarative claims first and the caveats last. The Scope section is one of the strongest parts of the project — it should be encountered early, not as a footnote.

**Affected locations:**
- README: Scope and Constraints (currently at bottom)

**Fix:** Move the Scope and Constraints section to immediately after the Summary of Findings and Contribution paragraph, before the Results sections.

---

### S4-03 🟡 MAJOR — No statistical confidence intervals on yield percentages (especially 2 µm at n=6)

**Finding:** "50% yield at 2 µm" is stated in multiple places. This comes from n=6 devices. The 95% binomial confidence interval for 3/6 = 50% at n=6 is approximately 12%–88% — a range that spans almost the entire possible space. Reporting "50%" without this context implies a precision the data cannot support. Similarly, "71–75% at wider geometries" comes from n=6–20 devices per group; CIs should be reported for all.

**Affected locations:**
- Notebook 07: Key Findings table, yield plot caption
- README: Summary of Findings Finding 4

**Fix:** Add binomial 95% CIs to all yield percentages. Use `scipy.stats.binom.interval()` or the Wilson score interval. Add to Notebook 07 after the yield calculation:
```python
from scipy.stats import binom
for w in widths:
    n = yield_data[w]['total']
    k = yield_data[w]['clean']
    ci_lo, ci_hi = binom.interval(0.95, n, k/n)
    print(f'{w}µm: {k}/{n} = {k/n*100:.0f}% (95% CI: {ci_lo/n*100:.0f}%–{ci_hi/n*100:.0f}%)')
```

---

### S4-04 🔴 CRITICAL — No citations anywhere; literature claims are unsubstantiated

**Finding:** The README and notebooks make multiple claims that invoke prior literature — *"consistent with printed MoS₂ literature"*, *"ALD nucleation failure on the inert van der Waals basal plane"*, *"grain boundary barriers"*, *"stochastic filament nucleation"* — but there is not a single reference or citation in the entire project. A peer reviewer will reject a paper where literature claims are unsubstantiated. Even for a GitHub portfolio, the absence of references makes physical interpretations appear speculative.

**Affected locations:**
- README: Research Context section, Results captions, Summary of Findings
- Notebook 04: physical mechanism cell
- Notebook 07: Key Findings physical interpretation

**Fix:** Add a References section to the README with the following structure. Real DOIs must be confirmed before submission, but the reference categories are:
1. MoS₂ resistive switching review (e.g., Sangwan & Hersam, Nature Nanotechnology, 2020 — TMD memristors)
2. Printed 2D device fabrication (relevant Torrisi Lab or Casiraghi group paper)
3. Electroforming mechanism in TMD devices
4. Sheet resistance of solution-processed MoS₂ (to validate the 1800–2000 Ω/□ value)
5. Van der Waals surface passivation and contact challenges (to support the basal plane nucleation argument)

Add inline citation tags in notebooks 04 and 07 wherever physical mechanisms are claimed: `# Ref: [Sangwan2020]` etc.

---

### S4-05 🟡 MAJOR — Negative R² in Notebook 02 will be misread as modelling failure by recruiter audience

**Finding:** R² = −0.09 is the correct result and a valid scientific finding. But to a data science recruiter unfamiliar with materials science, a negative R² looks like the model failed. The project's ML notebook is the most visible data science artefact for a portfolio audience and its headline number is negative.

**Affected locations:**
- Notebook 02: Summary cell
- README: Summary of Findings Finding 1, Results Section 2 caption

**Fix:** Add a framing sentence to Notebook 02's Summary cell and to the README Section 2 caption: *"A negative R² confirms that this model performs worse than simply predicting the mean ON/OFF ratio — this is the intended finding. It establishes quantitatively that layer count, vgs range, and measurement parameters carry no predictive signal for this target variable, which is a process-relevant scientific result, not a modelling failure."*

---

### S4-06 🟡 MAJOR — No "skills demonstrated" block for recruiter audience

**Finding:** A recruiter spending 90 seconds on this repo needs to understand the skill set demonstrated immediately. The README leads with Device Architecture — a technical description requiring domain knowledge. There is no quick summary of what data science and engineering skills are demonstrated.

**Affected locations:**
- README: immediately after research question paragraph

**Fix:** Add the following block after the research question paragraph and before the Device Architecture section:

> **Skills demonstrated:** experimental data analysis · feature engineering · Random Forest regression · statistical hypothesis testing (Mann-Whitney U, Pearson r, linear regression) · power law fitting · signal processing (log-domain SET/RESET detection) · spatial yield analysis · scientific data visualisation (matplotlib) · Jupyter notebook documentation · reproducible analysis pipeline

---

### S4-07 🟡 MAJOR — Quickstart does not mention data file dependencies

**Finding:** The Quickstart says `pip install -r requirements.txt && jupyter notebook notebooks/01_eda.ipynb`. This works for notebooks 01–06 but fails for notebooks 07–08 without `processedTable.csv` and `table.csv`.

**Affected locations:**
- README: Quickstart section

**Fix:** Add a data availability note to the Quickstart as described in S3-02.

---

### S4-08 🟢 MINOR — docs/ folder not surfaced for any audience

**Finding:** `device_physics_notes.md` is referenced once in the README but not described. For an examiner audience, this file is potentially the most valuable supplementary material — it may contain the measurement methodology and device physics that are missing from the notebooks.

**Affected locations:**
- README: Repository Structure section

**Fix:** After the docs/ tree entry, add a brief description of each doc file's contents:
- `device_physics_notes.md` — detailed reference on switching mechanisms, device physics, and measurement methodology
- `switching_variability_technical_note.md` — technical note on process implications of the variability findings
- `analysis_log.md` — per-notebook analysis log with decision rationale

---

### S4-09 🟢 MINOR — Notebook 08 Key Findings table has "—" placeholders in the Sweeps column

**Finding:** The Key Findings table in Notebook 08 shows "—" in the Sweeps column for all five devices. This was intentional but looks unfinished. The actual sweep counts are printed during cell execution.

**Affected locations:**
- Notebook 08: Cell 6 (Key Findings markdown)

**Fix:** After Cell 3 executes and prints sweep counts, update the markdown table in Cell 6 with the actual values. Alternatively, replace the static markdown table with a dynamically generated DataFrame print.

---

## Section 5 — Scope and Claims Alignment

### S5-01 🔴 CRITICAL — Device labelled "bipolar memristor (filamentary switching)" based on two observed switching events

**Finding:** The Key Specifications table states: *"Device type: Bipolar memristor (filamentary switching)."* The evidence for bipolar switching is exactly two IV sweeps — Run 33 and Run 35 on one device (Chip#1, CC1-T14). Every other device in the dataset was measured in the "already_on" (post-forming) state, meaning their switching behaviour was never directly observed. The filamentary mechanism is inferred from switching polarity, not confirmed by direct imaging or cycling data. No endurance, retention, or multi-cycle characterisation exists in this project.

**Affected locations:**
- README: Key Specifications table
- README: Device Architecture section ("bipolar memristor" in first line)
- Notebook 07: opening cell

**Fix:**
1. Change Key Specifications "Device type" row to: *"Two-terminal resistive switching device; bipolar filamentary switching inferred from polarity-dependent SET transitions (n=2 switching events directly observed)."*
2. Change Device Architecture opening from *"The device is a solution-processed bipolar memristor"* to *"The device is a solution-processed two-terminal resistive switching device exhibiting bipolar SET behaviour."*
3. Add a note in Notebook 07 opening: *"Switching characterisation data (SET/RESET cycling) is limited to two observed events in Chip#1 (Runs 33 and 35). The majority of devices in this dataset were measured post-forming. Endurance and retention data were not collected in this measurement campaign."*

---

### S5-02 🔴 CRITICAL — Resistance scaling (Finding 4) and switching yield (Finding 3) presented as one coherent experiment; they are from different chips

**Finding:** The Summary of Findings presents four findings as if they come from a unified experiment. Findings 1–3 are from Chip#14 and the gate-sweep dataset. Finding 4 is from Chip#1. These chips were not cross-validated against each other, and the finding that resistance scales with electrode width (Chip#1) does not directly confirm that yield also scales with electrode width (Chip#14) via the same mechanism.

**Affected locations:**
- README: Summary of Findings preamble
- README: Summary of Findings Finding 3 and Finding 4

**Fix:** Add to the Summary of Findings preamble: *"Findings 1–3 are derived from Chip#14 and the gate-sweep dataset. Finding 4 is derived from Chip#1. Results across chips have not been cross-validated."* This is also addressed by S2-05.

---

### S5-03 🟡 MAJOR — "Near-ideal sheet resistance scaling" understates a physically meaningful 15% deviation

**Finding:** The exponent −1.15 versus ideal −1.00 represents a 15% deviation. This is described as "near-ideal" throughout. But the deviation is physically meaningful: an exponent more negative than −1.00 implies either a width-dependent contact resistance contribution or edge-conduction effects. The project never discusses what causes the deviation or whether it is within measurement uncertainty. Calling it "near-ideal" without explanation will be flagged by a reviewer as either sloppy framing or a missed finding.

**Affected locations:**
- README: Summary of Findings Finding 4
- Notebook 07: Key Findings physical interpretation

**Fix:** Add a paragraph to Notebook 07's physical interpretation discussing the deviation: *"The exponent of −1.15 is slightly steeper than the ideal −1.00 expected for pure sheet resistance scaling (R = R_sheet × L / W). This 15% deviation could reflect: (1) a width-dependent contact resistance component, where narrower electrodes present proportionally more edge-dominated contact area with higher R_contact; (2) fringe field contributions at electrode edges; or (3) systematic uncertainty in the assumed channel length L = 20 µm. Distinguishing between these requires a transfer length method (TLM) measurement, which is beyond the scope of this dataset."* Update the README phrasing from "near-ideal" to *"near-ideal (exponent −1.15 vs theoretical −1.00; 15% deviation discussed in Notebook 07)."*

---

### S5-04 🟡 MAJOR — "Minimum reliable electrode CD: 12 µm" is a process claim from one fabrication run

**Finding:** This conclusion appears in the Key Specifications table and as a headline result in Notebook 04 and the README. But it is derived from one fabrication run, one chip, one ink batch, one specific spray-coating recipe, and one measurement setup. Minimum viable CD is a process-window parameter that requires multiple splits and statistical yield modelling to be generalised.

**Affected locations:**
- README: Key Specifications table
- Notebook 04: Results table heading, conclusion cell
- README: Results Section 8

**Fix:** Add qualification throughout: *"Minimum reliable electrode CD (this fabrication run, ink batch, and coating recipe): ≥12 µm. Generalisation to other process conditions requires a replicated DOE varying electrode width independently of other parameters."* Update the Key Specifications table entry to include this qualifier.

---

### S5-05 🟡 MAJOR — Electroforming trend conclusion conflates post-forming filament evolution with electroforming itself

**Finding:** The stability analysis (Notebook 03) filters on `switching_state == "already_on"` — devices already conducting when measured. The increasing ON-current trend across run numbers therefore shows post-forming filament evolution, not the electroforming process (initial filament nucleation and formation). The README states *"consistent with progressive conductive filament widening — the electroforming signature."* Post-forming filament evolution is physically distinct from electroforming. These are different phenomena with different mechanisms.

**Affected locations:**
- README: Summary of Findings Finding 2, Results Section 6 caption
- Notebook 03: Section 3 markdown, regression interpretation

**Fix:** Change the framing throughout: *"consistent with progressive post-forming filament evolution — continued widening or crystallisation of an already-formed filament. Note: this is distinct from the initial electroforming event (filament nucleation), which was not captured in this dataset as all analysed devices were already in the ON state at the start of measurement."*

---

### S5-06 🟡 MAJOR — Physical mechanism interpretations stated as confirmed; they are hypotheses

**Finding:** Multiple physical mechanism claims are presented as established fact:
- *"contact formation failure on the chemically inert MoS₂ basal plane"* — no surface characterisation data
- *"grain boundary barriers"* — no TEM, Raman, or XPS data
- *"ALD nucleation failure on vdW surface"* — this is ALD-specific terminology applied to a spray-coated device (category error)
- *"progressive conductive filament widening"* — consistent with the data but not the only explanation

**Affected locations:**
- README: Results Section 1 caption, Finding 4
- Notebook 04: Physical Origin cell
- Notebook 07: Key Findings physical interpretation

**Fix:**
1. Replace "ALD nucleation failure" with "nucleation/coverage gap on the van der Waals basal plane" throughout (spray coating, not ALD).
2. Downgrade all physical mechanism language from declarative to evidential: *"consistent with"* / *"suggestive of"* / *"a candidate explanation is"* rather than *"confirming"* / *"attributable to"* / *"driven by"*.
3. Full replacement list:
   - *"attributable to nucleation gaps"* → *"consistent with nucleation or coverage gaps in the spray-coated MoS₂ film"*
   - *"consistent with grain boundary barriers"* → *"consistent with grain boundary or localised defect barriers (not directly confirmed)"*
   - *"contact formation failure on the chemically inert MoS₂ basal plane"* → *"consistent with contact formation challenges on the chemically inert MoS₂ basal plane"*
   - *"ALD nucleation failure on vdW surface"* → *"nucleation gap on the van der Waals basal plane"* (remove ALD reference)

---

### S5-07 🟢 MINOR — Chip#6 interpretation stated with minimal evidence

**Finding:** The README states: *"Chip#6 devices remain in the pA range — consistent with incomplete electroforming or insufficient field to nucleate a stable filament."* Chip#6 appears in a boxplot in Notebook 03 but receives no dedicated analysis. Offering two specific physical hypotheses (incomplete electroforming vs insufficient field) for Chip#6 without any supporting data is a minor overclaim.

**Affected locations:**
- README: Results Section 5 caption

**Fix:** Change to: *"Chip#6 devices remain in the pA range — in the pre-forming state. The cause (insufficient applied field, thin MoS₂, or fabrication variation) was not investigated in this study."*

---

### S5-08 🟢 MINOR — Notebook 08 Key Findings table states "Sweeps" as "—" (placeholder)

*(Covered under S4-09 above.)*

---

## Implementation Priority Order

### Tier 1 — Fix immediately (blocks all three audiences)
1. S2-01: Translate Korean notebooks 01, 02, 03 → English
2. S4-04: Add references section and inline citation tags
3. S5-01: Fix "bipolar memristor" overclaim
4. S1-04: Remove/fix ON/OFF ratio from Key Specifications
5. S5-06: Fix "ALD nucleation failure" and downgrade mechanism language
6. S3-01: Fix L_ASSUMED — add sensitivity analysis

### Tier 2 — Fix before submission (scientific rigour and coherence)
7. S2-02: Add research question paragraph to README
8. S4-01: Add contribution statement
9. S4-02: Move Scope and Constraints before Results
10. S2-05: Add dataset overview table (two chips)
11. S1-01: Add single-fabrication-run caveat to Summary of Findings
12. S1-02: Fix Random Forest feature matrix framing
13. S1-05: Fix confound language in README and NB04
14. S5-02: Flag chip split in Summary of Findings
15. S5-04: Qualify "minimum reliable CD" claim
16. S5-05: Fix electroforming vs post-forming language
17. S2-06: Define "already_on" in README and NB03
18. S1-03: Add exponent SE and note on 4-point fit
19. S5-03: Discuss −1.15 exponent deviation
20. S4-03: Add binomial CIs on yield percentages

### Tier 3 — Polish (readability and reproducibility)
21. S3-02: Add table.csv to .gitignore; update README Quickstart
22. S3-03: Pin requirements.txt versions
23. S3-04: Remove repeated imports from NB04, NB05, NB06
24. S2-03: Add cross-reference opening sentences to NB01–04
25. S2-04: Trim NB05 Anscombe/SPC content
26. S4-05: Add negative R² framing sentence
27. S4-06: Add skills demonstrated block to README
28. S4-07: Update Quickstart with data availability note
29. S3-05: Add measurement conditions table
30. S3-06: Document R5 extraction in NB07
31. S3-07: Move processedTable.csv to data/processed/
32. S1-06: Add regression slope CI to NB03
33. S4-08: Describe docs/ files in README
34. S4-09: Fill in sweep counts in NB08 table
35. S2-07: Update pipeline diagram to show branching

---

## Items Requiring User Input Before Implementation

| Item | What is needed |
|---|---|
| S3-01 | Confirm actual channel length L from fabrication/mask records |
| S3-05 | Keithley settings: compliance current, sweep rate, integration time |
| S4-04 | Actual paper citations with DOIs for references section |
| S3-06 | Name of script/pipeline that generated processedTable.csv; definition of R5 metric |
| S2-07 | Updated pipeline image (FigJam export) showing branching structure |
