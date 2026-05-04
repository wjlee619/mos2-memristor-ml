
---
## Notebook 04 — Electrode Width vs Switching Performance

**Problem:** All devices on Chip#14 share identical MoS2 layer count.
Electrode width (2/6/12/18 um) was varied by design.
This notebook isolates contact geometry as the process variable driving
switching performance.

**Approach:** Filename position codes parsed to extract electrode width per
device. i_on_A compared across width groups. Spatial chip map built coloured
by ON-current.

**Findings:**
- ON current scales monotonically with electrode width
- 18 um: 9.44 mA | 12 um: 5.89 mA | 6 um: 2.22 mA
- 2 um devices: ~2.55 pA — leakage only, never switched
- Minimum viable electrode width lies between 2 and 6 um
- Spatial map shows no systematic positional clustering —
  geometry dominates over chip position

**Physical mechanism:** Wider electrode = larger contact area = lower contact
resistance = higher filament current. At 2 um, edge-dominated interface
defects prevent reliable filament formation.

**Output:** results/figures/04_electrode_width_ion.png,
results/figures/04_spatial_map.png

---
## Notebook 05 — Why Aggregate Statistics Hide Device Failures

**Problem:** Aggregate i_on statistics across electrode widths appear to
show a smooth monotonic trend. This obscures three fundamentally different
device populations.

**Approach:** Full distribution analysis per electrode width group using
violin plots (KDE in log space) and strip plots overlaid. Comparison of
mean-only view vs full distribution view.

**Findings:**
- 18 um/12 um: tight mA-range, consistent switching
- 6 um: bimodal — ~25% at pA (leakage), ~75% at mA. Mean of 2.22 mA
  represents neither population accurately.
- 2 um: exclusively pA (leakage). Not low performance — a different
  physical state entirely.
- All-device mean: ~5.49 mA — represents no real device accurately.

**Output:** results/figures/05_mean_bar.png,
05_distribution_grid.png, 05_mean_vs_distribution.png
