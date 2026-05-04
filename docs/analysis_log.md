
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
