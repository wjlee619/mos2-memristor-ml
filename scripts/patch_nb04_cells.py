"""Patch cells 2 and 4 in 04_electrode_width_analysis.ipynb, then re-execute."""

import nbformat
from pathlib import Path

PROJ   = Path(__file__).resolve().parent.parent
NB_PATH = PROJ / "notebooks" / "04_electrode_width_analysis.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

# ── Cell 2 (0-indexed) — Finding markdown ────────────────────────────────────
nb.cells[2].source = (
    "## Finding\n\n"
    "ON-current scales **monotonically** with electrode width:\n\n"
    "| Electrode width | Mean i_on_A | n | Status |\n"
    "|---|---|---|---|\n"
    "| 18 μm | 10.49 mA | 30 | reliable switching |\n"
    "| 12 μm | 5.89 mA  | 18 | reliable switching |\n"
    "| 6 μm  | 2.22 mA  | 35 | **HIGH VARIABILITY** — marginal |\n"
    "| 2 μm  | ~2.55 pA | 2  | leakage only, no switching |\n\n"
    "The 6 μm devices show **high variability**: some devices produce mA-range\n"
    "currents while others show leakage-level current, indicating 6 μm is a\n"
    "marginal geometry where switching is inconsistent. This variability is not\n"
    "random noise — it reflects whether a conductive filament successfully formed\n"
    "at all, and is likely driven by local MoS$_2$ coverage and interface quality\n"
    "at the edge of the viable contact area.\n\n"
    "**Revised conclusion:** The minimum *reliable* electrode width is **12 μm**.\n"
    "The 6 μm threshold is marginal, not reliable — it cannot be used as a\n"
    "manufacturing target. Any electrode patterned at 6 μm or below risks\n"
    "non-functional devices at unknown yield. The design window between 6 and\n"
    "12 μm is the priority range for a future width-sweep experiment to pin down\n"
    "the exact threshold."
)

# ── Cell 4 (0-indexed) — ALD/interface markdown — append coverage note ────────
coverage_note = (
    "\n\n**Note on spatial coverage:** Measurement coverage was concentrated on\n"
    "Row F devices. Rows A and E have sparse coverage — the map reflects where\n"
    "measurements were taken, not full chip characterisation. A complete spatial\n"
    "study would require probing all row/electrode combinations systematically."
)
nb.cells[4].source = nb.cells[4].source.rstrip() + coverage_note

# Clear all existing outputs so re-execution starts clean
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nbformat.write(nb, NB_PATH)
print("Cells patched and outputs cleared.")
print(f"Cell 2 preview: {nb.cells[2].source[:80]!r}")
print(f"Cell 4 tail:    {nb.cells[4].source[-100:]!r}")
