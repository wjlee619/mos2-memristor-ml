"""
Parse position-code fields from memeffect_sweep_aug30.csv filenames and
produce a derived CSV with three new columns:

    row_block  – first letter of position code  (e.g. 'E', 'F', 'A')
    electrode  – second letter mapped to width   (A→18, B→12, C→6, D→2) µm
    column     – digit from position code        (1–4)

Output: data/derived/memeffect_sweep_aug30_parsed.csv
"""

import re
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
SRC  = PROJ / "data" / "processed" / "memeffect_sweep_aug30.csv"
OUT  = PROJ / "data" / "derived"   / "memeffect_sweep_aug30_parsed.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(SRC)

# ── parse position code ───────────────────────────────────────────────────────
# Filename pattern: Chip#14-<POS>-<TCODE>-...
# POS is always 3 chars: letter + letter + digit  (e.g. ED4, FA4, FC1)
pos_pattern = re.compile(
    r"^Chip#\d+-([A-Z])([A-Z])(\d)-",
    re.IGNORECASE,
)

ELECTRODE_WIDTH = {"A": 18, "B": 12, "C": 6, "D": 2}

rows, bad = [], []
for fname in df["filename"]:
    m = pos_pattern.match(str(fname))
    if m:
        row_block, elec_letter, col = m.group(1), m.group(2), m.group(3)
        rows.append({
            "row_block": row_block.upper(),
            "electrode": ELECTRODE_WIDTH[elec_letter.upper()],
            "column":    int(col),
        })
    else:
        bad.append(fname)
        rows.append({"row_block": None, "electrode": None, "column": None})

if bad:
    print(f"WARNING: {len(bad)} filenames did not match expected pattern:")
    for f in bad:
        print(f"  {f}")

parsed = pd.DataFrame(rows)
df = pd.concat([df, parsed], axis=1)

# ── statistics ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Unique electrode widths and measurement counts")
print("=" * 60)
counts = df.groupby("electrode").size().rename("n_measurements")
print(counts.to_string())

print()
print("=" * 60)
print("Mean i_on_A per electrode width (NaN excluded)")
print("=" * 60)
mean_ion = (
    df[df["i_on_A"].notna()]
    .groupby("electrode")["i_on_A"]
    .mean()
    .rename("mean_i_on_A")
)
print(mean_ion.map("{:.4e}".format).to_string())

print()
print("=" * 60)
print("switching_state counts per electrode width")
print("=" * 60)
state_counts = (
    df.groupby(["electrode", "switching_state"])
    .size()
    .unstack(fill_value=0)
)
print(state_counts.to_string())

# ── save ─────────────────────────────────────────────────────────────────────
df.to_csv(OUT, index=False)
print(f"\nSaved → {OUT.relative_to(PROJ)}  ({len(df)} rows, {len(df.columns)} cols)")
