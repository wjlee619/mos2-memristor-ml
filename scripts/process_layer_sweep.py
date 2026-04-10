"""
Extract Id, on/off ratio per device from probe station CSV files.
Reads from: ~/Desktop/Imperial/Mres Soft electronics/DATA/probe/
Writes to:  ~/Desktop/mos2-memristor-ml/data/processed/layer_sweep.csv

CSV format:
  Row 0: metadata key-value pairs (semicolon-separated)
  Row 1: column headers — index, Vgs (V), Ig (A), Id (A), Id2 (A)
  Row 2+: data

On/off ratio = max(|Id|) / min(|Id|) over the full sweep.
Id_on  = max |Id| across all Vgs points in the file.
Id_off = min |Id| across all Vgs points (excluding exact zero to avoid
         instrument floor artefacts; falls back to global min if needed).
"""

import csv
import os
import re
import math
from pathlib import Path

PROBE_ROOT = Path.home() / "Desktop/Imperial/Mres Soft electronics/DATA/probe"
OUT_CSV = Path.home() / "Desktop/mos2-memristor-ml/data/processed/layer_sweep.csv"


def parse_layers(folder_name: str) -> int | None:
    """Extract layer count from a folder name like '20 layers', '30 layer', 'Mos2 10 layer'."""
    m = re.search(r"(\d+)\s*layer", folder_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_csv(path: Path):
    """
    Returns (vgs_list, id_list) — parallel lists of float values.
    Skips the metadata row (row 0) and uses row 1 as header.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Skip rows until we find the header (contains "Vgs")
        header = None
        for row in reader:
            if any("Vgs" in cell for cell in row):
                header = row
                break
        if header is None:
            raise ValueError("header row with 'Vgs' not found")

        # Find column indices
        vgs_col = next(i for i, h in enumerate(header) if "Vgs" in h)
        id_col  = next(i for i, h in enumerate(header) if h.strip() == "Id (A)")

        vgs_vals, id_vals = [], []
        for row in reader:
            if len(row) <= max(vgs_col, id_col):
                continue
            try:
                vgs_vals.append(float(row[vgs_col]))
                id_vals.append(float(row[id_col]))
            except ValueError:
                continue

    return vgs_vals, id_vals


def compute_metrics(vgs_vals, id_vals):
    """
    Returns (id_on, id_off, on_off_ratio).

    id_on  = max |Id| (on-state current)
    id_off = min |Id| excluding zero (instrument noise floor).
             If all values are zero fall back to global min.
    """
    abs_id = [abs(v) for v in id_vals]
    id_on = max(abs_id)

    nonzero = [v for v in abs_id if v > 0]
    id_off = min(nonzero) if nonzero else 0.0

    if id_off == 0.0:
        on_off_ratio = float("inf")
    else:
        on_off_ratio = id_on / id_off

    return id_on, id_off, on_off_ratio


def main():
    rows = []

    for csv_path in sorted(PROBE_ROOT.rglob("*.csv")):
        # Determine layer count from the parent folder name
        layer_folder = csv_path.parent.name
        layers = parse_layers(layer_folder)
        if layers is None:
            print(f"  [SKIP] cannot parse layer count from: {csv_path.parent}")
            continue

        date_folder = csv_path.parent.parent.name  # e.g. '16-08'
        filename = csv_path.stem

        try:
            vgs_vals, id_vals = parse_csv(csv_path)
        except Exception as e:
            print(f"  [ERROR] {csv_path.name}: {e}")
            continue

        if not id_vals:
            print(f"  [WARN] no data in {csv_path.name}")
            continue

        id_on, id_off, on_off_ratio = compute_metrics(vgs_vals, id_vals)

        rows.append({
            "date":          date_folder,
            "layers":        layers,
            "filename":      filename,
            "id_on_A":       id_on,
            "id_off_A":      id_off,
            "on_off_ratio":  on_off_ratio,
            "vgs_min_V":     min(vgs_vals),
            "vgs_max_V":     max(vgs_vals),
            "n_points":      len(vgs_vals),
        })

    if not rows:
        print("No data found — check PROBE_ROOT path.")
        return

    # Sort by layer count, then date, then filename
    rows.sort(key=lambda r: (r["layers"], r["date"], r["filename"]))

    # Write output CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "layers", "filename",
                  "id_on_A", "id_off_A", "on_off_ratio",
                  "vgs_min_V", "vgs_max_V", "n_points"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {OUT_CSV}")

    # Quick summary per layer
    from collections import defaultdict
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layers"]].append(r)

    print("\n--- Summary (median per layer) ---")
    print(f"{'layers':>8}  {'n_devices':>10}  {'median_id_on':>14}  {'median_id_off':>14}  {'median_on_off':>14}")
    for l in sorted(by_layer):
        devs = by_layer[l]
        def med(vals):
            s = sorted(v for v in vals if math.isfinite(v))
            return s[len(s)//2] if s else float("nan")
        m_on   = med([d["id_on_A"]      for d in devs])
        m_off  = med([d["id_off_A"]     for d in devs])
        m_rat  = med([d["on_off_ratio"] for d in devs])
        print(f"{l:>8}  {len(devs):>10}  {m_on:>14.3e}  {m_off:>14.3e}  {m_rat:>14.1f}")


if __name__ == "__main__":
    main()
