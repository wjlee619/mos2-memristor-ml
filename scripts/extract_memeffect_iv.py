"""
Phase 1b: MemEffect IV curve feature extraction
Reads all CSV files from 15082024 folder and extracts memristor characteristics.
Original data is never modified — read-only.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "Desktop/Imperial/Mres Soft electronics/DATA/15082024"
OUTPUT_PATH = Path.home() / "Desktop/mos2-memristor-ml/data/processed/memeffect_sweep.csv"


# ─── helpers ──────────────────────────────────────────────────────────────────

def find_header_row(filepath: Path) -> int:
    """Return 0-based line index of the data header row."""
    with open(filepath, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Index" in line and ("Voltage" in line or "Current" in line):
                return i
    return -1


def parse_metadata(filepath: Path) -> dict:
    """Extract sweep settings from the Keithley header block."""
    meta = {"v_stop": None, "n_sweep_points": None, "dual_sweep": None,
            "smu_channel": None, "current_limit": None}
    with open(filepath, encoding="utf-8-sig") as f:
        lines = f.readlines()

    # walk header only (before data)
    header_end = find_header_row(filepath)
    current_channel = None
    for line in lines[:header_end]:
        line = line.strip()
        if line.startswith(".Channel"):
            current_channel = line.split(",")[1].strip()
        if line.startswith("...Stop"):
            val = line.split(",")[1].strip()
            if current_channel:          # last SMU's Stop wins (SMU-2 if present)
                meta["v_stop"] = float(val)
                meta["smu_channel"] = current_channel
        if line.startswith("...Limit") and current_channel:
            meta["current_limit"] = float(line.split(",")[1].strip())
        if line.startswith(".Source/Sweep Points"):
            meta["n_sweep_points"] = int(line.split(",")[1].strip())
        if line.startswith("...Dual Sweep"):
            meta["dual_sweep"] = line.split(",")[1].strip()
    return meta


def read_iv_data(filepath: Path, header_row: int) -> pd.DataFrame:
    """Read voltage/current columns, return clean DataFrame with V and I columns.

    Keithley appends summary rows (Min, Max, Mean, StdDev, CV) at the end of
    the CSV.  These have a non-integer index, so we filter them out by requiring
    the index column to be a positive integer.
    """
    df = pd.read_csv(filepath, skiprows=header_row, quotechar='"', encoding="utf-8-sig",
                     dtype=str)  # read all as str first for robust filtering
    # normalise column names
    v_col = [c for c in df.columns if "Voltage" in c][0]
    i_col = [c for c in df.columns if "Current" in c][0]
    idx_col = df.columns[0]  # "Index" column

    # keep only rows where index is a positive integer (drops Min/Max/Mean/…)
    def is_int(x):
        try:
            return int(x) > 0
        except (ValueError, TypeError):
            return False

    df = df[df[idx_col].apply(is_int)][[v_col, i_col]].copy()
    df.columns = ["voltage_V", "current_A"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.reset_index(drop=True)
    return df


def parse_filename(fname: str) -> dict:
    """Extract device/condition metadata from filename."""
    info = {
        "chip": None,
        "contact": None,
        "device_id": None,
        "condition": None,
        "run_number": None,
        "timestamp": None,
        "file_type": None,
    }

    # run-style: "Chip#1-CC1-T14-MemEffect-Dark I-V Characterizer-1 Run 33 2024-08-15T16.38.34"
    run_match = re.match(
        r"(Chip#\d+)-(CC\d+)-(T\d+)-([\w\-]+?)\s+I-V Characterizer-\d+ Run (\d+) ([\dT\.\-]+)",
        fname
    )
    if run_match:
        info["chip"] = run_match.group(1)
        info["contact"] = run_match.group(2)
        info["device_id"] = run_match.group(3)
        info["condition"] = run_match.group(4)
        info["run_number"] = int(run_match.group(5))
        info["timestamp"] = run_match.group(6)
        info["file_type"] = "run"
        return info

    # simple style: "Chip#1-CC1-T14-MemEffect12"
    simple_match = re.match(r"(Chip#\d+)-(CC\d+)-(T\d+)-([\w\-]+?)(\d*)$", fname)
    if simple_match:
        info["chip"] = simple_match.group(1)
        info["contact"] = simple_match.group(2)
        info["device_id"] = simple_match.group(3)
        info["condition"] = simple_match.group(4)
        run_str = simple_match.group(5)
        info["run_number"] = int(run_str) if run_str else 0
        info["file_type"] = "simple"
        return info

    return info


def detect_sweep_segments(voltage: np.ndarray):
    """
    Returns (fwd_mask, rev_mask) boolean arrays.
    Forward: first half from start to peak |V|.
    Reverse: second half from peak back to end.
    Handles both positive and negative sweeps.
    """
    abs_v = np.abs(voltage)
    peak_idx = int(np.argmax(abs_v))
    fwd_mask = np.zeros(len(voltage), dtype=bool)
    rev_mask = np.zeros(len(voltage), dtype=bool)
    fwd_mask[:peak_idx + 1] = True
    rev_mask[peak_idx:] = True
    return fwd_mask, rev_mask


def smooth(arr: np.ndarray, window: int = 11) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def find_switching_voltage(voltage: np.ndarray, current: np.ndarray,
                           direction: str = "set",
                           min_decades_per_v: float = 0.5) -> float:
    """
    Detect SET or RESET voltage using steepest slope on log10(|I|) vs |V|.

    direction: "set"   → steepest positive slope in log|I| during forward sweep
               "reset" → steepest negative slope in log|I| during reverse sweep

    The reverse sweep is ordered peak→0, so |V| decreases with index.
    We use the slope w.r.t. index (uniform spacing), then check the sign.

    Returns the voltage at the steepest transition, or np.nan when the
    detected peak slope is below min_decades_per_v (in decades per volt).
    """
    if len(current) < 10:
        return np.nan

    abs_i = np.abs(current)
    log_i = np.log10(np.maximum(abs_i, 1e-16))
    log_i_smooth = smooth(log_i, window=max(7, len(log_i) // 80))

    # gradient w.r.t. index
    grad_idx = np.gradient(log_i_smooth)

    # convert to per-volt: need |dV/d_index|
    abs_v = np.abs(voltage)
    # avoid dividing by zero when V is near 0
    dv = np.gradient(abs_v)
    dv = np.where(np.abs(dv) < 1e-9, 1e-9, dv)
    grad_v = grad_idx / np.abs(dv)   # decades / V

    # skip 2% at each end (edge artefacts from smoothing)
    n = len(grad_v)
    trim = max(2, n // 50)
    search = grad_v[trim: n - trim]
    v_search = voltage[trim: n - trim]

    if direction == "set":
        peak = np.max(search)
        if peak < min_decades_per_v:
            return np.nan
        switch_idx = int(np.argmax(search))
    else:
        peak = np.min(search)
        if peak > -min_decades_per_v:
            return np.nan
        switch_idx = int(np.argmin(search))

    return float(v_search[switch_idx])


def find_reset_voltage_threshold(V_rev: np.ndarray, I_rev: np.ndarray,
                                  i_on: float, i_off: float) -> float:
    """
    RESET voltage via threshold crossing during the reverse sweep.

    Threshold = geometric mean of on-state and off-state currents.
    Returns the voltage where |I| first drops below that threshold,
    or np.nan if the current never crosses it cleanly.

    V_rev goes from peak → 0 (magnitude decreasing).
    """
    if len(I_rev) < 5 or np.isnan(i_on) or np.isnan(i_off) or i_off <= 0:
        return np.nan

    abs_i = np.abs(I_rev)
    # Geometric-mean threshold (midpoint on log scale between off and on)
    threshold = np.sqrt(i_on * i_off)

    # The reverse sweep starts near i_on and should end near i_off.
    # We want the first point where |I| drops BELOW threshold.
    # Skip the first 2% (smoothing artefacts near peak) and last 2% (noise near 0V).
    n = len(abs_i)
    trim = max(2, n // 50)
    search_i = abs_i[trim: n - trim]
    search_v = V_rev[trim: n - trim]

    below = np.where(search_i < threshold)[0]
    if len(below) == 0:
        return np.nan
    return float(search_v[below[0]])


def extract_features(filepath: Path) -> dict:
    """Main extraction routine for one file."""
    fname_stem = filepath.stem
    info = parse_filename(fname_stem)
    meta = parse_metadata(filepath)
    header_row = find_header_row(filepath)

    if header_row < 0:
        return {"filename": filepath.name, "error": "header not found", **info}

    df = read_iv_data(filepath, header_row)
    if len(df) < 5:
        return {"filename": filepath.name, "error": "too few data points", **info}

    V = df["voltage_V"].to_numpy()
    I = df["current_A"].to_numpy()
    n_total = len(V)

    v_max = float(np.max(np.abs(V)))
    sweep_sign = 1 if (meta["v_stop"] is not None and meta["v_stop"] > 0) else (
        -1 if (meta["v_stop"] is not None and meta["v_stop"] < 0)
        else int(np.sign(V[np.argmax(np.abs(V))]))
    )

    fwd_mask, rev_mask = detect_sweep_segments(V)
    V_fwd, I_fwd = V[fwd_mask], I[fwd_mask]
    V_rev, I_rev = V[rev_mask], I[rev_mask]
    n_fwd = len(I_fwd)
    has_return = rev_mask.sum() > 10

    # ── OFF current: min non-zero |I| in first 5% of forward sweep ──────────
    n_off = max(1, n_fwd // 20)
    off_region = np.abs(I_fwd[:n_off])
    off_region_nz = off_region[off_region > 0]
    i_off = float(np.min(off_region_nz)) if len(off_region_nz) > 0 else np.nan
    i_off_median = float(np.median(off_region_nz)) if len(off_region_nz) > 0 else np.nan

    # ── ON current: 95th-pct |I| in last 10% of forward sweep ───────────────
    n_on = max(1, n_fwd // 10)
    on_region = np.abs(I_fwd[-n_on:])
    i_on = float(np.percentile(on_region, 95)) if len(on_region) > 0 else np.nan

    # ── Switching state classification ───────────────────────────────────────
    # "already_on": device starts with high current (median of first 5% > 1% of peak)
    already_on = (
        not np.isnan(i_on) and not np.isnan(i_off_median) and i_on > 0
        and i_off_median > i_on * 0.01
    )
    # "low_voltage_sweep": small diagnostic sweep, no switching possible
    low_v = v_max < 2.0 and n_total < 500

    if low_v:
        switching_state = "low_voltage_sweep"
        v_set = np.nan
        v_reset = np.nan
    elif already_on:
        switching_state = "already_on"
        v_set = np.nan
        v_reset = np.nan
    else:
        # attempt SET detection (log-derivative method)
        v_set = find_switching_voltage(V_fwd, I_fwd, direction="set")
        # attempt RESET via threshold crossing
        v_reset = np.nan
        if has_return and not np.isnan(i_on) and not np.isnan(i_off):
            v_reset = find_reset_voltage_threshold(V_rev, I_rev, i_on, i_off)

        if not np.isnan(v_set):
            switching_state = "switched"
        else:
            switching_state = "no_switching_detected"

    on_off_ratio = (
        (i_on / i_off) if (not np.isnan(i_on) and not np.isnan(i_off) and i_off > 0)
        else np.nan
    )

    hysteresis_V = (
        abs(v_set - v_reset)
        if not (np.isnan(v_set) or np.isnan(v_reset))
        else np.nan
    )

    return {
        "filename": filepath.name,
        "chip": info["chip"],
        "contact": info["contact"],
        "device_id": info["device_id"],
        "condition": info["condition"],
        "run_number": info["run_number"],
        "timestamp": info["timestamp"],
        "file_type": info["file_type"],
        "smu_channel": meta["smu_channel"],
        "n_points": n_total,
        "dual_sweep": meta["dual_sweep"],
        "v_max_V": round(v_max, 4),
        "sweep_sign": sweep_sign,
        "switching_state": switching_state,
        "v_set_V": round(v_set, 4) if not np.isnan(v_set) else np.nan,
        "v_reset_V": round(v_reset, 4) if not np.isnan(v_reset) else np.nan,
        "i_on_A": float(f"{i_on:.6e}") if not np.isnan(i_on) else np.nan,
        "i_off_A": float(f"{i_off:.6e}") if not np.isnan(i_off) else np.nan,
        "on_off_ratio": round(on_off_ratio, 2) if not np.isnan(on_off_ratio) else np.nan,
        "hysteresis_window_V": round(hysteresis_V, 4) if not np.isnan(hysteresis_V) else np.nan,
        "current_limit_A": meta["current_limit"],
        "error": None,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    records = []
    for fp in csv_files:
        print(f"  Processing: {fp.name}")
        try:
            rec = extract_features(fp)
        except Exception as e:
            rec = {"filename": fp.name, "error": str(e)}
        records.append(rec)

    df_out = pd.DataFrame(records)

    # reorder columns for clarity
    col_order = [
        "filename", "chip", "contact", "device_id", "condition",
        "run_number", "timestamp", "file_type", "smu_channel",
        "n_points", "dual_sweep", "sweep_sign", "v_max_V",
        "v_set_V", "v_reset_V", "i_on_A", "i_off_A",
        "on_off_ratio", "hysteresis_window_V", "current_limit_A", "error",
    ]
    col_order.insert(col_order.index("v_set_V"), "switching_state")
    present = [c for c in col_order if c in df_out.columns]
    extra = [c for c in df_out.columns if c not in col_order]
    df_out = df_out[present + extra]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUTPUT_PATH}")

    # quick summary
    print("\n── Feature summary ──")
    numeric_cols = [c for c in ["v_set_V", "v_reset_V", "i_on_A", "i_off_A",
                                "on_off_ratio", "hysteresis_window_V"]
                    if c in df_out.columns]
    with pd.option_context("display.float_format", "{:.3e}".format):
        print(df_out[numeric_cols].describe())

    n_switched = df_out["v_set_V"].notna().sum()
    print(f"\nFiles with detected SET event : {n_switched}/{len(df_out)}")
    n_reset = df_out["v_reset_V"].notna().sum()
    print(f"Files with detected RESET event: {n_reset}/{len(df_out)}")
    print(f"\nErrors: {df_out['error'].notna().sum()}")
    if df_out["error"].notna().any():
        print(df_out.loc[df_out["error"].notna(), ["filename", "error"]])


if __name__ == "__main__":
    main()
