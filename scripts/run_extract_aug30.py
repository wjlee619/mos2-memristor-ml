"""
Phase 1c: 2024-08-30 데이터 처리
extract_memeffect_iv.py의 함수들을 임포트해서 새 경로로 실행.
원본 스크립트·원본 데이터 모두 수정하지 않음.
"""

import sys
from pathlib import Path

# 스크립트 디렉토리를 sys.path에 추가해서 임포트
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import numpy as np

# 원본 스크립트의 모든 함수 임포트 (DATA_DIR/OUTPUT_PATH 제외)
from extract_memeffect_iv import (
    find_header_row,
    parse_metadata,
    read_iv_data,
    parse_filename,
    detect_sweep_segments,
    smooth,
    find_switching_voltage,
    find_reset_voltage_threshold,
    extract_features,
)

# ─── 새 경로 정의 ──────────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / "Desktop/Imperial/Mres Soft electronics/DATA/2024-08-30"
OUTPUT_PATH = Path.home() / "Desktop/mos2-memristor-ml/data/processed/memeffect_sweep_aug30.csv"

# ─── 실행 ─────────────────────────────────────────────────────────────────────
csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in {DATA_DIR.name}/")

records = []
for fp in csv_files:
    print(f"  Processing: {fp.name}")
    try:
        rec = extract_features(fp)
    except Exception as e:
        rec = {"filename": fp.name, "error": str(e)}
    records.append(rec)

df_out = pd.DataFrame(records)

# 컬럼 순서 정렬
col_order = [
    "filename", "chip", "contact", "device_id", "condition",
    "run_number", "timestamp", "file_type", "smu_channel",
    "n_points", "dual_sweep", "sweep_sign", "v_max_V",
    "switching_state", "v_set_V", "v_reset_V", "i_on_A", "i_off_A",
    "on_off_ratio", "hysteresis_window_V", "current_limit_A", "error",
]
present = [c for c in col_order if c in df_out.columns]
extra   = [c for c in df_out.columns if c not in col_order]
df_out  = df_out[present + extra]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(df_out)} rows → {OUTPUT_PATH}")

# ─── switching_state 분포 ──────────────────────────────────────────────────────
print("\n── switching_state 분포 ──")
state_counts = df_out["switching_state"].value_counts()
total = len(df_out)
for state, cnt in state_counts.items():
    bar = "█" * cnt
    print(f"  {state:28s}  {cnt:3d} ({cnt/total*100:4.1f}%)  {bar}")

# ─── 수치 요약 ─────────────────────────────────────────────────────────────────
print("\n── Feature summary ──")
numeric_cols = [c for c in ["v_set_V", "v_reset_V", "i_on_A", "i_off_A",
                             "on_off_ratio", "hysteresis_window_V"]
                if c in df_out.columns]
with pd.option_context("display.float_format", "{:.3e}".format):
    print(df_out[numeric_cols].describe())

print(f"\nErrors: {df_out['error'].notna().sum()}")
if df_out["error"].notna().any():
    print(df_out.loc[df_out["error"].notna(), ["filename", "error"]])
