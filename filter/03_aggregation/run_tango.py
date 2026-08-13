"""
run_tango.py
批次對 196 條序列執行 TANGO，輸出彙整結果。

正確過濾標準（來自 TANGO 官方文件）：
  任何連續 5-6 個殘基的 Aggregation > 5% → 視為 APR（aggregation-prone region）→ FAIL
"""

import csv, subprocess
from pathlib import Path

HERE   = Path(__file__).parent
TANGO  = HERE / "tools" / "tango_x86_64_release"
INPUT  = HERE.parent / "sequences.csv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)

# TANGO 執行條件（生理條件）
PH   = "7.4"
TEMP = "310"   # Kelvin (37°C)
IO   = "0.1"   # ionic strength (M)
CT   = "Y"     # C-terminus amidation (amidated, -NH2)
NT   = "N"     # N-terminus free (no protection)

# 過濾標準（TANGO 官方 rule of thumb）
APR_THRESHOLD  = 5.0  # 每個殘基 Aggregation > 5%
APR_MIN_LENGTH = 5    # 連續幾個殘基以上才算 APR


def run_tango(seq_id: str, seq: str) -> dict:
    cmd = [
        str(TANGO), seq_id,
        f"ct={CT}", f"nt={NT}",
        f"ph={PH}", f"te={TEMP}", f"io={IO}",
        f"seq={seq}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=30, cwd=str(OUTDIR))
        summary = {}
        parts = result.stdout.strip().split()
        for i in range(0, len(parts)-1, 2):
            summary[parts[i]] = float(parts[i+1])
        return summary
    except Exception as e:
        print(f"  [ERROR] {seq_id}: {e}")
        return {}


def parse_per_residue(seq_id: str) -> list:
    """讀取 per-residue txt，回傳每個殘基的 Aggregation 分數。"""
    txt_path = OUTDIR / f"{seq_id}.txt"
    scores = []
    if not txt_path.exists():
        return scores
    with open(txt_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                scores.append(float(parts[5]))  # Aggregation column
    return scores


def find_aprs(scores: list,
              threshold: float = APR_THRESHOLD,
              min_len: int = APR_MIN_LENGTH) -> list:
    """找出連續 >= min_len 個殘基都 > threshold 的 APR 片段。"""
    aprs = []
    in_apr = False
    start = 0
    for i, s in enumerate(scores):
        if s > threshold:
            if not in_apr:
                in_apr = True
                start = i
        else:
            if in_apr:
                length = i - start
                if length >= min_len:
                    aprs.append((start + 1, i, length,
                                 max(scores[start:i])))
                in_apr = False
    if in_apr:
        length = len(scores) - start
        if length >= min_len:
            aprs.append((start + 1, len(scores), length,
                         max(scores[start:])))
    return aprs


TRACKER = HERE.parent / "filter_tracker.csv"


def update_tracker(tango_results: list):
    """將 TANGO 結果（1=PASS, 0=FAIL）寫入 filter_tracker.csv 的 aggregation 欄。
    若欄位不存在則新增；若 tracker 不存在則從 sequences.csv 建立。
    """
    # 建立 {id: pass} 對照表
    tango_map = {r['id']: 1 if r['TANGO_pass'] == 'PASS' else 0
                 for r in tango_results}

    # 讀現有 tracker（若不存在則從 sequences.csv 初始化）
    if TRACKER.exists():
        rows = list(csv.DictReader(open(TRACKER)))
        fields = list(rows[0].keys()) if rows else ['id', 'sequence']
    else:
        seq_rows = list(csv.DictReader(open(INPUT)))
        rows = [{'id': r['id'], 'sequence': r['full_37aa']} for r in seq_rows]
        fields = ['id', 'sequence']

    # 確保 aggregation 欄存在
    if 'aggregation' not in fields:
        fields.append('aggregation')
    for row in rows:
        if 'aggregation' not in row:
            row['aggregation'] = ''

    # 寫入結果
    for row in rows:
        if row['id'] in tango_map:
            row['aggregation'] = tango_map[row['id']]

    with open(TRACKER, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(tango_map.values())
    print(f"filter_tracker.csv 已更新 → aggregation: {n_pass} PASS / {len(tango_map)} 總計")


def main():
    sequences = list(csv.DictReader(open(INPUT)))
    print(f"共 {len(sequences)} 條序列，開始跑 TANGO...")
    print(f"條件: pH={PH}, T={TEMP}K, IS={IO}M, CT={CT}, NT={NT}")
    print(f"過濾標準: 無連續 ≥{APR_MIN_LENGTH} 個殘基 Aggregation > {APR_THRESHOLD}%\n")

    results = []
    n_pass = 0

    for i, row in enumerate(sequences, 1):
        seq_id = row['id']
        seq    = row['full_37aa']

        summary = run_tango(seq_id, seq)
        per_res = parse_per_residue(seq_id)
        aprs    = find_aprs(per_res)

        status = 'PASS' if len(aprs) == 0 else 'FAIL'
        if status == 'PASS':
            n_pass += 1

        apr_str = '; '.join(
            f"res{a[0]}-{a[1]}(len={a[2]},max={a[3]:.1f}%)"
            for a in aprs
        ) if aprs else 'none'

        result = {
            'id':         seq_id,
            'full_37aa':  seq,
            'AMY1R':      row['AMY1R'],
            'AMY2R':      row['AMY2R'],
            'AMY3R':      row['AMY3R'],
            'CTR':        row['CTR'],
            'AGG_total':  f"{summary.get('AGG', 0):.4f}",
            'n_APR':      len(aprs),
            'APR_detail': apr_str,
            'TANGO_pass': status,
        }
        results.append(result)
        print(f"  [{i:3d}/196] {seq_id} | n_APR={len(aprs)} | {status}"
              + (f" ← {apr_str}" if aprs else ""))

    # 儲存詳細結果
    fields = ['id','full_37aa','AMY1R','AMY2R','AMY3R','CTR',
              'AGG_total','n_APR','APR_detail','TANGO_pass']
    with open(OUTDIR / "tango_results.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    passed_rows = [r for r in results if r['TANGO_pass'] == 'PASS']
    with open(OUTDIR / "tango_passed.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(passed_rows)

    # 自動更新 filter_tracker.csv
    update_tracker(results)

    print(f"\n{'='*60}")
    print(f"結果: {n_pass}/{len(sequences)} 條通過")
    print(f"標準: 無連續 ≥{APR_MIN_LENGTH} 個殘基 Aggregation > {APR_THRESHOLD}%")
    print(f"完整結果 → outputs/tango_results.csv")
    print(f"通過序列 → outputs/tango_passed.csv")


if __name__ == "__main__":
    main()
