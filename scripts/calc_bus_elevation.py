"""
全国のバス停（P11-22_SHP）の標高を250mメッシュデータから一括算出し、
軽量な形式で保存する。

出力:
  bus_elevation.parquet  - データ分析用 (Parquet / pandas 直読み可)
  bus_elevation.csv      - データ分析用 (汎用バックアップ)
  bus_elevation.json.gz  - サーバー配信用 (主要属性のみのリスト / gzip圧縮)

標高データ出典: 国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）
"""

import os
import glob
import json
import gzip
import time
import numpy as np
import pandas as pd

# ── グリッドパラメータ (integrate_elev.py / server.py と一致) ──────────────
MIN_LAT, MAX_LAT = 20.0, 46.0
MIN_LON, MAX_LON = 122.0, 155.0
GRID_HEIGHT, GRID_WIDTH = 12480, 10560
DY = 7.5 / 3600     # 緯度方向の1画素幅 (度)
DX = 11.25 / 3600   # 経度方向の1画素幅 (度)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hishochi2/

INPUT_GEOJSON_DIR = os.path.join(_ROOT, "downloads/P11-22_SHP")
INPUT_AVG_NPZ     = os.path.join(_ROOT, "data/japan_elevation_data.npz")
INPUT_MIN_NPZ     = os.path.join(_ROOT, "data/japan_elevation_data_min.npz")
INPUT_TEMP_NPZ    = os.path.join(_ROOT, "data/japan_aug_max_temp_data.npz")
OUT_PARQUET       = os.path.join(_ROOT, "data/bus_elevation.parquet")
OUT_CSV           = os.path.join(_ROOT, "data/bus_elevation.csv")
OUT_JSON_GZ       = os.path.join(_ROOT, "docs/data/bus_elevation.json.gz")


# ── Step 1: GeoJSON 読み込み＆統合 ──────────────────────────────────────────

def load_bus_stops(dir_path: str) -> pd.DataFrame:
    """
    downloads/P11-22_SHP/ 配下の P11-22_*.geojson (47都道府県) を順次読み込む。
    """
    geojson_files = sorted(glob.glob(os.path.join(dir_path, "P11-22_*.geojson")))
    if not geojson_files:
        raise FileNotFoundError(f"GeoJSONファイルが {dir_path} 内に見つかりませんでした。")

    print(f"[Step 1] GeoJSONファイルの検出数: {len(geojson_files)} 件")
    records = []
    
    t_start = time.time()
    for idx, path in enumerate(geojson_files, 1):
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
        
        file_records_count = 0
        for feat in gj["features"]:
            props = feat["properties"]
            geom = feat["geometry"]
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom["coordinates"]  # [lon, lat]
            
            rec = {
                "bus_stop_name": props.get("P11_001"),
                "operator":      props.get("P11_002"),
                "lon":           coords[0],
                "lat":           coords[1],
            }
            # 系統名(P11_003_xx)と区分コード(P11_004_xx)をループで取得
            for i in range(1, 36):
                col_name_3 = f"P11_003_{i:02d}"
                col_name_4 = f"P11_004_{i:02d}"
                rec[col_name_3] = props.get(col_name_3)
                rec[col_name_4] = props.get(col_name_4)
            
            rec["note"] = props.get("P11_005")
            records.append(rec)
            file_records_count += 1
            
        print(f"         ({idx}/{len(geojson_files)}) {filename} ロード完了: {file_records_count} 件")

    df = pd.DataFrame(records)
    print(f"[Step 1] ロード完了 (所要時間: {time.time()-t_start:.1f}s)  総バス停数: {len(df)}")
    return df


# ── Step 2: 標高データのロード ─────────────────────────────────────────────

def load_elevation_grids():
    print("[Step 2] 標高および気温グリッドをメモリへロード中...")
    t0 = time.time()
    elev_avg = np.load(INPUT_AVG_NPZ)["elev"]
    elev_min = np.load(INPUT_MIN_NPZ)["elev"]
    temp_max = np.load(INPUT_TEMP_NPZ)["temp"]
    print(f"         完了 ({time.time()-t0:.1f}s)  shape: {elev_avg.shape}")
    return elev_avg, elev_min, temp_max


# ── Step 3: 緯度経度 → 標高のベクトル化ルックアップ ──────────────────────

def lookup_elevations(df: pd.DataFrame, elev_avg: np.ndarray, elev_min: np.ndarray, temp_max: np.ndarray) -> pd.DataFrame:
    """
    pandas の Series でグリッドインデックスを一括計算し、
    fancy indexing で全バス停の標高を一括取得する。
    """
    print("[Step 3] 標高および気温ルックアップ中...")
    t0 = time.time()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # 範囲外チェック
    in_range = (
        (lats >= MIN_LAT) & (lats <= MAX_LAT) &
        (lons >= MIN_LON) & (lons <= MAX_LON)
    )
    out_of_range = (~in_range).sum()
    if out_of_range:
        print(f"  警告: {out_of_range} バス停が日本領土の範囲外 → NaN にします")

    # グリッドインデックスの計算
    col = np.floor((lons - MIN_LON) / DX).astype(int)
    row = np.floor((lats - MIN_LAT) / DY).astype(int)
    img_row = GRID_HEIGHT - 1 - row
    img_col = col

    # 範囲外はクランプ（後で NaN 上書き）
    img_row_c = np.clip(img_row, 0, GRID_HEIGHT - 1)
    img_col_c = np.clip(img_col, 0, GRID_WIDTH  - 1)

    # 一括ルックアップ
    avg_vals = elev_avg[img_row_c, img_col_c]
    min_vals = elev_min[img_row_c, img_col_c]
    temp_vals = temp_max[img_row_c, img_col_c]

    # 範囲外 → NaN
    avg_vals[~in_range] = np.nan
    min_vals[~in_range] = np.nan
    temp_vals[~in_range] = np.nan

    df = df.copy()
    df["elev_avg_m"] = np.where(np.isfinite(avg_vals), avg_vals.round(1), np.nan)
    df["elev_min_m"] = np.where(np.isfinite(min_vals), min_vals.round(1), np.nan)
    df["temp_max_aug"] = np.where(np.isfinite(temp_vals), temp_vals.round(1), np.nan)
    df["is_ocean"]   = np.isnan(df["elev_avg_m"])

    land = (~df["is_ocean"]).sum()
    print(f"         完了 ({time.time()-t0:.1f}s): 陸地 {land} 件 / 海域or未計測 {df['is_ocean'].sum()} 件")
    return df


# ── Step 4: 保存 ───────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame):
    t0 = time.time()
    
    # 4-a) Parquet (データ分析用 - 全属性保持)
    print("[Step 4a] Parquet 保存中...")
    df.to_parquet(OUT_PARQUET, index=False)
    size_p = os.path.getsize(OUT_PARQUET) / (1024 * 1024)
    print(f"          完了: {OUT_PARQUET}  ({size_p:.2f} MB)")

    # 4-b) CSV (データ分析用バックアップ - 全属性保持)
    print("[Step 4b] CSV 保存中...")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    size_c = os.path.getsize(OUT_CSV) / (1024 * 1024)
    print(f"          完了: {OUT_CSV}  ({size_c:.2f} MB)")

    # 4-c) JSON.gz (サーバー配信用 - 主要属性のみのリスト)
    #   フォーマット: [ {"name": "...", "operator": "...", "lon": 139.xxx, "lat": 35.xxx, "avg": elev|null, "min": elev|null}, ... ]
    print("[Step 4c] 配信用 JSON.gz 作成および保存中...")
    payload = []
    for row in df.itertuples(index=False):
        avg = None if (row.is_ocean or not np.isfinite(row.elev_avg_m)) else row.elev_avg_m
        mn  = None if (row.is_ocean or not np.isfinite(row.elev_min_m)) else row.elev_min_m
        tmp = None if (row.is_ocean or not np.isfinite(row.temp_max_aug)) else row.temp_max_aug
        payload.append({
            "name":     row.bus_stop_name,
            "operator": row.operator,
            "lon":      round(row.lon, 5),
            "lat":      round(row.lat, 5),
            "avg":      avg,
            "min":      mn,
            "temp":     tmp,
        })

    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_JSON_GZ, "wb", compresslevel=9) as gz:
        gz.write(json_bytes)
        
    size_j_raw = len(json_bytes) / (1024 * 1024)
    size_j_gz  = os.path.getsize(OUT_JSON_GZ) / (1024 * 1024)
    print(f"          完了: {OUT_JSON_GZ}  (生JSON {size_j_raw:.2f} MB → gzip {size_j_gz:.2f} MB)")
    print(f"[Step 4] 全ファイルの保存完了 (所要時間: {time.time()-t0:.1f}s)")


# ── メイン ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 55)
    print("全国バス停標高算出・データ変換スクリプト")
    print("=" * 55)

    df                 = load_bus_stops(INPUT_GEOJSON_DIR)
    elev_avg, elev_min, temp_max = load_elevation_grids()
    df                 = lookup_elevations(df, elev_avg, elev_min, temp_max)
    save_outputs(df)

    print("-" * 55)
    print(f"すべて完了！ 総処理時間: {time.time()-t_start:.1f}s")
    print()
    print("出力ファイル:")
    print(f"  {OUT_PARQUET}  - データ分析用 (pandas: pd.read_parquet)")
    print(f"  {OUT_CSV}       - データ分析用 (汎用CSV)")
    print(f"  {OUT_JSON_GZ}  - サーバー配信用 (gzip JSON)")
    print()

    # サマリー統計
    df_land = df[~df["is_ocean"]]
    print("── 標高サマリー（陸地のバス停）────────────────")
    print(df_land[["elev_avg_m", "elev_min_m"]].describe().round(1).to_string())
    top5 = df_land.nlargest(5, "elev_avg_m")[["bus_stop_name", "operator", "elev_avg_m", "elev_min_m"]]
    print("\n平均標高トップ5のバス停:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
