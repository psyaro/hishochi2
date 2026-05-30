"""
全国の駅（N02-25_Station.geojson）の標高を250mメッシュデータから一括算出し、
軽量な形式で保存する。

出力:
  station_elevation.parquet  - データ分析用 (Parquet / pandas 直読み可)
  station_elevation.csv      - データ分析用 (汎用バックアップ)
  station_elevation.json     - サーバー配信用 (駅コードをキーとした辞書 / gzip同梱)

標高データ出典: 国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）
  https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html を加工して作成。
"""

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

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # hishochi2/

INPUT_GEOJSON  = _os.path.join(_ROOT, "downloads/N02-25_GML/UTF-8/N02-25_Station.geojson")
INPUT_AVG_NPZ  = _os.path.join(_ROOT, "data/japan_elevation_data.npz")
INPUT_MIN_NPZ  = _os.path.join(_ROOT, "data/japan_elevation_data_min.npz")
OUT_PARQUET    = _os.path.join(_ROOT, "data/station_elevation.parquet")
OUT_CSV        = _os.path.join(_ROOT, "data/station_elevation.csv")
OUT_JSON_GZ    = _os.path.join(_ROOT, "data/station_elevation.json.gz")


# ── Step 1: GeoJSON 読み込み＆重心計算 ────────────────────────────────────

def load_stations(path: str) -> pd.DataFrame:
    """
    N02-25_Station.geojson を読み込む。
    各フィーチャーは駅区間を表す LineString なので重心（全頂点の平均）を駅位置とする。
    同一駅コードが複数フィーチャーに分かれている場合は重心の平均を取る。
    """
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    records = []
    for feat in gj["features"]:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]  # [[lon, lat], ...]
        lon_c = sum(c[0] for c in coords) / len(coords)
        lat_c = sum(c[1] for c in coords) / len(coords)
        records.append({
            "station_code": props["N02_005c"],  # 駅コード (6桁)
            "station_name": props["N02_005"],
            "line_name":    props["N02_003"],
            "operator":     props["N02_004"],
            "line_type":    props["N02_001"],   # 路線種別
            "lon": lon_c,
            "lat": lat_c,
        })

    df = pd.DataFrame(records)

    # 同一駅コードを集約: 緯度経度は平均、名前系は最初の値を採用
    df = df.groupby("station_code", sort=False).agg(
        station_name=("station_name", "first"),
        line_name   =("line_name",    "first"),
        operator    =("operator",     "first"),
        line_type   =("line_type",    "first"),
        lon         =("lon",          "mean"),
        lat         =("lat",          "mean"),
    ).reset_index()

    print(f"[Step 1] 駅数: {len(df)} (元フィーチャー数: {len(records)})")
    return df


# ── Step 2: 標高データのロード ─────────────────────────────────────────────

def load_elevation_grids():
    print("[Step 2] 標高グリッドをメモリへロード中...")
    t0 = time.time()
    elev_avg = np.load(INPUT_AVG_NPZ)["elev"]
    elev_min = np.load(INPUT_MIN_NPZ)["elev"]
    print(f"         完了 ({time.time()-t0:.1f}s)  shape: {elev_avg.shape}")
    return elev_avg, elev_min


# ── Step 3: 緯度経度 → 標高のベクトル化ルックアップ ──────────────────────

def lookup_elevations(df: pd.DataFrame, elev_avg: np.ndarray, elev_min: np.ndarray) -> pd.DataFrame:
    """
    pandas の Series でグリッドインデックスを一括計算し、
    fancy indexing で全駅の標高を一括取得する。
    """
    print("[Step 3] 標高ルックアップ中...")
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # 範囲外チェック
    in_range = (
        (lats >= MIN_LAT) & (lats <= MAX_LAT) &
        (lons >= MIN_LON) & (lons <= MAX_LON)
    )
    out_of_range = (~in_range).sum()
    if out_of_range:
        print(f"  警告: {out_of_range} 駅が範囲外（海外？）→ NaN にします")

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

    # 範囲外 → NaN
    avg_vals[~in_range] = np.nan
    min_vals[~in_range] = np.nan

    df = df.copy()
    df["elev_avg_m"] = np.where(np.isfinite(avg_vals), avg_vals.round(1), np.nan)
    df["elev_min_m"] = np.where(np.isfinite(min_vals), min_vals.round(1), np.nan)
    df["is_ocean"]   = np.isnan(df["elev_avg_m"])

    land = (~df["is_ocean"]).sum()
    print(f"         完了: 陸地 {land} 駅 / 海域or未計測 {df['is_ocean'].sum()} 駅")
    return df


# ── Step 4: 保存 ───────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame):
    # 4-a) Parquet (データ分析用)
    df.to_parquet(OUT_PARQUET, index=False)
    import os
    size_p = os.path.getsize(OUT_PARQUET) / 1024
    print(f"[Step 4a] Parquet 保存: {OUT_PARQUET}  ({size_p:.1f} KB)")

    # 4-b) CSV (データ分析用バックアップ)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    size_c = os.path.getsize(OUT_CSV) / 1024
    print(f"[Step 4b] CSV 保存:    {OUT_CSV}  ({size_c:.1f} KB)")

    # 4-c) JSON.gz (サーバー配信用)
    #   フォーマット: { "駅コード": [lon, lat, elev_avg|null, elev_min|null], ... }
    #   キーは station_code、値は配列で最小バイト数に圧縮
    payload = {}
    for row in df.itertuples(index=False):
        avg = None if (row.is_ocean or not np.isfinite(row.elev_avg_m)) else row.elev_avg_m
        mn  = None if (row.is_ocean or not np.isfinite(row.elev_min_m)) else row.elev_min_m
        payload[row.station_code] = {
            "name": row.station_name,
            "line": row.line_name,
            "lon":  round(row.lon, 5),
            "lat":  round(row.lat, 5),
            "avg":  avg,
            "min":  mn,
        }

    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_JSON_GZ, "wb", compresslevel=9) as gz:
        gz.write(json_bytes)
    size_j_raw = len(json_bytes) / 1024
    size_j_gz  = os.path.getsize(OUT_JSON_GZ) / 1024
    print(f"[Step 4c] JSON.gz 保存: {OUT_JSON_GZ}  "
          f"(raw {size_j_raw:.0f} KB → gzip {size_j_gz:.0f} KB)")


# ── メイン ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 55)
    print("全国駅標高算出スクリプト")
    print("=" * 55)

    df             = load_stations(INPUT_GEOJSON)
    elev_avg, elev_min = load_elevation_grids()
    df             = lookup_elevations(df, elev_avg, elev_min)
    save_outputs(df)

    print("-" * 55)
    print(f"完了！  総処理時間: {time.time()-t_start:.1f}s")
    print()
    print("出力ファイル:")
    print(f"  {OUT_PARQUET}  - データ分析用 (pandas: pd.read_parquet)")
    print(f"  {OUT_CSV}       - データ分析用 (汎用CSV)")
    print(f"  {OUT_JSON_GZ}  - サーバー配信用 (gzip JSON)")
    print()
    print("標高データ出典: 国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）")

    # サマリー統計
    df_land = df[~df["is_ocean"]]
    print()
    print("── 標高サマリー（陸地の駅）────────────────")
    print(df_land[["elev_avg_m", "elev_min_m"]].describe().round(1).to_string())
    top10 = df_land.nlargest(5, "elev_avg_m")[["station_name","line_name","elev_avg_m","elev_min_m"]]
    print("\n平均標高トップ5:")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
