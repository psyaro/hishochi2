"""
全国のバス停（bus_elevation.parquet）について、
250mメッシュ標高データからレイキャスティングで天空率（sky_openness）を算出し、
bus_elevation.parquet に svf 列として追記する。
さらに docs/data/bus_elevation.json.gz を svf 込みで再生成する。

sky_openness = mean(1 - sin(H_i))
  H_i: 各方向の最大地平線仰角（地球曲率補正済み）
  H=0°(平地)→1.0 / H=14°(500m山が2km先)→0.76 / H=30°(深い谷)→0.50
"""

import os
import json
import gzip
import time
import numpy as np
import pandas as pd

# ── グリッドパラメータ ─────────────────────────────────────────────────────
MIN_LAT, MAX_LAT = 20.0, 46.0
MIN_LON, MAX_LON = 122.0, 155.0
GRID_HEIGHT, GRID_WIDTH = 12480, 10560
DY = 7.5  / 3600
DX = 11.25 / 3600

# ── 計算パラメータ（バス停は多いので駅より少し粗く） ──────────────────────
N_DIRS      = 12      # レイ方向数
MAX_DIST_KM = 25.0   # 最大参照距離 (km)
STEP_KM     = 0.5    # ステップ幅 (km)

R_EARTH = 6_371_000.0

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PARQUET  = os.path.join(_ROOT, "data", "bus_elevation.parquet")
IN_ELEV_NPZ = os.path.join(_ROOT, "data", "japan_elevation_data.npz")
OUT_PARQUET = IN_PARQUET
OUT_JSON_GZ = os.path.join(_ROOT, "docs", "data", "bus_elevation.json.gz")


def latlon_to_grid(lats, lons):
    col     = np.floor((lons - MIN_LON) / DX).astype(int)
    row     = np.floor((lats - MIN_LAT) / DY).astype(int)
    img_row = GRID_HEIGHT - 1 - row
    return np.clip(img_row, 0, GRID_HEIGHT - 1), np.clip(col, 0, GRID_WIDTH - 1)


def calc_sky_openness(df: pd.DataFrame, elev_grid: np.ndarray) -> np.ndarray:
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    N    = len(df)

    img_row, img_col = latlon_to_grid(lats, lons)
    station_elev = elev_grid[img_row, img_col].astype(float)

    azimuths       = np.linspace(0, 2 * np.pi, N_DIRS, endpoint=False)
    n_steps        = int(MAX_DIST_KM / STEP_KM)
    step_dists_m   = np.arange(1, n_steps + 1) * STEP_KM * 1000.0
    curvature_drop = step_dists_m ** 2 / (2.0 * R_EARTH)
    cos_lat        = np.cos(np.radians(lats))

    max_horizon = np.zeros((N, N_DIRS), dtype=np.float32)

    print(f"  {N_DIRS}方向 × {n_steps}ステップ = {N_DIRS * n_steps} 反復")
    t0 = time.time()

    for di, az in enumerate(azimuths):
        cos_az, sin_az = np.cos(az), np.sin(az)
        for si in range(1, n_steps + 1):
            dist_km = si * STEP_KM
            dist_m  = step_dists_m[si - 1]

            dlat = dist_km / 111.0 * cos_az
            dlon = dist_km / 111.0 * sin_az / cos_lat

            s_lats = lats + dlat
            s_lons = lons + dlon
            in_range = (
                (s_lats >= MIN_LAT) & (s_lats <= MAX_LAT) &
                (s_lons >= MIN_LON) & (s_lons <= MAX_LON)
            )

            s_row, s_col = latlon_to_grid(s_lats, s_lons)
            sample_elev  = elev_grid[s_row, s_col].astype(float)

            elev_diff     = (sample_elev - station_elev) - curvature_drop[si - 1]
            horizon_angle = np.arctan2(elev_diff, dist_m)
            horizon_angle = np.where(in_range & (horizon_angle > 0), horizon_angle, 0.0)

            np.maximum(max_horizon[:, di], horizon_angle, out=max_horizon[:, di])

        elapsed = time.time() - t0
        print(f"  方向 {di+1:2d}/{N_DIRS} 完了 ({elapsed:.1f}s)")

    sky = np.mean(1.0 - np.sin(max_horizon.astype(np.float64)), axis=1)
    print(f"  完了 (総時間 {time.time()-t0:.1f}s)")
    return np.clip(sky, 0.0, 1.0).astype(np.float32)


def regen_json(df: pd.DataFrame):
    """bus_elevation.json.gz を svf 込みで再生成する。"""
    payload = []
    for row in df.itertuples(index=False):
        is_ocean = bool(row.is_ocean)
        avg = None if (is_ocean or not np.isfinite(row.elev_avg_m))  else row.elev_avg_m
        mn  = None if (is_ocean or not np.isfinite(row.elev_min_m))  else row.elev_min_m
        tmp = None if (is_ocean or not np.isfinite(row.temp_max_aug)) else row.temp_max_aug
        sqm = None if (is_ocean or not np.isfinite(row.sqm))         else row.sqm
        svf = None if (is_ocean or not np.isfinite(row.svf))         else row.svf
        payload.append({
            "name":     row.bus_stop_name,
            "operator": row.operator,
            "lon":      round(row.lon, 5),
            "lat":      round(row.lat, 5),
            "avg":      avg,
            "min":      mn,
            "temp":     tmp,
            "sqm":      sqm,
            "svf":      svf,
        })

    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_JSON_GZ, "wb", compresslevel=9) as gz:
        gz.write(json_bytes)

    raw_mb = len(json_bytes) / (1024 * 1024)
    gz_mb  = os.path.getsize(OUT_JSON_GZ) / (1024 * 1024)
    print(f"  JSON.gz 保存: {OUT_JSON_GZ}  (raw {raw_mb:.2f}MB → gz {gz_mb:.2f}MB)")


def main():
    print("=" * 55)
    print("バス停天空率算出スクリプト")
    print("=" * 55)
    t_start = time.time()

    print(f"[1/4] Parquet 読み込み: {IN_PARQUET}")
    df = pd.read_parquet(IN_PARQUET)
    print(f"      バス停数: {len(df)}")

    print(f"[2/4] 標高グリッド読み込み: {IN_ELEV_NPZ}")
    t0 = time.time()
    elev_grid = np.load(IN_ELEV_NPZ)["elev"]
    print(f"      完了 ({time.time()-t0:.1f}s)  shape: {elev_grid.shape}")

    land_mask = ~df["is_ocean"].to_numpy()
    df_land   = df[land_mask].reset_index(drop=True)
    print(f"      陸地バス停: {land_mask.sum()} / {len(df)}")

    print("[3/4] 天空率を計算中...")
    sky_land = calc_sky_openness(df_land, elev_grid)

    svf_all = np.full(len(df), np.nan, dtype=np.float32)
    svf_all[land_mask] = sky_land
    df["svf"] = np.where(np.isfinite(svf_all), svf_all.round(3), np.nan)

    print(f"[4/4] 保存中...")
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"      Parquet: {OUT_PARQUET}")
    regen_json(df)

    print(f"\n完了！  総処理時間: {time.time()-t_start:.1f}s")

    sv = df["svf"].dropna()
    print(f"\n── SVF サマリー ─────────────────────────")
    print(sv.describe().round(3).to_string())


if __name__ == "__main__":
    main()
