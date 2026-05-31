"""
全国の駅（station_elevation.parquet）について、
250mメッシュ標高データからレイキャスティングで天空率（Sky View Factor / SVF）を算出し、
station_elevation.parquet に svf 列として追記する。

天空率（SVF）: 地点から見える天球半球のうち地形に遮られていない割合（0〜1）
  SVF = (1/N_DIRS) × Σ cos²(最大地平線仰角_i)
  完全に開けた平地: SVF≒1.0 / 山に囲まれた谷: SVF≒0.5以下

入力:
  data/station_elevation.parquet   (calc_station_elevation.py の出力)
  data/japan_elevation_data.npz    (統合済み標高グリッド)

出力:
  data/station_elevation.parquet   (svf 列を追記して上書き)
"""

import os
import time
import numpy as np
import pandas as pd

# ── グリッドパラメータ (calc_station_elevation.py と同一) ───────────────────
MIN_LAT, MAX_LAT = 20.0, 46.0
MIN_LON, MAX_LON = 122.0, 155.0
GRID_HEIGHT, GRID_WIDTH = 12480, 10560
DY = 7.5  / 3600   # 緯度方向の1画素幅 (度)
DX = 11.25 / 3600  # 経度方向の1画素幅 (度)
DY_M = DY * 111_000.0            # 緯度方向の1画素幅 (m) ≈ 231m
# 経度方向は後で各駅の緯度を使って補正

# ── SVF 計算パラメータ ─────────────────────────────────────────────────────
N_DIRS     = 16       # レイ方向数 (等間隔アジマス)
MAX_DIST_KM = 30.0   # 最大参照距離 (km)
STEP_KM    = 0.25    # ステップ幅 (km ≈ 250m)

_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PARQUET    = os.path.join(_ROOT, "data", "station_elevation.parquet")
IN_ELEV_NPZ   = os.path.join(_ROOT, "data", "japan_elevation_data.npz")
OUT_PARQUET   = IN_PARQUET  # 上書き


def latlon_to_grid(lats, lons):
    """緯度経度配列 → グリッド行列インデックス (img_row, img_col)。"""
    col = np.floor((lons - MIN_LON) / DX).astype(int)
    row = np.floor((lats - MIN_LAT) / DY).astype(int)
    img_row = GRID_HEIGHT - 1 - row
    img_row = np.clip(img_row, 0, GRID_HEIGHT - 1)
    img_col = np.clip(col,     0, GRID_WIDTH  - 1)
    return img_row, img_col


def calc_svf(df: pd.DataFrame, elev_grid: np.ndarray) -> np.ndarray:
    """
    全駅について天空率 SVF を一括計算する。

    処理フロー:
      1. 各駅の標高を取得
      2. N_DIRS 方向 × N_STEPS ステップのレイキャスティング
      3. 各方向の最大地平線仰角を追跡
      4. SVF = mean(cos²(max_horizon_angle)) を返す
    """
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    N    = len(df)

    # 駅自身の標高
    img_row, img_col = latlon_to_grid(lats, lons)
    station_elev = elev_grid[img_row, img_col].astype(float)

    azimuths = np.linspace(0, 2 * np.pi, N_DIRS, endpoint=False)
    n_steps  = int(MAX_DIST_KM / STEP_KM)

    # max_horizon[station, dir] = 各方向の最大地平線仰角 (radians)
    max_horizon = np.zeros((N, N_DIRS), dtype=np.float32)

    # 経度方向の1km当たり度数（各駅の緯度で補正）
    # cos(lat) の配列 shape: (N,)
    cos_lat = np.cos(np.radians(lats))

    print(f"  レイキャスティング: {N_DIRS}方向 × {n_steps}ステップ = {N_DIRS * n_steps} 反復")
    t0 = time.time()

    for di, az in enumerate(azimuths):
        cos_az = np.cos(az)  # 北方向成分
        sin_az = np.sin(az)  # 東方向成分

        for si in range(1, n_steps + 1):
            dist_km = si * STEP_KM
            dist_m  = dist_km * 1000.0

            # 各駅のサンプル点 (degree)
            dlat = dist_km / 111.0 * cos_az
            # 経度方向は cos(lat) で1度当たりのkm数が変わる
            dlon = dist_km / 111.0 * sin_az / cos_lat

            sample_lats = lats + dlat
            sample_lons = lons + dlon

            # 範囲内チェック
            in_range = (
                (sample_lats >= MIN_LAT) & (sample_lats <= MAX_LAT) &
                (sample_lons >= MIN_LON) & (sample_lons <= MAX_LON)
            )

            s_img_row, s_img_col = latlon_to_grid(sample_lats, sample_lons)
            sample_elev = elev_grid[s_img_row, s_img_col].astype(float)

            # 地平線仰角 (atan2 で符号付き、遮蔽は正値のみ)
            elev_diff = sample_elev - station_elev
            horizon_angle = np.arctan2(elev_diff, dist_m)  # radians

            # 範囲外 or 地表より低い箇所は無視
            horizon_angle = np.where(in_range & (horizon_angle > 0), horizon_angle, 0.0)

            # 各方向の最大仰角を更新
            np.maximum(max_horizon[:, di], horizon_angle, out=max_horizon[:, di])

        if (di + 1) % 4 == 0:
            elapsed = time.time() - t0
            print(f"  方向 {di+1:2d}/{N_DIRS} 完了 ({elapsed:.1f}s)")

    # SVF = (1/N_DIRS) × Σ cos²(最大仰角)
    svf = np.mean(np.cos(max_horizon.astype(np.float64)) ** 2, axis=1)
    svf = np.clip(svf, 0.0, 1.0)
    print(f"  SVF計算完了 (総時間 {time.time()-t0:.1f}s)")
    return svf.astype(np.float32)


def main():
    print("=" * 55)
    print("駅天空率（Sky View Factor）算出スクリプト")
    print("=" * 55)
    t_start = time.time()

    # ── データ読み込み ──────────────────────────────────────────────────────
    print(f"[1/3] Parquet 読み込み: {IN_PARQUET}")
    df = pd.read_parquet(IN_PARQUET)
    print(f"      駅数: {len(df)}")

    print(f"[2/3] 標高グリッド読み込み: {IN_ELEV_NPZ}")
    t0 = time.time()
    elev_grid = np.load(IN_ELEV_NPZ)["elev"]
    print(f"      完了 ({time.time()-t0:.1f}s)  shape: {elev_grid.shape}")

    # 海域駅（elev_avg_m が NaN）は SVF=NaN とする
    land_mask = ~df["is_ocean"].to_numpy()
    df_land   = df[land_mask].reset_index(drop=True)
    print(f"      陸地駅: {land_mask.sum()} / {len(df)}")

    # ── SVF 計算 ───────────────────────────────────────────────────────────
    print("[3/3] 天空率を計算中...")
    svf_land = calc_svf(df_land, elev_grid)

    # 全駅配列に埋め込み（海域は NaN）
    svf_all = np.full(len(df), np.nan, dtype=np.float32)
    svf_all[land_mask] = svf_land

    df["svf"] = np.where(np.isfinite(svf_all), svf_all.round(3), np.nan)

    # ── 保存 ──────────────────────────────────────────────────────────────
    df.to_parquet(OUT_PARQUET, index=False)
    size_kb = os.path.getsize(OUT_PARQUET) / 1024
    print(f"\n保存完了: {OUT_PARQUET}  ({size_kb:.0f} KB)")
    print(f"総処理時間: {time.time()-t_start:.1f}s")

    # サマリー
    sv = df["svf"].dropna()
    print(f"\n── SVF サマリー（陸地駅）─────────────────")
    print(sv.describe().round(3).to_string())
    print(f"\n天空率トップ5（最も開けた空）:")
    top = df.nlargest(5, "svf")[["station_name", "line_name", "svf"]]
    print(top.to_string(index=False))
    print(f"\n天空率ボトム5（最も閉じた空）:")
    bot = df.nsmallest(5, "svf")[["station_name", "line_name", "svf"]]
    print(bot.to_string(index=False))


if __name__ == "__main__":
    main()
