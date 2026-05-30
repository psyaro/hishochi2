"""
全国の駅標高（最低標高）をボロノイ分割した GeoJSON を生成する。
出力: docs/data/voronoi.geojson  (GitHub Pages 配信用)

標高データ出典: 国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）
  https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html を加工して作成。
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import voronoi_diagram
from shapely.geometry import MultiPoint, box, mapping
from shapely.validation import make_valid

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PKL  = os.path.join(_ROOT, "data", "station_elevation.parquet")
OUT_DIR = os.path.join(_ROOT, "docs", "data")
OUT_GJ  = os.path.join(OUT_DIR, "voronoi.geojson")

# ── 日本の大まかなクリップ矩形 (本州・四国・九州・北海道をカバー) ──────────
JAPAN_CLIP = box(129.0, 30.5, 146.5, 45.6)

# ポリゴン座標の丸め精度 (4桁 ≈ 11m 精度 / 250m メッシュに対して十分)
COORD_PRECISION = 4

# shapely simplify の許容誤差 (度)
SIMPLIFY_TOL = 0.005   # ≈ 500m


def round_coords(geometry):
    """GeoJSON geometry の座標を COORD_PRECISION 桁に丸める (ファイルサイズ削減)。"""
    def rnd(c):
        return (round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION))

    def rnd_ring(ring):
        return [rnd(c) for c in ring]

    gj = mapping(geometry)
    gt = gj["type"]
    if gt == "Polygon":
        gj["coordinates"] = [rnd_ring(r) for r in gj["coordinates"]]
    elif gt == "MultiPolygon":
        gj["coordinates"] = [
            [rnd_ring(r) for r in poly] for poly in gj["coordinates"]
        ]
    return gj


def build_voronoi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """scipy ではなく shapely の voronoi_diagram を使って Voronoi 領域を生成し、
    元の駅データと空間結合して標高属性を付与する。"""

    print("[1/4] ボロノイ図を計算中...")
    t0 = time.time()

    points = MultiPoint(list(zip(gdf["lon"].values, gdf["lat"].values)))

    # envelope を少し広げて端点の無限領域を安全に処理
    lon_min, lat_min = gdf["lon"].min() - 3, gdf["lat"].min() - 3
    lon_max, lat_max = gdf["lon"].max() + 3, gdf["lat"].max() + 3
    envelope = box(lon_min, lat_min, lon_max, lat_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regions = voronoi_diagram(points, envelope=envelope, tolerance=0.0)

    print(f"     生成ポリゴン数: {len(regions.geoms)}  ({time.time()-t0:.1f}s)")
    return regions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── データ読み込み ──────────────────────────────────────────────────
    df = pd.read_parquet(IN_PKL)
    # elev_min_m が NaN の駅は除外 (ないはずだが念のため)
    df = df.dropna(subset=["elev_min_m"]).reset_index(drop=True)
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    print(f"[0/4] 駅数: {len(gdf_pts)}")

    # ── ボロノイ生成 ───────────────────────────────────────────────────
    regions = build_voronoi(gdf_pts)
    gdf_vor = gpd.GeoDataFrame(
        geometry=list(regions.geoms), crs="EPSG:4326"
    )

    # ── 駅データとの空間結合 (各ポリゴン内の駅点を紐付け) ──────────────
    print("[2/4] 空間結合中...")
    t1 = time.time()
    join_cols = ["station_code", "station_name", "line_name", "operator",
                 "lon", "lat", "elev_min_m", "elev_avg_m", "temp_max_aug", "geometry"]
    joined = gpd.sjoin(
        gdf_vor,
        gdf_pts[join_cols],
        how="left",
        predicate="contains",
    )
    # 重複行 (境界上の点) を除去: ポリゴンに最も近い点を選ぶ
    joined = joined.drop_duplicates(subset=["geometry"])
    print(f"     結合完了 ({time.time()-t1:.1f}s)  欠損: {joined['station_code'].isna().sum()}")

    # ── クリップ・簡略化 ──────────────────────────────────────────────
    print("[3/4] クリップ・簡略化中...")
    t2 = time.time()
    clipped = joined.copy()
    clipped["geometry"] = clipped.geometry.intersection(JAPAN_CLIP)
    clipped = clipped[~clipped.geometry.is_empty].copy()
    clipped["geometry"] = clipped.geometry.apply(
        lambda g: make_valid(g.simplify(SIMPLIFY_TOL, preserve_topology=True))
    )
    clipped = clipped[~clipped.geometry.is_empty].copy()
    print(f"     完了 ({time.time()-t2:.1f}s)  ポリゴン数: {len(clipped)}")

    # ── GeoJSON 書き出し (座標を丸めてファイルサイズ削減) ────────────────
    print("[4/4] GeoJSON 書き出し中...")
    t3 = time.time()

    features = []
    for row in clipped.itertuples(index=False):
        if row.geometry is None or row.geometry.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": round_coords(row.geometry),
            "properties": {
                "code": row.station_code if pd.notna(row.station_code) else None,
                "name": row.station_name if pd.notna(row.station_name) else "不明",
                "line": row.line_name   if pd.notna(row.line_name)    else "",
                "op":   row.operator    if pd.notna(row.operator)     else "",
                "min":  round(float(row.elev_min_m), 1) if pd.notna(row.elev_min_m) else None,
                "avg":  round(float(row.elev_avg_m), 1) if pd.notna(row.elev_avg_m) else None,
                "temp": round(float(row.temp_max_aug), 1) if pd.notna(row.temp_max_aug) else None,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": (
                "国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）"
                " https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html"
                " / 駅データ: 国土数値情報「鉄道データ（N02-25）」（国土交通省）を加工して作成"
            ),
            "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
    }

    with open(OUT_GJ, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_GJ) / 1024
    print(f"     保存完了: {OUT_GJ}  ({size_kb:.0f} KB)  ({time.time()-t3:.1f}s)")
    print(f"\n完了！  総処理時間: {time.time()-t_start:.1f}s")
    print(f"フィーチャー数: {len(features)}")


if __name__ == "__main__":
    main()
