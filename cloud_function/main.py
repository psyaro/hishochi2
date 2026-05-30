"""
標高調査API — Google Cloud Functions (第2世代 / Functions Framework) 用エントリポイント。

分離構成:
  - フロントエンド(静的HTML)は GitHub Pages で配信する (../docs/index.html)。
  - この Cloud Function は JSON API (/api/elevation) のみを提供する。

データは japan_elevation_data.npz / japan_elevation_data_min.npz を
このディレクトリに同梱してデプロイする (デプロイ手順は README_DEPLOY.md を参照)。
コールドスタート時に一度だけメモリへロードし、以降は高速応答する。

標高データ出典: 国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」(国土交通省)
  https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html
を加工して作成。
"""
import os
import json
import numpy as np
import functions_framework

# グリッドパラメータの定義 (integrate_elev.py / server.py と一致)
MIN_LAT = 20.0
MAX_LAT = 46.0
MIN_LON = 122.0
MAX_LON = 155.0

GRID_HEIGHT = 12480
GRID_WIDTH = 10560

DY = 7.5 / 3600
DX = 11.25 / 3600

# データ出典 (レスポンスにも明記する)
DATA_SOURCE = (
    "国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）"
    "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html を加工して作成"
)

# 関数と同一ディレクトリにデータを同梱する想定
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_AVG_PATH = os.path.join(_BASE_DIR, "japan_elevation_data.npz")
_MIN_PATH = os.path.join(_BASE_DIR, "japan_elevation_data_min.npz")

# コールドスタート時に一度だけロードするためのキャッシュ
_elev_avg = None
_elev_min = None


def _ensure_loaded():
    """標高データを遅延ロードしてグローバルにキャッシュする。"""
    global _elev_avg, _elev_min
    if _elev_avg is None:
        _elev_avg = np.load(_AVG_PATH)["elev"]
    if _elev_min is None:
        _elev_min = np.load(_MIN_PATH)["elev"]


def _get_elevation_data(lat, lon):
    """指定された緯度経度から平均標高と最低標高を取得する。"""
    if not (MIN_LAT <= lat <= MAX_LAT) or not (MIN_LON <= lon <= MAX_LON):
        return None, ("範囲外の座標です (日本領土の緯度 20°〜46°、"
                      "経度 122°〜155° の範囲で指定してください)")

    _ensure_loaded()

    # 緯度経度からグローバルグリッドインデックス (南西原点) へのマッピング
    col = int((lon - MIN_LON) / DX)
    row = int((lat - MIN_LAT) / DY)

    # 画像インデックス (北西原点、上方向が北) への変換
    img_row = GRID_HEIGHT - 1 - row
    img_col = col

    if not (0 <= img_row < GRID_HEIGHT) or not (0 <= img_col < GRID_WIDTH):
        return None, "グリッド範囲外のインデックスです"

    val_avg = _elev_avg[img_row, img_col]
    val_min = _elev_min[img_row, img_col]

    val_avg_res = None if np.isnan(val_avg) else float(round(val_avg, 1))
    val_min_res = None if np.isnan(val_min) else float(round(val_min, 1))

    return {
        "latitude": lat,
        "longitude": lon,
        "average_elevation_m": val_avg_res,
        "minimum_elevation_m": val_min_res,
        "is_ocean": val_avg_res is None,
    }, None


def _json_response(status_code, payload):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        # GitHub Pages 等のどのオリジンからでも呼べるよう CORS を許可
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    body = json.dumps(payload, ensure_ascii=False)
    return (body, status_code, headers)


@functions_framework.http
def elevation(request):
    """HTTP エントリポイント。?lat=..&lon=.. で標高を返す。"""
    # CORS プリフライト
    if request.method == "OPTIONS":
        return _json_response(204, {})

    lat_str = request.args.get("lat")
    lon_str = request.args.get("lon")

    if not lat_str or not lon_str:
        return _json_response(400, {
            "status": "error",
            "message": ("引数 lat (緯度) と lon (経度) が必要です。"
                        "例: ?lat=35.3606&lon=138.7273"),
        })

    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return _json_response(400, {
            "status": "error",
            "message": "経緯度は有効な数値で指定してください。",
        })

    data, error = _get_elevation_data(lat, lon)
    if error:
        return _json_response(400, {"status": "error", "message": error})

    return _json_response(200, {
        "status": "success",
        "data": data,
        "source": DATA_SOURCE,
    })
