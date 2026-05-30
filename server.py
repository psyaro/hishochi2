import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ポート番号の設定
PORT = 8080

# グリッドパラメータの定義 (integrate_elev.py と一致)
MIN_LAT = 20.0
MAX_LAT = 46.0
MIN_LON = 122.0
MAX_LON = 155.0

GRID_HEIGHT = 12480
GRID_WIDTH = 10560

DY = 7.5 / 3600
DX = 11.25 / 3600

# データのロード用グローバル変数
elev_avg = None
elev_min = None

def load_data():
    """
    平均標高および最低標高のNumPyデータをメモリにロードする
    """
    global elev_avg, elev_min
    avg_path = "data/japan_elevation_data.npz"
    min_path = "data/japan_elevation_data_min.npz"
    
    print("--------------------------------------------------")
    print("標高調査データファイルをメモリにロードしています...")
    
    # 平均データのロード
    if os.path.exists(avg_path):
        t0 = np.load(avg_path)
        elev_avg = t0['elev']
        print(f"  [成功] 平均標高データをロードしました。形状: {elev_avg.shape}")
    else:
        print(f"  [エラー] 平均標高データ {avg_path} が見つかりません。")
        sys.exit(1)
        
    # 最低標高データのロード
    if os.path.exists(min_path):
        t1 = np.load(min_path)
        elev_min = t1['elev']
        print(f"  [成功] 最低標高データをロードしました。形状: {elev_min.shape}")
    else:
        print(f"  [エラー] 最低標高データ {min_path} が見つかりません。")
        sys.exit(1)
        
    print("全標高データのロードが完了し、極限高速応答モードになりました！🚀")
    print("--------------------------------------------------")

def get_elevation_data(lat, lon):
    """
    指定された緯度経度から平均標高と最低標高を取得する
    """
    if not (MIN_LAT <= lat <= MAX_LAT) or not (MIN_LON <= lon <= MAX_LON):
        return None, "範囲外の座標です (日本領土の緯度 20°〜46°、経度 122°〜155° の範囲で指定してください)"
        
    # 緯度経度からグローバルグリッドインデックス (南西原点) へのマッピング
    col = int((lon - MIN_LON) / DX)
    row = int((lat - MIN_LAT) / DY)
    
    # 画像インデックス (北西原点、上方向が北) への変換
    img_row = GRID_HEIGHT - 1 - row
    img_col = col
    
    if not (0 <= img_row < GRID_HEIGHT) or not (0 <= img_col < GRID_WIDTH):
        return None, "グリッド範囲外のインデックスです"
        
    # 配列から標高を抽出
    val_avg = elev_avg[img_row, img_col]
    val_min = elev_min[img_row, img_col]
    
    # NaNチェック (海域や未計測部分)
    if np.isnan(val_avg):
        val_avg_res = None
    else:
        val_avg_res = float(round(val_avg, 1))
        
    if np.isnan(val_min):
        val_min_res = None
    else:
        val_min_res = float(round(val_min, 1))
        
    return {
        "latitude": lat,
        "longitude": lon,
        "average_elevation_m": val_avg_res,
        "minimum_elevation_m": val_min_res,
        "is_ocean": val_avg_res is None
    }, None

class ElevationAPIRequestHandler(BaseHTTPRequestHandler):
    """
    APIリクエストおよびWeb UI画面のリクエストを処理するハンドラー
    """
    def log_message(self, format, *args):
        # コンソールログをスッキリさせるため、標準ログ出力を無効化（またはカスタム出力）
        pass
        
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # 1. APIエンドポイントの処理
        if path == "/api/elevation":
            query = urllib.parse.parse_qs(parsed_url.query)
            lat_str = query.get("lat", [None])[0]
            lon_str = query.get("lon", [None])[0]
            
            if not lat_str or not lon_str:
                self.send_json_response(400, {
                    "status": "error",
                    "message": "引数 lat (緯度) と lon (経度) が必要です。例: /api/elevation?lat=35.3606&lon=138.7273"
                })
                return
                
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                self.send_json_response(400, {
                    "status": "error",
                    "message": "経緯度は有効な数値で指定してください。"
                })
                return
                
            data, error = get_elevation_data(lat, lon)
            if error:
                self.send_json_response(400, {
                    "status": "error",
                    "message": error
                })
            else:
                self.send_json_response(200, {
                    "status": "success",
                    "data": data
                })
            return
            
        # 2. Web UI HTML 画面の提供
        elif path == "/" or path == "/index.html":
            self.send_html_response()
            return
            
        # 404 エラー
        else:
            self.send_json_response(404, {
                "status": "error",
                "message": "見つかりません。Web UI画面は / に、標高調査APIは /api/elevation にアクセスしてください。"
            })
            
    def send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*") # CORSを有効にし、どこからでもAPIを叩けるように設定
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        
    def send_html_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_CONTENT.encode('utf-8'))

# 美しく洗練されたWeb UI画面のHTML/CSS/JS定義
HTML_CONTENT = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>日本全国250mメッシュ標高調査システム</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+JP:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.7);
            --border-color: rgba(240, 246, 252, 0.1);
            --primary-glow: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --accent-glow: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', 'Noto Sans JP', sans-serif;
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(79, 172, 254, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 650px;
            z-index: 10;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(120deg, #00f2fe, #4facfe, #00f2fe);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
            animation: shine 4s linear infinite;
        }

        header p {
            font-size: 0.95rem;
            color: var(--text-sub);
            font-weight: 300;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 35px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            margin-bottom: 25px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-group label {
            font-size: 0.85rem;
            color: var(--text-sub);
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .input-group input {
            background: rgba(13, 17, 23, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 1.1rem;
            color: var(--text-main);
            font-family: inherit;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .input-group input:focus {
            outline: none;
            border-color: #4facfe;
            box-shadow: 0 0 15px rgba(79, 172, 254, 0.2);
            background: rgba(13, 17, 23, 0.8);
        }

        .btn-submit {
            width: 100%;
            background: var(--primary-glow);
            border: none;
            border-radius: 14px;
            padding: 16px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(0, 242, 254, 0.25);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            letter-spacing: 1px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(0, 242, 254, 0.35);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* 結果表示エリア */
        .result-container {
            display: none;
            margin-top: 30px;
            padding-top: 25px;
            border-top: 1px solid var(--border-color);
            animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-title {
            font-size: 0.9rem;
            color: var(--text-sub);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .result-title::before {
            content: '';
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #00f2fe;
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .result-box {
            background: rgba(13, 17, 23, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }

        .result-box::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }

        .result-box.average::after {
            background: var(--primary-glow);
        }

        .result-box.minimum::after {
            background: var(--accent-glow);
        }

        .result-label {
            font-size: 0.8rem;
            color: var(--text-sub);
            margin-bottom: 8px;
            font-weight: 400;
        }

        .result-val {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .result-unit {
            font-size: 1rem;
            font-weight: 400;
            color: var(--text-sub);
            margin-left: 4px;
        }

        .ocean-tag {
            grid-column: span 2;
            background: rgba(79, 172, 254, 0.1);
            border: 1px dashed rgba(79, 172, 254, 0.3);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            font-size: 0.9rem;
            color: #00f2fe;
        }

        /* 開発者向けAPIパネル */
        .api-panel {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 25px;
            font-family: 'Outfit', sans-serif;
        }

        .api-panel h2 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: -0.2px;
            color: var(--text-main);
        }

        .api-panel p {
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-bottom: 15px;
        }

        .code-block {
            background: #0d1117;
            border-radius: 10px;
            padding: 12px 16px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.8rem;
            color: #00f2fe;
            overflow-x: auto;
            border: 1px solid var(--border-color);
            white-space: nowrap;
        }

        /* アニメーション */
        @keyframes shine {
            to { background-position: 200% auto; }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* データ出典クレジット */
        .credit {
            text-align: center;
            margin-top: 25px;
            font-size: 0.78rem;
            line-height: 1.7;
            color: var(--text-sub);
        }
        .credit a {
            color: #4facfe;
            text-decoration: none;
        }
        .credit a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GEO-ELEVATION PORTAL</h1>
            <p>日本全土 250mメッシュ高精度標高調査システム (API搭載)</p>
        </header>

        <main>
            <!-- 検索入力フォームカード -->
            <div class="card">
                <form id="searchForm" onsubmit="searchElevation(event)">
                    <div class="form-grid">
                        <div class="input-group">
                            <label for="lat">緯度 (Latitude)</label>
                            <input type="number" id="lat" name="lat" step="any" min="20.0" max="46.0" placeholder="例: 35.3606" required>
                        </div>
                        <div class="input-group">
                            <label for="lon">経度 (Longitude)</label>
                            <input type="number" id="lon" name="lon" step="any" min="122.0" max="155.0" placeholder="例: 138.7273" required>
                        </div>
                    </div>
                    <button type="submit" class="btn-submit">調査開始 (Query Data)</button>
                </form>

                <!-- 結果エリア -->
                <div class="result-container" id="resultContainer">
                    <div class="result-title">調査結果 (Search Results)</div>
                    <div class="result-grid" id="resultGrid">
                        <!-- 動的に挿入 -->
                    </div>
                </div>
            </div>

            <!-- API開発者向けパネル -->
            <div class="api-panel">
                <h2>標高調査API (RESTful API)</h2>
                <p>プログラムや外部システムから、この高精度データに直接HTTPリクエストを送信してアクセスできます。</p>
                <div class="code-block" id="apiExample">
                    curl "http://localhost:8080/api/elevation?lat=35.3606&lon=138.7273"
                </div>
            </div>

            <!-- データ出典クレジット -->
            <footer class="credit">
                標高データ出典:
                <a href="https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html" target="_blank" rel="noopener">
                    国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）
                </a>
                を加工して作成。
            </footer>
        </main>
    </div>

    <script>
        // サンプルのプレースホルダー値を設定 (富士山周辺)
        document.getElementById('lat').value = "35.3606";
        document.getElementById('lon').value = "138.7273";

        function searchElevation(event) {
            event.preventDefault();
            const lat = document.getElementById('lat').value;
            const lon = document.getElementById('lon').value;
            const btn = document.querySelector('.btn-submit');
            
            btn.textContent = 'データ解析中...';
            btn.disabled = true;

            // APIの呼び出し
            const url = `/api/elevation?lat=${lat}&lon=${lon}`;
            
            // 開発者用APIサンプルコマンドの更新
            document.getElementById('apiExample').textContent = `curl "http://${window.location.host}/api/elevation?lat=${lat}&lon=${lon}"`;

            fetch(url)
                .then(response => response.json())
                .then(res => {
                    btn.textContent = '調査開始 (Query Data)';
                    btn.disabled = false;

                    const container = document.getElementById('resultContainer');
                    const grid = document.getElementById('resultGrid');
                    
                    container.style.display = 'block';
                    grid.innerHTML = '';

                    if (res.status === 'success') {
                        const data = res.data;
                        
                        if (data.is_ocean) {
                            grid.innerHTML = `
                                <div class="ocean-tag">
                                    🌊 指定された座標は「海域」または「未計測領域」です (標高データがありません)。
                                </div>
                            `;
                        } else {
                            // 平均標高表示
                            const avgVal = data.average_elevation_m !== null ? data.average_elevation_m : 'N/A';
                            // 最低標高表示
                            const minVal = data.minimum_elevation_m !== null ? data.minimum_elevation_m : 'N/A';

                            grid.innerHTML = `
                                <div class="result-box average">
                                    <div class="result-label">平均標高 (Average Elevation)</div>
                                    <div class="result-val" id="avgCounter">${avgVal}<span class="result-unit">m</span></div>
                                </div>
                                <div class="result-box minimum">
                                    <div class="result-label">最低標高 (Minimum Elevation)</div>
                                    <div class="result-val" id="minCounter">${minVal}<span class="result-unit">m</span></div>
                                </div>
                            `;

                            // 簡単なカウンタアニメーション
                            animateValue("avgCounter", 0, data.average_elevation_m, 600);
                            animateValue("minCounter", 0, data.minimum_elevation_m, 600);
                        }
                    } else {
                        grid.innerHTML = `
                            <div class="ocean-tag" style="background: rgba(255, 8, 68, 0.1); border-color: rgba(255, 8, 68, 0.3); color: #ffb199;">
                                ⚠️ エラー: ${res.message}
                            </div>
                        `;
                    }
                })
                .catch(err => {
                    btn.textContent = '調査開始 (Query Data)';
                    btn.disabled = false;
                    alert('通信エラーが発生しました。サーバーが稼働しているか確認してください。');
                });
        }

        // 数値上昇アニメーション
        function animateValue(id, start, end, duration) {
            if (end === null || isNaN(end)) return;
            const obj = document.getElementById(id);
            const range = end - start;
            let current = start;
            const increment = range / (duration / 16);
            const step = () => {
                current += increment;
                if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                    obj.innerHTML = `${end.toFixed(1)}<span class="result-unit">m</span>`;
                } else {
                    obj.innerHTML = `${current.toFixed(1)}<span class="result-unit">m</span>`;
                    requestAnimationFrame(step);
                }
            };
            requestAnimationFrame(step);
        }
    </script>
</body>
</html>
"""

def main():
    # 起動前にデータをメモリにロード
    load_data()
    
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, ElevationAPIRequestHandler)
    
    print("==================================================")
    print(f"標高調査Webサーバーが正常に起動しました！🎉")
    print(f"  ブラウザで以下にアクセスしてください:")
    print(f"  👉  http://localhost:{PORT}")
    print(f"  標高調査APIエンドポイント:")
    print(f"  👉  http://localhost:{PORT}/api/elevation?lat=35.3606&lon=138.7273")
    print("==================================================")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しています...")
        httpd.server_close()
        print("サーバーが正常に停止しました。")

if __name__ == "__main__":
    main()
