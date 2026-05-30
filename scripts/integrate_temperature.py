import os
import sys
import glob
import time
import struct
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# 日本全土のグリッドパラメータの定義 (標高データと完全一致)
# 緯度: 20.0度 〜 46.0度 (1.5 * 26 = 39個の1次メッシュ) -> 39 * 320 = 12,480 ピクセル
# 経度: 122.0度 〜 155.0度 (33個の1次メッシュ) -> 33 * 320 = 10,560 ピクセル
MIN_LAT = 20.0
MAX_LAT = 46.0
MIN_LON = 122.0
MAX_LON = 155.0

LAT_MESH_COUNT = 39
LON_MESH_COUNT = 33

CELLS_PER_MESH = 320  # 1次メッシュ内の5次メッシュの分割数 (縦横)

GRID_HEIGHT = LAT_MESH_COUNT * CELLS_PER_MESH  # 12480
GRID_WIDTH = LON_MESH_COUNT * CELLS_PER_MESH   # 10560

# 5次メッシュの1画素あたりの緯度経度幅 (度)
DY = 7.5 / 3600      # 7.5秒
DX = 11.25 / 3600    # 11.25秒

# デフォルトパス設定
INPUT_DIR = r"K:\scripts\geog\data\Temperature"
OUTPUT_DIR = r"K:\scripts\geog\data\temperature_integrated"


def decode_3rd_mesh(mesh_code_str):
    """
    8桁の3次メッシュコードをデコードし、
    5次メッシュ用グローバルグリッド (12480x10560) における
    対応する4x4のセル領域 (row_start, row_end, col_start, col_end) を返す。
    """
    c12 = int(mesh_code_str[0:2])
    c34 = int(mesh_code_str[2:4])
    
    y2 = int(mesh_code_str[4])
    x2 = int(mesh_code_str[5])
    
    y3 = int(mesh_code_str[6])
    x3 = int(mesh_code_str[7])
    
    # 1次メッシュ内の3次メッシュインデックス (0〜79)
    y_3rd = y2 * 10 + y3
    x_3rd = x2 * 10 + x3
    
    # 5次メッシュ解像度 (320x320) にスケールアップするため4倍する (1次メッシュ内 0〜319)
    y_5th_start = y_3rd * 4
    x_5th_start = x_3rd * 4
    
    # 1次メッシュのグローバル位置
    lat_1st_idx = c12 - 30
    lon_1st_idx = c34 - 22
    
    # グローバルなグリッドインデックス (南西原点)
    G_row_start = lat_1st_idx * CELLS_PER_MESH + y_5th_start
    G_col_start = lon_1st_idx * CELLS_PER_MESH + x_5th_start
    
    # 画像インデックス (北西原点、上方向が北)
    # 4x4ブロックに対応するため、行は反転して境界を設定
    img_row_end = GRID_HEIGHT - 1 - G_row_start
    img_row_start = GRID_HEIGHT - 1 - (G_row_start + 3)
    
    img_col_start = G_col_start
    img_col_end = G_col_start + 3
    
    return img_row_start, img_row_end, img_col_start, img_col_end


def process_dbf(dbf_path, grid, processed_counts, attr_name='G02_036'):
    """
    1つのDBFファイルを直接バイナリ読み込みし、グローバルグリッドにプロットする (3次メッシュ対応版)。
    """
    count = 0
    skipped = 0
    
    try:
        with open(dbf_path, 'rb') as f:
            # dBaseヘッダーの読み込み
            header = f.read(32)
            version, y, m, d, num_records, header_len, record_len = struct.unpack('<BBBBLHH20x', header)
            
            # フィールド定義の読み込み
            fields = []
            num_fields = (header_len - 33) // 32
            for _ in range(num_fields):
                field_desc = f.read(32)
                name, ftype, flen, fdec = struct.unpack('<11sc4xBB14x', field_desc)
                name = name.decode('ascii').strip('\x00').strip()
                fields.append((name, ftype.decode('ascii'), flen, fdec))
                
            # ヘッダー末尾へ移動
            f.seek(header_len)
            
            # 必要なフィールド (G02_001 と attr_name) の位置を決定
            field_offsets = {}
            offset = 1  # 削除フラグの1バイトをスキップ
            for name, ftype, flen, fdec in fields:
                field_offsets[name] = (offset, flen)
                offset += flen
                
            if 'G02_001' not in field_offsets or attr_name not in field_offsets:
                print(f"警告: {dbf_path} に必要な属性が見つかりません。")
                return 0
                
            idx_code_off, idx_code_len = field_offsets['G02_001']
            idx_val_off, idx_val_len   = field_offsets[attr_name]
            
            # レコードの読み込みとプロット
            for _ in range(num_records):
                record_data = f.read(record_len)
                if len(record_data) < record_len:
                    break
                
                # 削除フラグチェック
                if record_data[0] == 0x2a:  # '*'
                    skipped += 1
                    continue
                    
                mesh_code = record_data[idx_code_off : idx_code_off + idx_code_len].decode('ascii').strip()
                val_str = record_data[idx_val_off : idx_val_off + idx_val_len].decode('ascii').strip()
                
                if not mesh_code or not val_str or val_str == 'unknown':
                    skipped += 1
                    continue
                    
                try:
                    val_raw = float(val_str)
                    # 欠損値 "999999" のチェック
                    if val_raw >= 99999.0:
                        skipped += 1
                        continue
                        
                    # 0.1℃ 単位から ℃ にデコード
                    temp_val = val_raw / 10.0
                    
                    row_s, row_e, col_s, col_e = decode_3rd_mesh(mesh_code)
                    
                    if (0 <= row_s < GRID_HEIGHT and 0 <= row_e < GRID_HEIGHT and
                        0 <= col_s < GRID_WIDTH and 0 <= col_e < GRID_WIDTH):
                        # 4x4ピクセルブロックに同一値をプロット
                        grid[row_s : row_e + 1, col_s : col_e + 1] = temp_val
                        count += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
    except Exception as e:
        print(f"エラー ({dbf_path} の読み込み失敗): {e}")
        
    processed_counts['total_records'] += count
    processed_counts['skipped_records'] += skipped
    return count


def save_visualization(grid, output_path):
    """
    気温カラーマップを適用した、見栄えの良いRGB PNG画像を生成・保存する。
    """
    print("可視化ヒートマップ画像の生成中...")
    # NaNをマスクした配列を作成
    masked_grid = np.ma.masked_invalid(grid)
    
    # 気温分布を直感的に表すカラーマップの適用
    # 'inferno' または 'plasma' は低温が暗く、高温が極めて明るく表示され、ヒートマップとして非常に美しいです
    cmap = plt.colormaps['inferno']
    
    # 8月の最高気温の表示範囲を 15℃〜40℃ に正規化
    norm = plt.Normalize(vmin=15.0, vmax=38.0)
    
    # カラーマップを適用してRGBAを取得
    rgba = cmap(norm(masked_grid))
    
    # データがない海域（NaN）を深みのある青にする
    blue_bg = np.array([0.05, 0.1, 0.25, 1.0])
    rgba[masked_grid.mask] = blue_bg
    
    # 8-bitのRGB形式に変換
    rgb_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 画像の保存
    img = Image.fromarray(rgb_8bit)
    img.save(output_path, "PNG")
    print(f"可視化ヒートマップ画像を保存しました: {output_path}")


def save_geotiff(grid, output_path):
    """
    rasterioを用いて位置情報付きの float32 GeoTIFF を出力する。
    """
    print("GeoTIFFデータの出力中...")
    # 左上角 (西端 122.0, 北端 46.0) からアフィントランスフォームを作成
    transform = from_origin(MIN_LON, MAX_LAT, DX, DY)
    
    # CRS: JGD2000 地理座標系 (EPSG:4612)
    crs = rasterio.crs.CRS.from_epsg(4612)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': GRID_WIDTH,
        'height': GRID_HEIGHT,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(grid, 1)
        
    print(f"GeoTIFFデータを保存しました: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 統合用の巨大NumPyグリッドを確保 (NaNで初期化)
    print(f"日本全土のグリッド配列を初期化中... サイズ: {GRID_HEIGHT} x {GRID_WIDTH}")
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan, dtype=np.float32)
    
    # 2022年版（G02-22_*_GML）のDBFファイルを探す
    search_pattern = os.path.join(INPUT_DIR, "G02-22_*-jgd_GML", "G02-22_*-jgd.dbf")
    dbf_files = sorted(glob.glob(search_pattern))
    
    if not dbf_files:
        print(f"エラー: {INPUT_DIR} に2022年版のDBFファイルが見つかりませんでした。パターン: {search_pattern}")
        sys.exit(1)
        
    print(f"発見された2022年版DBFファイルの数: {len(dbf_files)}")
    
    # 一括処理
    print(f"\n--- 【8月最高気温メッシュ平年値の統合処理】 {len(dbf_files)} ファイルを処理します ---")
    processed_counts = {'total_records': 0, 'skipped_records': 0}
    start_time = time.time()
    
    for idx, dbf_path in enumerate(dbf_files, 1):
        file_start = time.time()
        file_name = os.path.basename(dbf_path)
        mesh_id = file_name.split('_')[1].split('-')[0]
        
        count = process_dbf(dbf_path, grid, processed_counts, attr_name='G02_036')
        file_elapsed = time.time() - file_start
        
        if idx == 1 or idx % 10 == 0 or idx == len(dbf_files):
            print(f"[{idx}/{len(dbf_files)}] メッシュ {mesh_id}: {count} レコードプロット (このファイル: {file_elapsed:.2f}秒, 累計プロット: {processed_counts['total_records']})")
            
    total_elapsed = time.time() - start_time
    print(f"\n全ファイルの読み込み＆プロットが完了しました！")
    print(f"  総プロットレコード数: {processed_counts['total_records']}")
    print(f"  スキップレコード数: {processed_counts['skipped_records']}")
    print(f"  データパース所要時間: {total_elapsed:.1f}秒")
    
    # 成果物の保存
    vis_path = os.path.join(OUTPUT_DIR, "japan_aug_max_temp_map.png")
    save_visualization(grid, vis_path)
    
    tif_path = os.path.join(OUTPUT_DIR, "japan_aug_max_temp_data.tif")
    save_geotiff(grid, tif_path)
    
    npz_path = os.path.join(OUTPUT_DIR, "japan_aug_max_temp_data.npz")
    print("NumPy圧縮バイナリの保存中...")
    np.savez_compressed(npz_path, temp=grid)
    print(f"NumPy圧縮バイナリを保存しました: {npz_path}")
    
    print("\nすべてのデータ統合＆出力処理が完全に完了しました！🎉")
    
    # 簡易統計情報の出力
    temp_valid = grid[~np.isnan(grid)]
    print("\n── 気温データサマリー ────────────────")
    print(f"  データ総ピクセル数: {len(temp_valid)} (5次メッシュ換算)")
    print(f"  全国平均値: {np.mean(temp_valid):.2f} ℃")
    print(f"  全国最低値: {np.min(temp_valid):.2f} ℃")
    print(f"  全国最高値: {np.max(temp_valid):.2f} ℃")


if __name__ == "__main__":
    main()
