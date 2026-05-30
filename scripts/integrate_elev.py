import os
import sys
import glob
import argparse
import time
import struct
import numpy as np
from PIL import Image
import fiona
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# 日本全土のグリッドパラメータの定義
# 緯度: 20.0度 〜 46.0度 (1.5 * 26 = 39個の1次メッシュ) -> 39 * 320 = 12,480 ピクセル
# 経度: 122.0度 〜 155.0度 (33個の1次メッシュ) -> 33 * 320 = 10,560 ピクセル
MIN_LAT = 20.0
MAX_LAT = 46.0
MIN_LON = 122.0
MAX_LON = 155.0

LAT_MESH_COUNT = 39  # 68 - 30 + 1
LON_MESH_COUNT = 33  # 54 - 22 + 1

CELLS_PER_MESH = 320  # 1次メッシュ内の5次メッシュの分割数 (縦横)

GRID_HEIGHT = LAT_MESH_COUNT * CELLS_PER_MESH  # 12480
GRID_WIDTH = LON_MESH_COUNT * CELLS_PER_MESH   # 10560

# 5次メッシュの1画素あたりの緯度経度幅
DY = 7.5 / 3600      # 7.5秒 (0.0020833333333333333度)
DX = 11.25 / 3600    # 11.25秒 (0.003125度)

def decode_5th_mesh(mesh_code_str):
    """
    10桁の5次メッシュコードをデコードし、グローバルグリッド上の行・列インデックスを計算する。
    """
    # 1次メッシュ (4桁)
    c12 = int(mesh_code_str[0:2])
    c34 = int(mesh_code_str[2:4])
    
    # 2次メッシュ (2桁)
    y2 = int(mesh_code_str[4])
    x2 = int(mesh_code_str[5])
    
    # 3次メッシュ (2桁)
    y3 = int(mesh_code_str[6])
    x3 = int(mesh_code_str[7])
    
    # 4次メッシュ (1桁)
    c9 = int(mesh_code_str[8])
    if c9 == 1:
        y4, x4 = 0, 0
    elif c9 == 2:
        y4, x4 = 0, 1
    elif c9 == 3:
        y4, x4 = 1, 0
    elif c9 == 4:
        y4, x4 = 1, 1
    else:
        y4, x4 = 0, 0
        
    # 5次メッシュ (1桁)
    c10 = int(mesh_code_str[9])
    if c10 == 1:
        y5, x5 = 0, 0
    elif c10 == 2:
        y5, x5 = 0, 1
    elif c10 == 3:
        y5, x5 = 1, 0
    elif c10 == 4:
        y5, x5 = 1, 1
    else:
        y5, x5 = 0, 0
        
    # 1次メッシュ内のインデックスに合成 (0〜319)
    y = y2 * 40 + y3 * 4 + y4 * 2 + y5
    x = x2 * 40 + x3 * 4 + x4 * 2 + x5
    
    # 1次メッシュのグローバル位置
    lat_1st_idx = c12 - 30
    lon_1st_idx = c34 - 22
    
    # グローバルなグリッドインデックス (南西原点)
    G_row = lat_1st_idx * CELLS_PER_MESH + y
    G_col = lon_1st_idx * CELLS_PER_MESH + x
    
    # 画像インデックス (北西原点、上方向が北)
    img_row = GRID_HEIGHT - 1 - G_row
    img_col = G_col
    
    return img_row, img_col

def process_dbf(dbf_path, grid, processed_counts, attr_name='G04d_002'):
    """
    1つのDBFファイルを直接読み込み、グローバルグリッドにプロットする (超高速版)。
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
                
            # ヘッダー末尾 of the header to seek
            f.seek(header_len)
            
            # 必要なフィールド (G04d_001 と attr_name) の位置を決定
            field_offsets = {}
            offset = 1 # 削除フラグの1バイトをスキップ
            for name, ftype, flen, fdec in fields:
                field_offsets[name] = (offset, flen)
                offset += flen
                
            if 'G04d_001' not in field_offsets or attr_name not in field_offsets:
                print(f"警告: {dbf_path} に必要な属性 {attr_name} が見つかりません。")
                return 0
                
            idx_1_off, idx_1_len = field_offsets['G04d_001']
            idx_2_off, idx_2_len = field_offsets[attr_name]
            
            # レコードの一括ロード・プロット
            for _ in range(num_records):
                record_data = f.read(record_len)
                if len(record_data) < record_len:
                    break
                
                # 削除フラグチェック
                if record_data[0] == 0x2a: # '*'
                    skipped += 1
                    continue
                    
                mesh_code = record_data[idx_1_off : idx_1_off + idx_1_len].decode('ascii').strip()
                elev_str = record_data[idx_2_off : idx_2_off + idx_2_len].decode('ascii').strip()
                
                if not mesh_code or not elev_str or elev_str == 'unknown':
                    skipped += 1
                    continue
                    
                try:
                    elev = float(elev_str)
                    row, col = decode_5th_mesh(mesh_code)
                    
                    if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
                        grid[row, col] = elev
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
    地形カラーマップを適用した、見栄えの良いRGB PNG画像を生成・保存する。
    """
    print("可視化RGB画像の生成中...")
    # NaNをマスクした配列を作成
    masked_grid = np.ma.masked_invalid(grid)
    
    # 地形カラーマップの適用 (低地: 緑 -> 中地: 黄・茶 -> 高地: 白)
    cmap = plt.colormaps['terrain']
    
    # 標高0〜3000mの範囲で正規化 (日本国内の大部分を綺麗に表現)
    norm = plt.Normalize(vmin=0, vmax=3000)
    
    # カラーマップを適用してRGBAを取得 (0〜1の浮動小数点)
    rgba = cmap(norm(masked_grid))
    
    # 標高データがない部分 (NaN = マスクされた部分) を海として青色にする
    # RGBA: 青は (0.1, 0.25, 0.5, 1.0) 程度の深みのあるダークブルー
    blue_bg = np.array([0.1, 0.25, 0.5, 1.0])
    
    # マスクがTrueの部分に背景色を代入
    rgba[masked_grid.mask] = blue_bg
    
    # 8-bitのRGB形式に変換
    rgb_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 画像の保存
    img = Image.fromarray(rgb_8bit)
    img.save(output_path, "PNG")
    print(f"可視化RGB画像を保存しました: {output_path}")

def save_geotiff(grid, output_path):
    """
    rasterioを用いて、位置情報付きの本格的なfloat32 GeoTIFFを出力する。
    """
    print("GeoTIFFデータの出力中...")
    # 左上角の緯度経度 (西端 122.0, 北端 46.0) からアフィントランスフォームを作成
    # rasterio では y解像度は負にする (北から南へ下がるため)
    transform = from_origin(MIN_LON, MAX_LAT, DX, DY)
    
    # CRS: JGD2000 地理座標系 (EPSG:4612)
    crs = rasterio.crs.CRS.from_epsg(4612)
    
    # 書き込み設定
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': GRID_WIDTH,
        'height': GRID_HEIGHT,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'  # LZW圧縮を適用してファイルサイズを劇的に軽量化
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(grid, 1)
        
    print(f"GeoTIFFデータを保存しました: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="国土数値情報 標高5次メッシュデータを軽量な画像・データファイルに統合します")
    parser.add_argument("--test", action="store_true", help="テスト実行モード (特定の1ファイルのみを処理し、切り出し画像を保存)")
    parser.add_argument("--input-dir", default=r"K:\scripts\geog\data\elev250m", help="入力データの親ディレクトリ")
    parser.add_argument("--output-dir", default=r"K:\scripts\geog\data\elev250m_integrated", help="出力先ディレクトリ")
    parser.add_argument("--attr", default="G04d_002", help="読み込む属性フィールド (平均: G04d_002, 最低: G04d_004)")
    parser.add_argument("--suffix", default="", help="出力ファイル名の接尾辞 (例: _min)")
    args = parser.parse_args()
    
    # ディレクトリ準備
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 統合用の巨大NumPyグリッドを確保 (NaNで初期化)
    print(f"日本全土のグリッド配列を初期化中... サイズ: {GRID_HEIGHT} x {GRID_WIDTH} ({GRID_HEIGHT*GRID_WIDTH/1e6:.1f} Mpx)")
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan, dtype=np.float32)
    
    # 対象のDBFファイルを探す
    search_pattern = os.path.join(args.input_dir, "G04-d-11_*_GML", "*_ElevationAndSlopeAngleFifthMesh.dbf")
    dbf_files = glob.glob(search_pattern)
    
    if not dbf_files:
        print(f"エラー: {args.input_dir} にDBFファイルが見つかりませんでした。パターン: {search_pattern}")
        sys.exit(1)
        
    print(f"発見されたDBFファイルの数: {len(dbf_files)}")
    
    if args.test:
        # テストモード：東京・三浦半島・房総半島をカバーする '5339' を優先して1つだけ処理
        test_file = None
        for f in dbf_files:
            if "5339" in f:
                test_file = f
                break
        if not test_file:
            test_file = dbf_files[0]
            
        print(f"\n--- 【テストモード】以下のファイルのみを処理します ---")
        print(f"対象ファイル: {test_file}")
        
        processed_counts = {'total_records': 0, 'skipped_records': 0}
        start_time = time.time()
        process_dbf(test_file, grid, processed_counts, args.attr)
        elapsed = time.time() - start_time
        
        print(f"処理完了: {processed_counts['total_records']} レコードプロット済 (所要時間: {elapsed:.2f}秒)")
        
        # テスト用の切り出し画像の保存 (5339メッシュ周辺のみ)
        # 1次メッシュ 5339 は lat_idx = 53-30 = 23, lon_idx = 39-22 = 17
        lat_idx = 23
        lon_idx = 17
        
        r_start = GRID_HEIGHT - (lat_idx + 1) * CELLS_PER_MESH
        r_end = GRID_HEIGHT - lat_idx * CELLS_PER_MESH
        c_start = lon_idx * CELLS_PER_MESH
        c_end = (lon_idx + 1) * CELLS_PER_MESH
        
        # 該当部分をスライス抽出 (320x320)
        sub_grid = grid[r_start:r_end, c_start:c_end]
        
        # テスト用の切り出し画像を保存
        test_vis_path = os.path.join(args.output_dir, f"test_vis_5339{args.suffix}.png")
        save_visualization(sub_grid, test_vis_path)
        
        # テスト用のGeoTIFFも保存 (320x320用の簡易アフィン)
        test_tif_path = os.path.join(args.output_dir, f"test_data_5339{args.suffix}.tif")
        transform = from_origin(MIN_LON + lon_idx, MIN_LAT + lat_idx + 1.0/1.5, DX, DY)
        crs = rasterio.crs.CRS.from_epsg(4612)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'nodata': np.nan,
            'width': CELLS_PER_MESH, 'height': CELLS_PER_MESH, 'count': 1,
            'crs': crs, 'transform': transform, 'compress': 'lzw'
        }
        with rasterio.open(test_tif_path, 'w', **profile) as dst:
            dst.write(sub_grid, 1)
        print(f"テスト用GeoTIFFを保存しました: {test_tif_path}")
        
        # テスト用 NumPy .npz も保存
        test_npz_path = os.path.join(args.output_dir, f"test_data_5339{args.suffix}.npz")
        np.savez_compressed(test_npz_path, elev=sub_grid)
        print(f"テスト用NumPyデータを保存しました: {test_npz_path}")
        
        print("\nテスト実行が正常に終了しました！出力ファイルを確認してください。")
        return

    # 本番一括処理モード
    print(f"\n--- 【本番モード（属性: {args.attr}）】 {len(dbf_files)} ファイルを一括処理します ---")
    processed_counts = {'total_records': 0, 'skipped_records': 0}
    start_time = time.time()
    
    for i, dbf_path in enumerate(dbf_files, 1):
        file_start = time.time()
        file_name = os.path.basename(dbf_path)
        # フォルダ名やメッシュコードを簡易取得
        mesh_id = file_name.split('_')[1].split('-')[0]
        
        count = process_dbf(dbf_path, grid, processed_counts, args.attr)
        file_elapsed = time.time() - file_start
        
        # 10ファイルごとに進捗を簡潔に出力
        if i == 1 or i % 10 == 0 or i == len(dbf_files):
            print(f"[{i}/{len(dbf_files)}] メッシュ {mesh_id}: {count} レコードプロット (このファイルの処理: {file_elapsed:.2f}秒, 累計プロット: {processed_counts['total_records']})")
            
    total_elapsed = time.time() - start_time
    print(f"\n全ファイルの読み込み＆プロットが完了しました！")
    print(f"  総処理ファイル数: {len(dbf_files)}")
    print(f"  総プロットレコード数: {processed_counts['total_records']}")
    print(f"  スキップレコード数: {processed_counts['skipped_records']}")
    print(f"  データパース所要時間: {total_elapsed:.1f}秒 (平均: {total_elapsed/len(dbf_files):.2f}秒/ファイル)")
    
    # 成果物の保存
    # 1. 可視化RGB画像の保存
    vis_path = os.path.join(args.output_dir, f"japan_elevation_map{args.suffix}.png")
    save_visualization(grid, vis_path)
    
    # 2. 地理情報付きGeoTIFFの保存
    tif_path = os.path.join(args.output_dir, f"japan_elevation_data{args.suffix}.tif")
    save_geotiff(grid, tif_path)
    
    # 3. 圧縮NumPy配列データの保存
    npz_path = os.path.join(args.output_dir, f"japan_elevation_data{args.suffix}.npz")
    print("NumPy圧縮バイナリの保存中...")
    np.savez_compressed(npz_path, elev=grid)
    print(f"NumPy圧縮バイナリを保存しました: {npz_path}")
    
    print("\nすべてのデータ統合＆出力処理が完全に完了しました！🎉")

if __name__ == "__main__":
    main()
