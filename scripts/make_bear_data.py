"""
downloads/kuma/bear.csv → docs/data/bear.json.gz に変換する。

出力フォーマット（配列の配列でサイズ最小化）:
  [[lat, lon, "YYYY-MM-DD", "都道府県市区地名", "コメント", headcount], ...]

フィルタ: ParentFlag=True のみ（重複サブレコードを除外）
"""

import os
import json
import gzip
import pandas as pd

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(_ROOT, "downloads", "kuma", "bear.csv")
OUT_GZ  = os.path.join(_ROOT, "docs", "data", "bear.json.gz")


def main():
    df = pd.read_csv(IN_CSV)
    print(f"読み込み完了: {len(df)} 件")

    df = df[df["ParentFlag"] == True].reset_index(drop=True)
    print(f"ParentFlag=True でフィルタ後: {len(df)} 件")

    df["date"] = pd.to_datetime(df["IssueDate"]).dt.strftime("%Y-%m-%d")

    payload = []
    for row in df.itertuples(index=False):
        # IssueComment は "#" 区切りで「コメント#出没原因:xxx#出典:xxx#外部ID:xxx」
        raw_cmnt = str(row.IssueComment) if pd.notna(row.IssueComment) else ""
        comment  = raw_cmnt.split("#")[0].strip()

        location = (
            (str(row.PrefectureName) if pd.notna(row.PrefectureName) else "") +
            (str(row.CityName)       if pd.notna(row.CityName)       else "") +
            (str(row.SectionNameText) if pd.notna(row.SectionNameText) else "")
        )

        n = int(row.HeadCount) if pd.notna(row.HeadCount) else 1

        payload.append([
            round(float(row.Latitude),  4),
            round(float(row.Longitude), 4),
            row.date,
            location,
            comment,
            n,
        ])

    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(OUT_GZ), exist_ok=True)
    with gzip.open(OUT_GZ, "wb", compresslevel=9) as f:
        f.write(json_bytes)

    raw_kb = len(json_bytes) / 1024
    gz_kb  = os.path.getsize(OUT_GZ) / 1024
    print(f"保存完了: {OUT_GZ}")
    print(f"  raw {raw_kb:.0f} KB → gz {gz_kb:.0f} KB")
    print(f"  件数: {len(payload)}")


if __name__ == "__main__":
    main()
