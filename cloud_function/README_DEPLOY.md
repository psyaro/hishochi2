# デプロイ手順（分離構成）

- **フロントエンド**: GitHub Pages で `docs/index.html` を配信（静的・即表示）
- **API**: Google Cloud Functions が `/elevation` を提供（標高データはここに同梱）

---

## 1. Cloud Function（API）をデプロイ

データファイルを関数フォルダに同梱する必要があります。デプロイ前にコピー:

```powershell
Copy-Item ..\japan_elevation_data.npz     .
Copy-Item ..\japan_elevation_data_min.npz .
```

第2世代でデプロイ（東京リージョン例）:

```powershell
gcloud functions deploy elevation `
  --gen2 `
  --runtime python312 `
  --region asia-northeast1 `
  --source . `
  --entry-point elevation `
  --trigger-http `
  --allow-unauthenticated `
  --memory 512Mi `
  --timeout 60s
```

> メモリは npz（合計 約31MB）をロードするため 512Mi 以上を推奨。
> デプロイ完了後に表示される **トリガーURL** を控えてください。

動作確認:

```powershell
curl "https://<表示されたURL>?lat=35.3606&lon=138.7273"
```

## 2. GitHub Pages（フロントエンド）を公開

1. `docs/index.html` の `API_BASE` を、手順1のトリガーURLに書き換える。
2. リポジトリにプッシュ。
3. GitHub → Settings → Pages → Source を **「main ブランチ / docs フォルダ」** に設定。
4. 公開された `https://<user>.github.io/<repo>/` を開く。

データ本体（.npz, 計 約31MB）は GitHub Pages には置きません（API側にのみ同梱）。
静的HTMLは即表示され、コールドスタートの遅延はデータ取得時のみに限定されます。

---

## データ出典

国土数値情報「標高・傾斜度5次メッシュデータ（G04-d）」（国土交通省）
https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-G04-d.html
を加工して作成。出典表記はローカル版・API版・静的版すべてに明記済み。
