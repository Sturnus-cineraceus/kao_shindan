# kao_cal

正面顔写真から顔ランドマークを取得し、日本人の公開参照統計との比較につなげるWebアプリです。現在は統計比較コア、画像デコード、MediaPipe Face Landmarkerの内部処理に加え、顔ランドマークからの距離計測と比率特徴量生成を実装しています。

平均顔近似スコアは顔形態の統計的な平均への近さを示す試作指標です。普遍的・客観的な美醜や実測人口順位を表しません。

## 現在実装済みの範囲

- JPEG／PNGのバイト列をメモリ上でデコードし、EXIF Orientationを適用してRGBへ正規化
- 入力サイズ・画像寸法の上限確認と、解析用データからのメタデータ除去
- MediaPipe Face Landmarkerの静止画像解析
- 顔ごとのランドマークおよびfacial transformation matrixをアプリ内部の型へ変換
- 指定ランドマークから4つのpixel距離を計算し、外眼角幅を分母とする3つの無次元比率を生成
- 参照統計データの読み込みと統計比較コア

元画像を保存するWebアップロード画面、`POST /analyze`、顔数や姿勢のvalidation、画質判定、特徴量から統計比較コアへの解析パイプライン統合、結果画面、ランドマーク描画は未実装です。

## Face Landmarkerモデル

MediaPipe公式Face Landmarker配布物を使用します。

- モデル: Face Landmarker float16、モデルディレクトリ版 `1`
- 識別子: `face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- 公式配布元: [Google Cloud Storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- 配置先: コンテナ内 `/opt/kao-cal/models/face_landmarker.task`
- 固定方法: Docker build時に取得し、SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` を検証
- 手動取得: `python scripts/download_face_landmarker.py`（`FACE_LANDMARKER_MODEL_PATH` で保存先を変更可能）

モデルファイルはGitに含めません。Docker buildでモデル取得に失敗した場合やチェックサムが一致しない場合はbuildを失敗させます。公式ガイドに沿い、画像モード、`num_faces=2`、行列出力有効、顔検出・顔存在confidence 0.7で生成します。

## Docker Compose

必要環境はDockerとDocker Composeです。モデル取得のためDocker build時にネットワーク接続が必要です。

```sh
docker compose build
docker compose up
```

ブラウザで http://localhost:8000/ を開きます。ソース変更時にはUvicornが再読み込みします。停止は次のコマンドです。

```sh
docker compose down
```

全テストを実行します。

```sh
docker compose run --rm app python -m pytest -q
```

テストには画像デコードの単体テスト、MediaPipeアダプターのスタブテスト、コンテナに取得済みの実モデルを使う安定した空画像の推論スモークテストを含みます。Face Landmarker APIの依存確認は実モデルテストで行います。

## 設計

- [設計文書の索引](docs/README.md)
- [アーキテクチャ](docs/architecture.md)
- [画像解析](docs/image-analysis.md)
- [テスト方針](docs/testing.md)
