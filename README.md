# kao_cal

正面顔写真から顔の寸法比を抽出し、日本人の参照統計と比較して、形態統計上の平均への近さを可視化するWebアプリです。
現在は設計／MVP段階で、このリポジトリには設計文書を整備しています。

MVP v1は、個票取得を待たず、公開平均値と標準偏差から比率分布を近似する方式です。
「平均顔に近いほど美しい」というMVP上の仮説を置きますが、平均顔近似スコアは普遍的な美醜や実測人口順位を表しません。

予定する技術構成は、Python、FastAPI、Jinja2、HTML/CSS、必要最小限のJavaScript、MediaPipe Face Landmarker、NumPy、SciPy、OpenCV、Docker、docker composeです。
現在はPhase 1の環境構築段階です。画像アップロードや解析はまだ実装していません。

## 開発環境の起動

必要環境はDockerとDocker Composeです。ホストへのPythonのインストールは不要です。
Linux x86-64向けのMediaPipeを使用します。

```sh
docker compose build
docker compose up
```

ブラウザで http://localhost:8000/ を開くと、起動確認のメッセージを表示します。
`app/` と `tests/` の変更はコンテナへ反映され、アプリのソース変更時にはUvicornが再読み込みします。

停止するには別のターミナルで次を実行します。

```sh
docker compose down
```

テストは次のコマンドで実行します。

```sh
docker compose run --rm app python -m pytest -q
```

Python 3.12.9とPython依存のバージョンは `Dockerfile`、`requirements.txt`、`constraints.txt` に固定しています。
Face LandmarkerのAPIはテストで確認します。モデルファイルは今回使用しないため取得していません。
画像解析の実装時に配布元、モデルの実体と識別子を固定し、互換性と採用点を検証します。

- [設計文書の索引](docs/README.md)
