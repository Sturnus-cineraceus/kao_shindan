# 解析サービス

本書を解析サービスの内部契約の正本とする。実装済みの画像デコード、ランドマーク、特徴抽出、統計の各層を、Web層から独立したアプリケーションサービスが順序どおりに呼び出す。HTTPの入出力や画面表示は扱わない。

## 入力と責務

解析要求は画像 `bytes`、`reference_group`、任意の `additional_features` からなる。[統計モデル](statistics.md)に定義した5種類を選択でき、未知IDと重複を拒否する。参照群はユーザーが明示選択する `young_male`、`young_female`、`older_male`、`older_female` のいずれかで、画像から性別や年齢を推定して選ばない。未知のIDは拒否し、35〜59歳を別群へ自動割当てしない。

Web層はリクエスト値の構文・必須性を検証し、アップロードから得たbytesと参照群IDを渡す。解析サービスも参照データ境界でIDの妥当性を確認し、不正な群では処理を始めない。`Request`、`UploadFile`、`HTMLResponse`、`TemplateResponse`、Jinja2型はサービス契約に含めない。

解析サービスは次の順に処理を制御する。

```text
入力bytesとreference_group
  → 画像デコード・正規化（DecodedImage）
  → Face Landmarker（LandmarkResult）
  → validation・roll座標補正
  → 4距離の抽出（measurements）
  → 3比率と選択された追加比率の生成（features）
  → statistics（compare_features）
  → 型付き解析成功結果
```

validationが成功するまでfeature extractionとstatisticsを呼ばない。feature extractionにはvalidationが生成した補正後の解析用ランドマークを渡し、元座標と補正後座標はHTTPへ返さない。品質条件に不合格の顔に参考スコアを返す経路は設けない。Face Landmarkerの `num_faces=2` は複数顔を拒否するための検出上限であり、許可する顔数はちょうど1人である。0人または2人以上なら失敗とし、複数顔から最大顔・先頭顔などを選択せず、後続の統計計算へ進まない。

## validation境界

validationはLandmarkResultとDecodedImageの寸法・画素を使って、解析可否と品質チェック結果を判定する。解析サービスは判定結果を制御フローに反映し、失敗したチェックをWeb層へ渡せる形で保持する。チェック対象は顔数、顔サイズ、画像端への接触・近接、頭部姿勢、目、口、ぼけ、露出である。具体的な定義と既存の初期値は[validation.md](validation.md)を参照する。

validationの点集合、姿勢算出規約、画質評価前処理と初期閾値は[validation.md](validation.md)を正本とする。失敗理由コードと優先順は本書のHTTP契約に定める。validationが失敗した場合は失敗チェックを含むvalidation結果を保持して処理を打ち切り、成功結果型やスコアを生成しない。

## 成功結果

成功時の内部結果は型付きの値とする。以下の項目を持ち、既存層と重複する項目はその定義を引き継ぐ。

| 項目 | 意味・出所 |
| --- | --- |
| `reference_group` | 選択された比較集団。statistics結果の値と一致する |
| `model_mode` | 統計方式。MVPは `summary_delta`。将来 `empirical` に交換できる境界を保つ |
| `profile_id` | `core3` に選択した追加特徴IDを内部の固定順で `+` 連結したもの |
| `measurements` | pixel単位の4距離。後述のキーを使う |
| `features` | 無次元比率。statistics `FEATURE_KEYS` のキーと順序を維持する |
| `z_scores` | statistics層の項目別近似zスコアをそのまま引き継ぐ |
| `distance` | statistics層の距離をそのまま引き継ぐ |
| `score` | statistics層の美形スコアをそのまま引き継ぐ |
| `volume_zone_score` | statistics層のボリュームゾーン度をそのまま引き継ぐ |
| `used_features` | 統計計算に使用した特徴IDをモデルの順序で並べた配列 |
| `dimension` | 使用した特徴数。現在のcore3では3 |
| `volume_zone_distribution` | `mode_distance` と、`distance`・`relative_density` を持つ101個の `points`。statistics層の計算結果をそのまま引き継ぐ |
| `quality_warnings` | 検証成功時の品質警告。安定したコードと日本語メッセージを保持する |
| `quality_checks` | 実施したチェック、通過可否、必要に応じ判定値を追跡可能にする。詳細型は実装時に定める |
| `approximation_notice` | UIが公開統計量由来の近似値で、実測人口順位ではないと表示するための情報 |
| `algorithm_version` | 画像から特徴量を生成する解析アルゴリズム全体の版 |
| `reference_model_version` | statistics結果から引き継ぐ参照モデルの版 |
| `regularization_used` | statistics結果から引き継ぐ正則化使用有無 |

`measurements` のキーは `interpupillary_breadth`、`interocular_breadth`、`biectocanthion_breadth`、`mouth_breadth`。単位は抽出に使った正規化画像上のpixelである。比率キーと順序は `statistics.FEATURE_KEYS` の `interpupillary_breadth_ratio`、`interocular_breadth_ratio`、`mouth_breadth_ratio` とする。statisticsの `StatisticsResult.features` を結果へ統合し、比率の再計算や値の意味変更をしない。

`algorithm_version` は画像由来の特徴生成処理を追跡し、現在は `2` とする。参照統計の版とは別概念であり、`reference_model_version` を使って参照データ／モデルを追跡する。MVPの `model_mode` は `summary_delta` とし、その式はstatistics層だけが所有する。

multipart入力は `image` と `reference_group` を各1件、`additional_feature` を0件以上受け付ける。同じ追加特徴の重複と未知IDは `400 invalid_feature_selection` とする。`profile_id` と `dimension` は入力として受け付けず、選択から決定する。

フロントエンドは分布グラフにAPIが返す計算済み点列を使用し、統計式を再実装しない。結果画面では点列に加えてAPIの距離、ボリュームゾーン度、最頻距離を表示する。

## 失敗と例外境界

失敗は原因の責務を保った分類でWeb層に提示する。HTTPステータスはWeb層が決定し、サービス例外へHTTPコードを埋め込まない。

| 分類 | 例 | 現行の下位層との関係 |
| --- | --- | --- |
| 入力画像 | 空bytes、未対応形式、デコード失敗、サイズ超過 | `ImageDecodeError` とその派生型を原因情報として保持 |
| ランドマーク解析 | モデル読込、MediaPipe実行失敗 | `LandmarkModelError`、`LandmarkAnalysisError` 等を保持 |
| validation | 顔なし、複数顔、品質条件不適合 | 失敗チェックを含むvalidation結果として表す |
| feature extraction | 必須点欠損、非有限値、0または不正分母 | `FeatureExtractionError` を保持 |
| statistics | 不正参照群、数値計算失敗、参照モデル不整合 | `ModelError` を保持。参照データ読込エラーは統計境界で扱われる |

下位層の具体的例外は原因として調査可能にしつつ、Web層が内部モジュール全体の例外一覧を把握しなくて済む解析失敗境界を置く。各下位例外の無差別な握りつぶしや、HTTP例外への変換はしない。想定外の障害は成功結果へ変換せず、内部詳細は画面に露出しない。サービスは原因コードと失敗チェックを持つ例外を返す。

## データの寿命

画像の寿命は1要求内に限る。入力bytesをデコードして得たDecodedImageをLandmarker、validation、feature extractionおよび必要な同一要求内の表示に使い、処理後に破棄する。EXIF等のメタデータは既存 `image_processing` の方針どおり解析データへ保持しない。

元画像をDBや恒久ファイルへ保存せず、元画像URL、再解析用画像ストレージを作らない。生ランドマークはvalidation、feature extraction、必要な同一要求内ガイド表示に一時利用するだけで、478点全量を通常の解析結果や永続データへ含めない。成功結果には必要な計測値と統計結果のみを含める。

## Web層との接続

```text
POST /analyze
  → Web入力検証
  → analysis service（bytes、reference_group）
  → 型付き解析結果
  → JSON応答
```

解析サービスはFastAPIから独立する。HTTP入力の検証・ステータス決定・テンプレートへの表示変換はWeb層の責務である。サービス結果や失敗情報をWeb層が `result_view` に変換してから描画する。


## HTTP APIと実行時の契約

`POST /analyze` は `multipart/form-data` の `image`（JPEGまたはPNGのファイル）と `reference_group`（文字列）を各1件必須とし、任意の `additional_feature` を複数件受け付ける。成功時に内部成功結果をJSONで返す。生ランドマークは返さない。画像はメモリ内でのみ受信し、本文上限は画像上限20 MiBにフォームの余裕64 KiBを加えた値とする。必須項目の重複、不正な構造、必須項目の欠落は `400 invalid_request`、追加特徴の未知IDまたは重複は `400 invalid_feature_selection`、未知の参照群は `400 invalid_reference_group`、画像の形式・デコード不正は `400 invalid_image`、入力サイズ・寸法制限違反は `413 image_size_limit` とする。

顔なし、複数顔、その他の品質不合格はそれぞれ `422 no_face`、`422 multiple_faces`、`422 validation_failed` とし、失敗したチェック名と理由を `checks` に返す。顔数を最優先し、その他は実行順に並べる。モデルのロード失敗は `503 model_unavailable`、その他の内部解析失敗は `500 analysis_error` とし、内部詳細を返さない。エラーは `{"error":{"code":...,"message":...,"checks":[...]}}` の形とし、チェックがない場合は空配列とする。ブラウザーからのクロスオリジン送信はOriginを同一オリジンに制限する。認証やCookieを用いないAPIであり、Originのない非ブラウザクライアントも受け付ける。

Landmarkerはアプリケーション単位で遅延ロードし、ロックで推論を直列化して再利用する。終了時に閉じる。アルゴリズム版は初期値 `1` とする。

## API issueとログ

HTTP応答は既存のトップレベル error.code を維持し、validation時に `issues: [{code, message}]` を返す。CheckResultの内部値・閾値・ランドマークIDは応答へ含めない。同時に確定したissueはすべて列挙し、顔未検出時は顔依存チェックを行わない。必要ランドマークの不足IDはログだけに記録し、画面では人が理解できる部位名にまとめる。

## 解析ログ

共通の標準logging設定からconsoleとUTF-8ファイルへ出力する。解析開始・完了・失敗概要はINFO、工程値とvalidation比較はDEBUG、入力・品質拒否はWARNING、モデル障害と想定外障害はERRORとする。想定外例外はスタックトレースを記録する。画像本体、bytes、全ランドマーク座標、multipart本文は記録しない。
