# 設計文書の索引

このディレクトリの文書をアプリケーションの正式な設計書とする。
数式や閾値は各専門文書で定義し、他の文書はその定義を参照する。
未決定の事項はTODO、要検証、初期値、暫定として記す。

| 文書 | 内容 | 主な読者 | 関連文書 |
| --- | --- | --- | --- |
| [overview.md](overview.md) | 目的、思想、MVPと将来の範囲 | 企画担当、開発者、レビュー担当 | design、dataset |
| [design.md](design.md) | 入力から破棄までの全体フロー | 開発者全員 | 各専門文書 |
| [statistics.md](statistics.md) | 特徴の順序、近似数式、スコア、モデル交換 | 統計実装者、検証担当 | dataset、image-analysis、testing |
| [image-analysis.md](image-analysis.md) | 採用点、座標、ピクセル距離、画像前処理 | 画像処理実装者 | validation、statistics |
| [validation.md](validation.md) | 解析不可条件、初期閾値、理由表示 | 実装者、画面設計者、検証担当 | image-analysis、testing |
| [dataset.md](dataset.md) | 参照群、取得条件、データ制約、保持形式 | データ担当、統計実装者 | statistics、overview |
| [architecture.md](architecture.md) | 技術構成、責務、HTTPと内部結果 | 実装者、運用担当 | design、analysis-service、statistics、non-functional-requirements |
| [analysis-service.md](analysis-service.md) | 解析サービスの入出力、処理順、結果、失敗、寿命 | 解析サービス実装者 | architecture、validation、statistics、testing |
| [non-functional-requirements.md](non-functional-requirements.md) | 性能、安全性、プライバシー、運用、保守 | 実装者、運用担当、レビュー担当 | architecture、validation、testing |
| [testing.md](testing.md) | 単体、境界値、画像、回帰、受入検証 | 実装者、検証担当 | statistics、validation、non-functional-requirements |
| [decisions.md](decisions.md) | 旧仕様の廃止、設計判断、未決定事項 | レビュー担当、実装開始担当 | 全文書 |
