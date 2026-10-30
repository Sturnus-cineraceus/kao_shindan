from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationConfig:
    # MVPの顔サイズ・端寄り判定。値の根拠と境界条件は docs/validation.md を参照。
    face_size: float = 0.30
    edge_margin: float = 0.02
    # 姿勢角とランドマーク間距離比による、正面・口閉じ・両目開きの判定閾値。
    yaw: float = 10.0
    pitch: float = 10.0
    roll: float = 5.0
    mouth_ratio: float = 0.08
    eye_ratio: float = 0.10
    # 画質判定の初期値。blur は Laplacian 分散、dark/bright はグレースケール平均。
    # clipping は暗部・明部の画素割合で判定する。各条件の定義は docs/validation.md。
    blur: float = 80.0
    dark: float = 40.0
    bright: float = 215.0
    black_clipping: float = 0.15
    white_clipping: float = 0.15
    # 変換行列を回転行列として扱えるか、Euler角が特異点に近すぎないかの数値許容値。
    orthogonality_tolerance: float = 1e-3
    singularity_tolerance: float = 1e-6
    # 画質評価の計算量を抑えるため、縦横比を保ってこの長辺まで縮小する。
    quality_max_dimension: int = 512
    # 8bitグレースケールで暗部・明部の clipping 割合を数える画素境界。
    black_max: int = 5
    white_min: int = 250
