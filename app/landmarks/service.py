from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np

from app.image_processing import DecodedImage
from .models import FaceLandmark, LandmarkResult, TransformMatrix


class LandmarkError(RuntimeError):
    pass


class LandmarkModelError(LandmarkError):
    pass


class LandmarkAnalysisError(LandmarkError):
    pass


class FaceLandmarkerService:
    def __init__(self, model_path: str | Path, landmarker_factory: Callable[..., Any] | None = None) -> None:
        self.model_path = Path(model_path)
        self._landmarker_factory = landmarker_factory
        self._task = None
        self._lock = Lock()

    def analyze(self, image: DecodedImage) -> LandmarkResult:
        with self._lock:
            if self._task is None:
                if not self.model_path.is_file():
                    raise LandmarkModelError("Face Landmarker model is missing")
                try:
                    self._task = self._create_landmarker()
                except Exception as exc:
                    raise LandmarkModelError("Face Landmarker model could not be loaded") from exc
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.rgb)
                raw = self._task.detect(mp_image)
            except Exception as exc:
                raise LandmarkAnalysisError("Face Landmarker analysis failed") from exc
        try:
            raw_faces = raw.face_landmarks or []
            raw_matrices = raw.facial_transformation_matrixes or []
            faces = tuple(tuple(FaceLandmark(float(p.x), float(p.y), float(p.z)) for p in face) for face in raw_faces)
            matrices: tuple[TransformMatrix | None, ...] = tuple(
                tuple(tuple(float(value) for value in row) for row in np.asarray(matrix).tolist())
                for matrix in raw_matrices
            )
            if len(matrices) < len(faces):
                matrices += (None,) * (len(faces) - len(matrices))
            return LandmarkResult(image.width, image.height, len(faces), faces, matrices)
        except Exception as exc:
            raise LandmarkAnalysisError("Face Landmarker analysis failed") from exc

    def close(self) -> None:
        with self._lock:
            if self._task is not None:
                task, self._task = self._task, None
                if hasattr(task, "close"):
                    task.close()

    def _create_landmarker(self) -> Any:
        if self._landmarker_factory is not None:
            return self._landmarker_factory(self.model_path)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
        )
        return vision.FaceLandmarker.create_from_options(options)
