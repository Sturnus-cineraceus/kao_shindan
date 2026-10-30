from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.image_processing import DecodedImage
from app.landmarks import FaceLandmarkerService, LandmarkAnalysisError, LandmarkModelError


class StubTask:
    def __init__(self, result: object) -> None:
        self.result = result

    def __enter__(self) -> "StubTask":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def detect(self, _image: object) -> object:
        return self.result


def test_converts_mediapipe_result_to_internal_types(tmp_path: Path) -> None:
    model = tmp_path / "model.task"
    model.touch()
    points = [SimpleNamespace(x=i / 500, y=i / 600, z=0.1) for i in range(478)]
    matrix = np.eye(4, dtype=np.float32)
    raw = SimpleNamespace(face_landmarks=[points], facial_transformation_matrixes=[matrix])
    service = FaceLandmarkerService(model, lambda _: StubTask(raw))
    decoded = DecodedImage(np.zeros((8, 12, 3), dtype=np.uint8), 12, 8)
    result = service.analyze(decoded)
    assert (result.image_width, result.image_height, result.face_count) == (12, 8, 1)
    assert len(result.faces[0]) == 478
    assert result.faces[0][468].x == pytest.approx(468 / 500)
    assert result.faces[0][473].x == pytest.approx(473 / 500)
    assert result.faces[0][133].y == pytest.approx(133 / 600)
    assert result.faces[0][362].y == pytest.approx(362 / 600)
    assert result.faces[0][33].z == pytest.approx(0.1)
    assert result.faces[0][263].z == pytest.approx(0.1)
    assert result.faces[0][61].x == pytest.approx(61 / 500)
    assert result.faces[0][291].x == pytest.approx(291 / 500)
    assert result.facial_transformation_matrices[0][0][0] == pytest.approx(1.0)


def test_model_and_analysis_failures_are_distinct(tmp_path: Path) -> None:
    missing = FaceLandmarkerService(tmp_path / "missing.task")
    with pytest.raises(LandmarkModelError):
        missing.analyze(DecodedImage(np.zeros((2, 2, 3), dtype=np.uint8), 2, 2))

    model = tmp_path / "model.task"
    model.touch()
    broken = FaceLandmarkerService(model, lambda _: (_ for _ in ()).throw(RuntimeError("bad model")))
    with pytest.raises(LandmarkModelError):
        broken.analyze(DecodedImage(np.zeros((2, 2, 3), dtype=np.uint8), 2, 2))
    broken_detect = FaceLandmarkerService(model, lambda _: StubTask(SimpleNamespace(face_landmarks=None, facial_transformation_matrixes=None)))
    class FailingTask(StubTask):
        def detect(self, _image: object) -> object:
            raise RuntimeError("inference error")

    broken_detect = FaceLandmarkerService(model, lambda _: FailingTask(None))
    with pytest.raises(LandmarkAnalysisError):
        broken_detect.analyze(DecodedImage(np.zeros((2, 2, 3), dtype=np.uint8), 2, 2))


def test_real_model_image_mode_smoke(tmp_path: Path) -> None:
    model_path = Path("/opt/kao-cal/models/face_landmarker.task")
    if not model_path.is_file():
        pytest.skip("Model asset is provisioned by the Docker image")
    # A deterministic blank image exercises real model loading and image-mode inference.
    pixels = np.full((128, 128, 3), 127, dtype=np.uint8)
    result = FaceLandmarkerService(model_path).analyze(DecodedImage(pixels, 128, 128))
    assert result.face_count == 0
    assert result.faces == ()
