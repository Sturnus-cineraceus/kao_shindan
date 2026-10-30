from fastapi.testclient import TestClient

from app.main import app


def test_root() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.text == "kao_cal is running"


def test_scientific_dependencies_and_face_landmarker_api() -> None:
    import cv2
    import mediapipe
    import numpy
    import scipy
    from mediapipe.tasks.python import vision

    assert all((cv2.__version__, mediapipe.__version__, numpy.__version__, scipy.__version__))
    assert vision.FaceLandmarker is not None
    assert vision.FaceLandmarkerOptions is not None
