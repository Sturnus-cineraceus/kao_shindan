from fastapi.testclient import TestClient

from app.main import app


def test_root() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.text == "kao_cal is running"


def test_scientific_dependencies_are_available() -> None:
    import cv2
    import mediapipe
    import numpy
    import scipy

    assert all((cv2.__version__, mediapipe.__version__, numpy.__version__, scipy.__version__))
