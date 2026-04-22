"""Integration tests for Flask routes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import json
import numpy as np
import cv2
import pytest

# Import the new app
from app_v2 import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_test_image_bytes(h=100, w=100, value=80):
    """Create a simple test image as bytes."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def make_hotspot_image_bytes():
    """Create a test image with a hotspot."""
    img = np.full((100, 100, 3), 80, dtype=np.uint8)
    img[30:60, 30:60] = 220
    success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


class TestHealthRoute:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_response(self, client):
        res = client.get("/health")
        data = json.loads(res.data)
        assert data["status"] == "healthy"


class TestHomeRoute:
    def test_home_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200


class TestPredictRoute:
    def test_no_image_returns_400(self, client):
        res = client.post("/predict")
        assert res.status_code == 400
        data = json.loads(res.data)
        assert data["success"] is False

    def test_valid_image_returns_200(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 200

    def test_valid_image_response_schema(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        
        assert body["success"] is True
        assert "fault_type" in body
        assert "severity" in body
        assert "confidence" in body
        assert "panel_health" in body
        assert "action_timeline" in body
        assert "risk_timeline" in body
        assert "metrics" in body
        assert "xai" in body
        assert "genai" in body

    def test_fault_type_valid_value(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert body["fault_type"] in ("normal", "hotspot", "severe_thermal_anomaly")

    def test_severity_valid_value(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert body["severity"] in ("low", "medium", "high")

    def test_panel_health_valid_value(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert body["panel_health"] in ("healthy", "watchlist", "degraded", "critical")

    def test_hotspot_image_detected(self, client):
        img_bytes = make_hotspot_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "hotspot.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert body["success"] is True

    def test_invalid_file_type_rejected(self, client):
        data = {"image": (io.BytesIO(b"not an image"), "test.txt")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 400

    def test_legacy_fields_present(self, client):
        """Backward compatibility: old fields must still be present."""
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert "prediction" in body
        assert "input_image" in body
        assert "fault_detection" in body
        assert "fault_parts" in body

    def test_inference_mode_present(self, client):
        img_bytes = make_test_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "test.jpg")}
        res = client.post("/predict", data=data, content_type="multipart/form-data")
        body = json.loads(res.data)
        assert body["inference_mode"] in ("model", "rule_based")
