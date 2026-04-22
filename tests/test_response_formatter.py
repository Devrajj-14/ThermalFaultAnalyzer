"""Tests for response_formatter.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.response_formatter import build_success_response, build_error_response, compute_score


SAMPLE_METRICS = {
    "hotspot_area_percent": 8.4,
    "region_count": 2,
    "estimated_temp_delta": 14.2,
    "power_drop_estimate": 4.2,
    "power_output_drop": {"observed_value": 4.2, "standard_threshold": "<10%", "status": "✅", "inference": "Normal"},
    "performance_ratio": {"observed_value": 72.0, "standard_threshold": "≥75%", "status": "🔴", "inference": "PR < 75%"},
    "degradation_rate": {"observed_value": 0.5, "standard_threshold": "<0.7%/year", "status": "✅", "inference": "Normal"},
    "temperature_loss": {"observed_value": 100.0, "inference": "Temperature Loss: 100.0 W"},
    "derived_temperature": {"value": 47.0, "units": "°C"}
}

SAMPLE_XAI = {
    "top_reason": "Hotspot detected",
    "explanation_points": ["Point 1", "Point 2"],
    "visual_path": "/static/generated/xai/test.png",
    "xai_image_base64": ""
}

SAMPLE_GENAI = {
    "user_summary": "Test summary",
    "maintenance_advice": "Test advice",
    "technical_summary": "Test technical",
    "source": "template"
}

SAMPLE_IMAGES = {
    "input_image": "base64data",
    "fault_detection": "base64data",
    "fault_parts": []
}


class TestBuildSuccessResponse:
    def _build(self, fault_type="hotspot", severity="medium", confidence=0.82):
        return build_success_response(
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            panel_health="degraded",
            score=65.0,
            timeline={"action_timeline": "Inspect in 7 days", "risk_timeline": "Risk in 2 months"},
            metrics=SAMPLE_METRICS,
            xai=SAMPLE_XAI,
            genai=SAMPLE_GENAI,
            images=SAMPLE_IMAGES,
            inference_mode="rule_based"
        )

    def test_success_true(self):
        r = self._build()
        assert r["success"] is True

    def test_required_fields_present(self):
        r = self._build()
        required = ["fault_type", "severity", "confidence", "panel_health", "score",
                    "action_timeline", "risk_timeline", "metrics", "analysis", "xai", "genai", "images"]
        for field in required:
            assert field in r, f"Missing field: {field}"

    def test_legacy_fields_present(self):
        r = self._build()
        assert "prediction" in r
        assert "input_image" in r
        assert "fault_detection" in r
        assert "fault_parts" in r

    def test_legacy_prediction_mapping(self):
        r = self._build(fault_type="normal")
        assert r["prediction"] == "No Fault (Normal)"

        r = self._build(fault_type="hotspot")
        assert r["prediction"] == "Hotspot Detected"

        r = self._build(fault_type="severe_thermal_anomaly")
        assert r["prediction"] == "Severe Thermal Anomaly Detected"

    def test_metrics_subfields(self):
        r = self._build()
        m = r["metrics"]
        assert "hotspot_area_percent" in m
        assert "region_count" in m
        assert "estimated_temp_delta" in m
        assert "power_drop_estimate" in m

    def test_xai_subfields(self):
        r = self._build()
        x = r["xai"]
        assert "top_reason" in x
        assert "explanation_points" in x
        assert isinstance(x["explanation_points"], list)

    def test_genai_subfields(self):
        r = self._build()
        g = r["genai"]
        assert "user_summary" in g
        assert "maintenance_advice" in g
        assert "technical_summary" in g


class TestBuildErrorResponse:
    def test_returns_tuple(self):
        result = build_error_response("Something went wrong", 400)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_success_false(self):
        body, code = build_error_response("Error", 400)
        assert body["success"] is False
        assert body["error"] == "Error"
        assert code == 400


class TestComputeScore:
    def test_normal_low_score(self):
        features = {"hotspot_area_percent": 0.5, "thermal_contrast": 5.0, "region_count": 0}
        score = compute_score(features, "normal", "low")
        assert score < 10

    def test_severe_high_score(self):
        features = {"hotspot_area_percent": 20.0, "thermal_contrast": 55.0, "region_count": 7}
        score = compute_score(features, "severe_thermal_anomaly", "high")
        assert score > 50

    def test_score_in_range(self):
        features = {"hotspot_area_percent": 8.0, "thermal_contrast": 30.0, "region_count": 3}
        score = compute_score(features, "hotspot", "medium")
        assert 0 <= score <= 100
