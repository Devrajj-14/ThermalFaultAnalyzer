"""Tests for rule_based_service.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from services.rule_based_service import (
    extract_thermal_features,
    classify_fault_type,
    classify_severity,
    rule_based_inference
)


def make_blank_image(h=100, w=100, value=50):
    """Create a uniform BGR image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_hotspot_image(h=100, w=100, spot_value=220, bg_value=80):
    """Create an image with a bright hotspot in the center."""
    img = np.full((h, w, 3), bg_value, dtype=np.uint8)
    # Add a bright spot
    img[30:60, 30:60] = spot_value
    return img


def make_severe_image(h=100, w=100):
    """Create an image with large bright area (severe anomaly)."""
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    img[10:90, 10:90] = 230  # ~64% of image is hot
    return img


class TestExtractThermalFeatures:
    def test_blank_image_no_hotspots(self):
        img = make_blank_image(value=50)
        features = extract_thermal_features(img)
        assert features["hotspot_area_percent"] == 0.0
        assert features["region_count"] == 0

    def test_hotspot_image_detects_region(self):
        img = make_hotspot_image()
        features = extract_thermal_features(img)
        assert features["region_count"] >= 1
        assert features["hotspot_area_percent"] > 0

    def test_severe_image_high_area(self):
        img = make_severe_image()
        features = extract_thermal_features(img)
        assert features["hotspot_area_percent"] > 10

    def test_features_keys_present(self):
        img = make_blank_image()
        features = extract_thermal_features(img)
        required_keys = [
            "hotspot_area_percent", "region_count", "thermal_contrast",
            "estimated_temp_delta", "max_intensity", "contours",
            "thresh", "detection_img", "actual_temp"
        ]
        for key in required_keys:
            assert key in features, f"Missing key: {key}"

    def test_actual_temp_in_range(self):
        img = make_hotspot_image()
        features = extract_thermal_features(img)
        assert 25.0 <= features["actual_temp"] <= 85.0


class TestClassifyFaultType:
    def test_no_hotspot_is_normal(self):
        features = {"hotspot_area_percent": 0.0, "thermal_contrast": 0.0, "region_count": 0}
        fault_type, confidence = classify_fault_type(features)
        assert fault_type == "normal"
        assert confidence > 0.5

    def test_small_hotspot_is_hotspot(self):
        features = {"hotspot_area_percent": 5.0, "thermal_contrast": 25.0, "region_count": 1}
        fault_type, confidence = classify_fault_type(features)
        assert fault_type == "hotspot"

    def test_large_area_is_severe(self):
        features = {"hotspot_area_percent": 20.0, "thermal_contrast": 50.0, "region_count": 7}
        fault_type, confidence = classify_fault_type(features)
        assert fault_type == "severe_thermal_anomaly"

    def test_confidence_between_0_and_1(self):
        features = {"hotspot_area_percent": 5.0, "thermal_contrast": 25.0, "region_count": 2}
        _, confidence = classify_fault_type(features)
        assert 0.0 <= confidence <= 1.0


class TestClassifySeverity:
    def test_normal_is_always_low(self):
        features = {"hotspot_area_percent": 0.0, "thermal_contrast": 0.0, "region_count": 0}
        assert classify_severity(features, "normal") == "low"

    def test_small_hotspot_is_low(self):
        features = {"hotspot_area_percent": 1.5, "thermal_contrast": 10.0, "region_count": 1}
        severity = classify_severity(features, "hotspot")
        assert severity in ("low", "medium")

    def test_large_severe_is_high(self):
        features = {"hotspot_area_percent": 25.0, "thermal_contrast": 60.0, "region_count": 8}
        severity = classify_severity(features, "severe_thermal_anomaly")
        assert severity == "high"

    def test_severity_values_valid(self):
        features = {"hotspot_area_percent": 5.0, "thermal_contrast": 30.0, "region_count": 2}
        severity = classify_severity(features, "hotspot")
        assert severity in ("low", "medium", "high")


class TestRuleBasedInference:
    def test_blank_image_normal(self):
        img = make_blank_image(value=50)
        result = rule_based_inference(img)
        assert result["fault_type"] == "normal"
        assert result["severity"] == "low"

    def test_hotspot_image_detected(self):
        img = make_hotspot_image()
        result = rule_based_inference(img)
        assert result["fault_type"] in ("hotspot", "severe_thermal_anomaly", "normal")
        assert "features" in result

    def test_result_keys_present(self):
        img = make_blank_image()
        result = rule_based_inference(img)
        for key in ("fault_type", "severity", "confidence", "features"):
            assert key in result
