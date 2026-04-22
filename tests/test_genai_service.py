"""Tests for genai_service.py — specifically the fallback template behavior."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Ensure no API key is set for these tests
os.environ.pop("GEMINI_API_KEY", None)

from services.genai_service import _template_summary, generate_summary


SAMPLE_PAYLOAD = {
    "fault_type": "hotspot",
    "severity": "medium",
    "confidence": 0.82,
    "panel_health": "degraded",
    "action_timeline": "Inspect within 7-14 days",
    "risk_timeline": "May worsen in 2-4 months",
    "hotspot_area_percent": 8.4,
    "region_count": 2,
    "estimated_temp_delta": 14.2,
    "power_drop_estimate": 4.2
}


class TestTemplateSummary:
    def test_returns_all_keys(self):
        result = _template_summary(SAMPLE_PAYLOAD)
        assert "user_summary" in result
        assert "maintenance_advice" in result
        assert "technical_summary" in result
        assert "source" in result

    def test_source_is_template(self):
        result = _template_summary(SAMPLE_PAYLOAD)
        assert result["source"] == "template"

    def test_normal_fault_advice(self):
        payload = {**SAMPLE_PAYLOAD, "fault_type": "normal"}
        result = _template_summary(payload)
        assert "routine" in result["maintenance_advice"].lower() or "monitor" in result["maintenance_advice"].lower()

    def test_severe_fault_advice_urgent(self):
        payload = {**SAMPLE_PAYLOAD, "fault_type": "severe_thermal_anomaly", "severity": "high"}
        result = _template_summary(payload)
        assert "urgent" in result["maintenance_advice"].lower() or "severe" in result["maintenance_advice"].lower()

    def test_user_summary_contains_fault_type(self):
        result = _template_summary(SAMPLE_PAYLOAD)
        assert "hotspot" in result["user_summary"].lower()

    def test_technical_summary_contains_metrics(self):
        result = _template_summary(SAMPLE_PAYLOAD)
        assert "8.4" in result["technical_summary"]


class TestGenerateSummaryFallback:
    def test_no_api_key_uses_template(self):
        """When GEMINI_API_KEY is not set, should use template."""
        result = generate_summary(SAMPLE_PAYLOAD)
        assert result["source"] in ("template", "template_fallback")
        assert "user_summary" in result
        assert "maintenance_advice" in result
