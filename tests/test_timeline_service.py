"""Tests for timeline_service.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from services.timeline_service import get_timeline


class TestGetTimeline:
    def test_normal_returns_monitor(self):
        result = get_timeline("normal", "low")
        assert "30 days" in result["action_timeline"]
        assert "6" in result["risk_timeline"] or "12" in result["risk_timeline"]

    def test_hotspot_low_returns_7_14_days(self):
        result = get_timeline("hotspot", "low")
        assert "7" in result["action_timeline"] or "14" in result["action_timeline"]

    def test_hotspot_high_returns_2_5_days(self):
        result = get_timeline("hotspot", "high")
        assert "2" in result["action_timeline"] or "5" in result["action_timeline"]

    def test_severe_high_returns_24_48_hours(self):
        result = get_timeline("severe_thermal_anomaly", "high")
        assert "24" in result["action_timeline"] or "48" in result["action_timeline"]

    def test_severe_medium_returns_2_5_days(self):
        result = get_timeline("severe_thermal_anomaly", "medium")
        assert "2" in result["action_timeline"] or "5" in result["action_timeline"]

    def test_unknown_fault_type_falls_back(self):
        result = get_timeline("unknown_type", "medium")
        assert "action_timeline" in result
        assert "risk_timeline" in result

    def test_all_keys_present(self):
        result = get_timeline("hotspot", "medium")
        assert "action_timeline" in result
        assert "risk_timeline" in result

    def test_case_insensitive(self):
        r1 = get_timeline("HOTSPOT", "HIGH")
        r2 = get_timeline("hotspot", "high")
        assert r1 == r2
