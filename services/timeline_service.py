"""
Timeline recommendation service.
Returns action and risk timelines based on fault type and severity.

NOTE: These are recommended action timelines, NOT exact remaining panel life predictions.
Exact lifetime estimation requires time-series performance history and is not
inferable from a single thermal image.

UPDATED: More realistic and less alarmist timelines.
"""

from typing import Dict, Tuple


# Centralised timeline rules with more realistic recommendations
_TIMELINE_RULES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "normal": {
        "low":    ("Monitor in 30 days", "Low risk over the next 6–12 months"),
        "medium": ("Monitor in 30 days", "Low risk over the next 6–12 months"),
        "high":   ("Monitor in 30 days", "Low risk over the next 6–12 months"),
    },
    "hotspot": {
        "low":    ("Inspect within 2–4 weeks", "May worsen over 3–6 months if untreated"),
        "medium": ("Inspect within 7–14 days", "May affect performance in 2–4 months"),
        "high":   ("Inspect within 3–7 days", "May reduce performance in 1–3 months"),
    },
    "severe_thermal_anomaly": {
        "low":    ("Inspect within 7–14 days", "Risk of further degradation within 2–3 months"),
        "medium": ("Inspect within 3–7 days", "May cause damage within 1–2 months"),
        "high":   ("Inspect within 24–72 hours", "May cause serious damage within weeks to 2 months"),
    },
}


def get_timeline(fault_type: str, severity: str) -> Dict[str, str]:
    """
    Return action and risk timeline recommendations.
    
    Args:
        fault_type: "normal" | "hotspot" | "severe_thermal_anomaly"
        severity: "low" | "medium" | "high"
    
    Returns:
        dict with "action_timeline" and "risk_timeline"
    """
    # Normalise inputs
    fault_type = fault_type.lower().strip()
    severity = severity.lower().strip()
    
    # Fallback to hotspot/medium if unknown
    rule = _TIMELINE_RULES.get(fault_type, _TIMELINE_RULES["hotspot"])
    action, risk = rule.get(severity, rule["medium"])
    
    return {
        "action_timeline": action,
        "risk_timeline": risk
    }
