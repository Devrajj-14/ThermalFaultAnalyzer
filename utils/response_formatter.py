"""
Response formatting utilities.
Builds the standardised API response JSON.
"""

from typing import Dict


def build_success_response(
    fault_type: str,
    severity: str,
    confidence: float,
    panel_health: str,
    score: float,
    timeline: Dict,
    metrics: Dict,
    xai: Dict,
    genai: Dict,
    images: Dict,
    inference_mode: str,
    ranked_regions: list = None,
) -> Dict:
    """
    Build the standardised success response.
    
    Args:
        fault_type: classified fault type
        severity: classified severity
        confidence: prediction confidence (0-1)
        panel_health: derived health status
        score: composite severity score (0-100)
        timeline: action and risk timeline dict
        metrics: performance metrics dict
        xai: XAI output dict
        genai: GenAI summary dict
        images: dict of base64 encoded images
        inference_mode: "model" or "rule_based"
    
    Returns:
        Complete response dict
    """
    return {
        "success": True,
        "fault_type": fault_type,
        "severity": severity,
        "confidence": confidence,
        "panel_health": panel_health,
        "score": score,
        "inference_mode": inference_mode,
        "action_timeline": timeline.get("action_timeline", ""),
        "risk_timeline": timeline.get("risk_timeline", ""),
        "metrics": {
            "hotspot_area_percent": metrics.get("hotspot_area_percent", 0.0),
            "region_count": metrics.get("region_count", 0),
            "thermal_contrast": metrics.get("thermal_contrast", 0.0),
            "estimated_temp_delta": metrics.get("estimated_temp_delta", 0.0),
            "power_drop_estimate": metrics.get("power_drop_estimate", 0.0),
        },
        "analysis": {
            "power_output_drop": metrics.get("power_output_drop", {}),
            "performance_ratio": metrics.get("performance_ratio", {}),
            "degradation_rate": metrics.get("degradation_rate", {}),
            "temperature_effect": metrics.get("temperature_effect", {}),
            "thermal_estimate": metrics.get("thermal_estimate", {})
        },
        "xai": {
            "top_reason": xai.get("top_reason", ""),
            "explanation_points": xai.get("explanation_points", []),
            "visual_path": xai.get("visual_path", ""),
            "xai_image_base64": xai.get("xai_image_base64", ""),
            "ranked_regions": ranked_regions or [],
        },
        "genai": {
            "user_summary": genai.get("user_summary", ""),
            "maintenance_advice": genai.get("maintenance_advice", ""),
            "technical_summary": genai.get("technical_summary", ""),
            "source": genai.get("source", "template")
        },
        "images": {
            "input_image": images.get("input_image", ""),
            "fault_detection": images.get("fault_detection", ""),
            "fault_parts": images.get("fault_parts", [])
        },
        # Legacy fields for backward compatibility
        "prediction": _fault_type_to_legacy(fault_type),
        "input_image": images.get("input_image", ""),
        "fault_detection": images.get("fault_detection", ""),
        "fault_parts": images.get("fault_parts", [])
    }


def build_error_response(message: str, status_code: int = 500) -> tuple:
    """Build a standardised error response."""
    return {"success": False, "error": message}, status_code


def _fault_type_to_legacy(fault_type: str) -> str:
    """Map new fault_type to legacy prediction string for backward compatibility."""
    mapping = {
        "normal": "No Fault (Normal)",
        "hotspot": "Hotspot Detected",
        "severe_thermal_anomaly": "Severe Thermal Anomaly Detected"
    }
    return mapping.get(fault_type, fault_type)


def compute_score(features: Dict, fault_type: str, severity: str) -> float:
    """
    Compute a composite severity score (0-100) for display.

    Uses the same weights as classify_severity so the score is consistent
    with the severity label shown in the banner.
    """
    if fault_type == "normal":
        return round(min(features.get("hotspot_area_percent", 0) * 2, 15), 1)

    area = features.get("hotspot_area_percent", 0)
    contrast = features.get("thermal_contrast", 0)
    regions = features.get("region_count", 0)

    # Normalise on the same scale used by classify_severity
    area_score = min((area / 20.0) * 100, 100)       # 20% area = max
    contrast_score = min((contrast / 35.0) * 100, 100)  # 35 contrast = max
    region_score = min((regions / 5.0) * 100, 100)   # 5 regions = max

    raw = area_score * 0.45 + contrast_score * 0.35 + region_score * 0.20

    if fault_type == "severe_thermal_anomaly":
        raw = min(raw * 1.15, 100)

    return round(raw, 1)
