"""
Inference service: orchestrates model-based or rule-based inference.
Tries model-based first; falls back to rule-based if model is unavailable.
"""

import os
import logging
import numpy as np
from typing import Dict

from services.rule_based_service import rule_based_inference
from config.settings import (
    BASE_EXPECTED_OUTPUT_KWH, INSTALLED_POWER_KWP,
    IRRADIANCE_KWH_M2_DAY, DAYS_PER_YEAR, DEGRADATION_YEARS,
    STC_TEMP_C, TEMP_COEFF, RATED_POWER_W
)

logger = logging.getLogger(__name__)

# Model path — if this file exists, model-based inference is attempted
MODEL_PATH = os.path.join("models", "solar_panel_gnn.pth")


def _is_model_available() -> bool:
    """Check if a trained model file exists and is non-empty."""
    return os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024


def run_inference(img_bgr: np.ndarray) -> Dict:
    """
    Main inference entry point.
    
    Tries model-based inference first; falls back to rule-based.
    
    Args:
        img_bgr: OpenCV BGR image array
    
    Returns:
        Standardised inference result dict
    """
    if _is_model_available():
        try:
            result = _model_based_inference(img_bgr)
            result["inference_mode"] = "model"
            return result
        except Exception as exc:
            logger.warning("Model inference failed (%s). Falling back to rule-based.", exc)

    result = rule_based_inference(img_bgr)
    result["inference_mode"] = "rule_based"
    return result


def _model_based_inference(img_bgr: np.ndarray) -> Dict:
    """
    Placeholder for future CNN/GNN model-based inference.
    
    The GNN model in this repo operates on graph data built from images,
    not directly on raw pixel arrays. Integrating it requires the full
    graph-construction pipeline (see utils/preprocess.py).
    
    For now this raises NotImplementedError so the fallback is used.
    Future work: replace this with a CNN classifier trained on the
    C/H/S labelled dataset.
    """
    raise NotImplementedError("Model-based inference not yet integrated into the live pipeline.")


def compute_performance_metrics(features: Dict, fault_type: str, severity: str) -> Dict:
    """
    Compute solar performance metrics from thermal features.

    Power drop is estimated from hotspot area + severity so it stays
    consistent with the classification narrative.
    Temperature effect is only shown when the panel is genuinely hot
    (fault case) and uses correct sign convention.
    Thermal estimate is clearly labelled as a visual proxy.
    """
    hotspot_area = features.get("hotspot_area_percent", 0.0)
    thermal_contrast = features.get("thermal_contrast", 0.0)

    # ── Power drop estimate ────────────────────────────────────────────────────
    # Scale with both area and severity so it doesn't contradict the narrative.
    severity_multiplier = {"low": 0.4, "medium": 0.8, "high": 1.4}.get(severity, 0.8)
    if fault_type == "normal":
        power_drop_estimate = round(hotspot_area * 0.1, 2)
    else:
        power_drop_estimate = round(hotspot_area * severity_multiplier, 2)

    # ── Simulated annual output ────────────────────────────────────────────────
    fault_factor = 1.0 - (power_drop_estimate / 100.0)
    actual_output = BASE_EXPECTED_OUTPUT_KWH * fault_factor
    power_drop_pct = round(100.0 - fault_factor * 100.0, 2)

    # ── Performance Ratio ─────────────────────────────────────────────────────
    theoretical_max = INSTALLED_POWER_KWP * IRRADIANCE_KWH_M2_DAY * DAYS_PER_YEAR
    pr = round((actual_output / theoretical_max) * 100.0, 2)

    # ── Annual Degradation Rate ───────────────────────────────────────────────
    degradation_rate = round(
        ((BASE_EXPECTED_OUTPUT_KWH - actual_output) / (BASE_EXPECTED_OUTPUT_KWH * DEGRADATION_YEARS)) * 100.0, 2
    )

    # ── Temperature effect ────────────────────────────────────────────────────
    # Only shown for fault cases. thermal_contrast (0-100 scale) is used as a
    # proxy for temperature elevation above background.
    # TEMP_COEFF is negative (power loss per °C above STC).
    # We only show this for anomaly cases; for normal panels it's omitted.
    if fault_type != "normal" and thermal_contrast > 0:
        # Map contrast (0-100) to a rough °C delta (0-40°C range)
        estimated_delta_c = round(thermal_contrast * 0.4, 1)
        power_loss_w = round(abs(TEMP_COEFF) * estimated_delta_c * RATED_POWER_W, 1)
        temp_effect = {
            "observed_value": power_loss_w,
            "label": "Thermal Power Loss",
            "inference": (
                f"Estimated {power_loss_w} W reduction due to elevated panel temperature "
                f"(~{estimated_delta_c}°C above background, visual proxy only)."
            ),
        }
    else:
        temp_effect = None  # omit entirely for normal panels

    # ── Thermal estimate ──────────────────────────────────────────────────────
    # Clearly labelled as a non-calibrated visual proxy.
    if fault_type != "normal" and thermal_contrast > 0:
        thermal_estimate = {
            "value": "Elevated",
            "units": "",
            "note": f"Thermal contrast score: {thermal_contrast:.1f} (visual proxy, not sensor-calibrated)",
        }
    else:
        thermal_estimate = {
            "value": "Normal",
            "units": "",
            "note": "No significant thermal elevation detected",
        }

    # ── Power drop status — consistent with severity ──────────────────────────
    # For fault cases, always flag as requiring attention regardless of % value.
    if fault_type == "normal":
        pd_status = "✅"
        pd_inference = "Within normal range"
    elif severity == "low":
        pd_status = "⚠️"
        pd_inference = f"Minor estimated impact (~{power_drop_estimate:.1f}%). Monitor at next inspection."
    elif severity == "medium":
        pd_status = "🔴"
        pd_inference = f"Moderate estimated impact (~{power_drop_estimate:.1f}%). Inspection recommended."
    else:  # high
        pd_status = "🔴"
        pd_inference = f"Significant estimated impact (~{power_drop_estimate:.1f}%). Prompt inspection required."

    return {
        "power_output_drop": {
            "observed_value": power_drop_estimate,
            "standard_threshold": "<10%",
            "status": pd_status,
            "inference": pd_inference,
        },
        "performance_ratio": {
            "observed_value": pr,
            "standard_threshold": "≥75%",
            "status": "🔴" if pr < 75 else "✅",
            "inference": "PR below standard — system inefficiencies detected." if pr < 75 else "Within acceptable range",
        },
        "degradation_rate": {
            "observed_value": degradation_rate,
            "standard_threshold": "<0.7%/year",
            "status": "🔴" if degradation_rate > 0.7 else "✅",
            "inference": "Elevated degradation rate — monitor closely." if degradation_rate > 0.7 else "Within normal range",
        },
        "temperature_effect": temp_effect,   # None when not applicable
        "thermal_estimate": thermal_estimate,
        # ── Flat fields (single source of truth for all downstream consumers) ──
        "hotspot_area_percent": features.get("hotspot_area_percent", 0.0),
        "region_count": features.get("region_count", 0),
        "thermal_contrast": features.get("thermal_contrast", 0.0),
        "estimated_temp_delta": features.get("estimated_temp_delta", 0.0),
        "power_drop_estimate": power_drop_estimate,
    }


def derive_panel_health(fault_type: str, severity: str, confidence: float) -> str:
    """
    Derive panel health status from classification results.
    
    Uses more conservative mapping - "critical" only for truly severe cases.
    
    Returns:
        "healthy" | "watchlist" | "degraded" | "critical"
    """
    if fault_type == "normal":
        return "healthy"
    
    if fault_type == "hotspot":
        if severity == "low":
            return "watchlist"
        elif severity == "medium":
            return "degraded"
        else:  # high
            return "degraded"
    
    if fault_type == "severe_thermal_anomaly":
        if severity == "low":
            return "degraded"
        elif severity == "medium":
            return "degraded"
        else:  # high - only this gets critical
            return "critical"
    
    return "watchlist"
