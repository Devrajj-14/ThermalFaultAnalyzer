"""
Explainable AI (XAI) service.

Generates visual heatmap overlays and explanation metadata
based on actual detected thermal regions.

Uses the thermal score map (not raw grayscale) for the heatmap,
and ranks regions by importance score (not just area).
"""

import os
import uuid
import logging
import cv2
import numpy as np
from typing import Dict, List

from config.settings import XAI_OUTPUT_DIR

logger = logging.getLogger(__name__)


def _ensure_output_dir() -> None:
    os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)


def generate_xai_heatmap(img_bgr: np.ndarray, features: Dict) -> tuple:
    """
    Generate a clean heatmap overlay highlighting top detected fault regions.

    Uses the thermal_score_map for the heatmap base (reflects actual thermal
    distribution), and annotates the top ranked regions by importance score.

    Returns (file_path, base64_encoded_image).
    """
    _ensure_output_dir()

    ranked_regions = features.get("ranked_regions", [])
    thermal_score = features.get("thermal_score_map")

    # Fallback: if no thermal score map, use the thresh mask
    if thermal_score is None:
        thermal_score_u8 = features.get("thresh")
    else:
        thermal_score_u8 = (thermal_score * 255).astype(np.uint8)

    try:
        if thermal_score_u8 is None or len(ranked_regions) == 0:
            # No anomalies — return original image
            filename = f"xai_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(XAI_OUTPUT_DIR, filename)
            cv2.imwrite(filepath, img_bgr)
            import base64
            ok, buf = cv2.imencode(".png", img_bgr)
            b64 = base64.b64encode(buf).decode() if ok else ""
            return "/" + filepath.replace(os.sep, "/"), b64

        # Build heatmap from thermal score map
        heatmap = cv2.applyColorMap(thermal_score_u8, cv2.COLORMAP_INFERNO)

        # Blend: keep original image visible but overlay thermal heatmap
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

        total_area = img_bgr.shape[0] * img_bgr.shape[1]

        # Draw top regions ranked by importance (already sorted)
        for i, region in enumerate(ranked_regions[:5]):
            cnt = region["contour"]
            area = region["area"]
            importance = region["importance"]
            pct = (area / total_area) * 100

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Color by rank: #1 brightest red, descending
            colors = [
                (0, 0, 255),    # #1 red
                (0, 100, 255),  # #2 orange-red
                (0, 180, 255),  # #3 orange
                (0, 220, 255),  # #4 yellow
                (0, 255, 200),  # #5 yellow-green
            ]
            color = colors[i]
            thickness = 3 if i == 0 else 2

            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness)

            # Label: rank + area%
            label = f"#{i+1}  {pct:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 1)

            # Place label above box, clamp to image bounds
            lx = x
            ly = max(y - 6, lh + 4)

            # Dark background pill for readability
            cv2.rectangle(overlay, (lx - 1, ly - lh - 3), (lx + lw + 5, ly + 2), (15, 15, 15), -1)
            cv2.putText(overlay, label, (lx + 2, ly - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Minimal legend bottom-left
        _add_legend(overlay, features, len(ranked_regions))

        filename = f"xai_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(XAI_OUTPUT_DIR, filename)
        cv2.imwrite(filepath, overlay)

        import base64
        ok, buf = cv2.imencode(".png", overlay)
        b64 = base64.b64encode(buf).decode() if ok else ""

        return "/" + filepath.replace(os.sep, "/"), b64

    except Exception as exc:
        logger.error("XAI heatmap generation failed: %s", exc)
        return "", ""


def _add_legend(img: np.ndarray, features: Dict, region_count: int) -> None:
    """Add a minimal info legend."""
    h = img.shape[0]
    area = features.get("hotspot_area_percent", 0)
    contrast = features.get("thermal_contrast", 0)

    lines = [
        f"Anomaly area: {area:.1f}%",
        f"Regions: {region_count}",
        f"Thermal contrast: {contrast:.1f}",
    ]

    box_h = 16 + len(lines) * 18
    overlay = img.copy()
    cv2.rectangle(overlay, (6, h - box_h - 6), (210, h - 6), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, h - box_h + 14 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


def build_explanation_points(features: Dict, fault_type: str, severity: str) -> List[str]:
    """Build human-readable explanation points grounded in computed values."""
    points = []
    area = features.get("hotspot_area_percent", 0)
    regions = features.get("region_count", 0)
    contrast = features.get("thermal_contrast", 0)

    if fault_type == "normal":
        points.append("No significant thermal anomalies detected above the classification threshold.")
        if area > 0:
            points.append(f"Minor warm areas detected ({area:.1f}% of panel) — within normal operating range.")
        return points

    # Area
    if area >= 12:
        points.append(f"Significant thermal anomaly covering {area:.1f}% of the panel area.")
    elif area >= 5:
        points.append(f"Moderate hotspot coverage at {area:.1f}% of panel area.")
    else:
        points.append(f"Localised thermal anomaly covering {area:.1f}% of panel area.")

    # Regions
    if regions == 1:
        points.append("Single concentrated anomaly region identified.")
    elif regions <= 3:
        points.append(f"{regions} distinct anomaly regions detected — suggests localised fault.")
    else:
        points.append(f"{regions} anomaly regions detected — distributed thermal pattern.")

    # Contrast
    if contrast > 25:
        points.append(f"Strong thermal contrast ({contrast:.1f}) — significant temperature differential above background.")
    elif contrast > 12:
        points.append(f"Moderate thermal contrast ({contrast:.1f}) between anomaly and background.")
    else:
        points.append(f"Low thermal contrast ({contrast:.1f}) — anomaly is mild relative to background.")

    # Severity context
    if severity == "high":
        points.append("Combined metrics indicate elevated concern — prompt inspection warranted.")
    elif severity == "medium":
        points.append("Metrics suggest moderate concern — inspection recommended within standard timeframe.")
    else:
        points.append("Metrics indicate low severity — routine monitoring recommended.")

    return points


def get_top_reason(features: Dict, fault_type: str, severity: str) -> str:
    """Return the single most important reason for the classification."""
    area = features.get("hotspot_area_percent", 0)
    regions = features.get("region_count", 0)
    contrast = features.get("thermal_contrast", 0)

    if fault_type == "normal":
        return "No thermal anomalies detected above the classification threshold."

    if area >= 12:
        return f"Significant thermal anomaly covering {area:.1f}% of panel area across {regions} region(s)."
    if contrast > 25:
        return f"Strong thermal contrast ({contrast:.1f}) with anomaly area at {area:.1f}% across {regions} region(s)."
    return f"Thermal anomaly detected: {area:.1f}% area, {regions} region(s), contrast {contrast:.1f}."


def generate_xai_output(img_bgr: np.ndarray, features: Dict, fault_type: str, severity: str) -> Dict:
    """Full XAI output generation."""
    visual_path, xai_b64 = generate_xai_heatmap(img_bgr, features)
    explanation_points = build_explanation_points(features, fault_type, severity)
    top_reason = get_top_reason(features, fault_type, severity)

    return {
        "top_reason": top_reason,
        "explanation_points": explanation_points,
        "visual_path": visual_path,
        "xai_image_base64": xai_b64,
    }
