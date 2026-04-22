"""
Image quality assessment for thermal images.

Checks whether an uploaded image is:
  - Readable and has valid dimensions
  - A genuine thermal image (has thermal colormap characteristics)
  - Sharp enough to extract meaningful features
  - Has sufficient dynamic range for analysis

Returns a structured quality report used by the pipeline to:
  - Reject clearly invalid images early
  - Attach quality warnings to results
  - Reduce confidence when quality is marginal
"""

import cv2
import numpy as np
from typing import Dict


# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_DIMENSION = 64          # px — smaller images are not useful
MIN_LAPLACIAN = 1.5         # below this = severely blurry / synthetic
MIN_SCORE_RANGE = 30.0      # thermal score (R-B) range — below = not a thermal image
MIN_GRAY_STD = 4.0          # below = nearly uniform / blank image
MIN_DYNAMIC_RANGE = 20.0    # gray std — below = very low contrast


def assess_image_quality(img_bgr: np.ndarray) -> Dict:
    """
    Assess the quality of a thermal image.

    Returns a dict with:
        is_valid (bool)         — False = reject entirely
        quality_level (str)     — 'good', 'marginal', 'poor'
        warnings (list[str])    — human-readable warnings
        confidence_penalty (float) — 0.0-0.4 subtracted from classification confidence
        metrics (dict)          — raw quality metrics for debugging
    """
    warnings = []
    confidence_penalty = 0.0

    if img_bgr is None:
        return {
            "is_valid": False,
            "quality_level": "poor",
            "warnings": ["Image could not be read or is corrupted."],
            "confidence_penalty": 1.0,
            "metrics": {},
        }

    h, w = img_bgr.shape[:2]

    # ── Dimension check ───────────────────────────────────────────────────────
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        return {
            "is_valid": False,
            "quality_level": "poor",
            "warnings": [f"Image too small ({w}×{h}px). Minimum is {MIN_DIMENSION}×{MIN_DIMENSION}px."],
            "confidence_penalty": 1.0,
            "metrics": {"width": w, "height": h},
        }

    # ── Compute quality metrics ───────────────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_bgr)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gray_std = float(gray.std())
    gray_mean = float(gray.mean())

    # Thermal score: R - B captures JET colormap hot-to-cold axis
    thermal_score = r.astype(float) - b.astype(float)
    score_range = float(thermal_score.max() - thermal_score.min())
    score_std = float(thermal_score.std())

    metrics = {
        "width": w,
        "height": h,
        "laplacian_var": round(laplacian_var, 2),
        "gray_std": round(gray_std, 2),
        "gray_mean": round(gray_mean, 2),
        "thermal_score_range": round(score_range, 2),
        "thermal_score_std": round(score_std, 2),
    }

    # ── Check: is this a thermal image at all? ────────────────────────────────
    is_thermal = score_range >= MIN_SCORE_RANGE and gray_std >= MIN_GRAY_STD

    if not is_thermal:
        if score_range < MIN_SCORE_RANGE:
            warnings.append(
                f"Image does not appear to be a thermal image — "
                f"insufficient color variation (thermal score range: {score_range:.0f}, expected ≥{MIN_SCORE_RANGE:.0f}). "
                "Results may be unreliable."
            )
            confidence_penalty += 0.35
        if gray_std < MIN_GRAY_STD:
            warnings.append(
                f"Image has very low contrast (std: {gray_std:.1f}). "
                "It may be blank, overexposed, or not a thermal image."
            )
            confidence_penalty += 0.25

    # ── Check: sharpness ─────────────────────────────────────────────────────
    if laplacian_var < MIN_LAPLACIAN:
        warnings.append(
            f"Image is severely blurry or low-resolution (sharpness score: {laplacian_var:.2f}). "
            "Hotspot boundaries may not be accurately detected."
        )
        confidence_penalty += 0.20
    elif laplacian_var < 5.0:
        warnings.append(
            f"Image has low sharpness (score: {laplacian_var:.1f}). "
            "Detection accuracy may be reduced."
        )
        confidence_penalty += 0.10

    # ── Check: dynamic range ─────────────────────────────────────────────────
    if gray_std < MIN_DYNAMIC_RANGE and gray_std >= MIN_GRAY_STD:
        warnings.append(
            f"Low thermal dynamic range (contrast: {gray_std:.1f}). "
            "Temperature differences between regions are small — minor anomalies may be missed."
        )
        confidence_penalty += 0.10

    # ── Determine overall quality level ──────────────────────────────────────
    confidence_penalty = min(confidence_penalty, 0.50)

    if confidence_penalty == 0.0:
        quality_level = "good"
    elif confidence_penalty <= 0.15:
        quality_level = "marginal"
    else:
        quality_level = "poor"

    # Images are only rejected if they are completely non-thermal or blank
    # Blurry/marginal images are processed but with warnings
    is_valid = score_range >= MIN_SCORE_RANGE or gray_std >= MIN_GRAY_STD * 2

    return {
        "is_valid": is_valid,
        "quality_level": quality_level,
        "warnings": warnings,
        "confidence_penalty": round(confidence_penalty, 2),
        "metrics": metrics,
    }
