"""
Rule-based thermal fault classification service.

KEY INSIGHT: Thermal images in this dataset are color-mapped (JET/similar colormap).
Hot regions appear RED/ORANGE, cold regions appear BLUE/PURPLE.
Converting to grayscale destroys this information because red and blue
produce similar luminance values (~0.3R + 0.59G + 0.11B).

Detection strategy:
  1. Compute a "thermal score" per pixel using color channel relationships
  2. Use percentile-based adaptive thresholding on this thermal score
  3. Score each candidate region by: area, peak thermal score, contrast,
     compactness, and distance from image borders
  4. Penalise thin edge-hugging fragments
  5. Keep top meaningful regions only
  6. Compute a spatial concentration score to distinguish genuine hotspots
     from normal thermal gradients (the key discriminator for this dataset)

IMPORTANT LIMITATION:
  The C (clean), H (hotspot), and S (severe) classes in this dataset have
  heavily overlapping feature distributions. Rule-based classification cannot
  reliably separate them with high confidence. When features fall in the
  ambiguous zone, confidence is reduced and a quality warning is attached.
  A trained ML model is required for reliable classification.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from config.settings import (
    MIN_PANEL_TEMP_C, MAX_PANEL_TEMP_C,
    HOTSPOT_AREA_LOW, HOTSPOT_AREA_MED, HOTSPOT_AREA_HIGH,
    CONTRAST_LOW, CONTRAST_HIGH,
    REGION_COUNT_MED, REGION_COUNT_HIGH,
    SEVERITY_AREA_WEIGHT, SEVERITY_CONTRAST_WEIGHT, SEVERITY_REGION_WEIGHT,
)


# ---------------------------------------------------------------------------
# Spatial concentration score
# ---------------------------------------------------------------------------

def _compute_concentration_score(thermal_score: np.ndarray) -> Dict:
    """
    Compute spatial concentration metrics that help distinguish genuine
    hotspots from normal thermal gradients.

    A genuine hotspot is:
      - Spatially concentrated (small area, high peak)
      - Isolated (surrounded by cooler background)
      - Has a sharp peak relative to the surrounding area

    A normal thermal gradient is:
      - Diffuse (large area, moderate temperature)
      - Not isolated (smoothly transitions to background)

    Returns dict with:
      concentration (float 0-1): how concentrated the hottest pixels are
      isolation (float 0-1): how isolated the peak is from surrounding area
      peak_sharpness (float): ratio of peak to local neighborhood mean
      is_concentrated (bool): True if pattern looks like a genuine hotspot
    """
    h, w = thermal_score.shape

    # Top 1% pixels
    p99 = np.percentile(thermal_score, 99)
    p80 = np.percentile(thermal_score, 80)
    p50 = np.percentile(thermal_score, 50)

    top1_area = float(np.sum(thermal_score >= p99)) / (h * w)
    top20_area = float(np.sum(thermal_score >= p80)) / (h * w)

    # Concentration: how much of the "hot energy" is in the top 1% of pixels
    top1_energy = float(np.sum(thermal_score[thermal_score >= p99]))
    total_energy = float(np.sum(thermal_score[thermal_score >= p50]))
    concentration = top1_energy / max(total_energy, 1e-6)

    # Peak sharpness: peak value vs mean of surrounding 10% pixels
    p90 = np.percentile(thermal_score, 90)
    peak_val = float(thermal_score.max())
    surround_mean = float(np.mean(thermal_score[(thermal_score >= p80) & (thermal_score < p90)]))
    peak_sharpness = peak_val / max(surround_mean, 1e-6)

    # Isolation: difference between top-1% mean and top-20% mean (normalised)
    top1_mean = float(np.mean(thermal_score[thermal_score >= p99])) if np.any(thermal_score >= p99) else 0
    top20_mean = float(np.mean(thermal_score[thermal_score >= p80])) if np.any(thermal_score >= p80) else 0
    score_range = float(thermal_score.max() - thermal_score.min())
    isolation = (top1_mean - top20_mean) / max(score_range, 1e-6)

    # A concentrated hotspot: concentration > 0.08, isolation > 0.05, sharpness > 1.3
    is_concentrated = (
        concentration > 0.08 and
        isolation > 0.05 and
        peak_sharpness > 1.25
    )

    return {
        "concentration": round(concentration, 4),
        "isolation": round(isolation, 4),
        "peak_sharpness": round(peak_sharpness, 4),
        "is_concentrated": is_concentrated,
        "top1_area_pct": round(top1_area * 100, 3),
    }


# ---------------------------------------------------------------------------
# Thermal score map
# ---------------------------------------------------------------------------

def _compute_thermal_score(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a color-mapped thermal image into a float32 thermal score map
    where higher values = hotter.

    JET colormap order (cold→hot):
        blue (0,0,255) → cyan (0,255,255) → green (0,255,0)
        → yellow (255,255,0) → orange (255,165,0) → red (255,0,0) → white

    We approximate this with:
        score = R - B  (positive = warm, negative = cold)
    then add a boost for orange/yellow (high R + moderate G + low B)
    and for near-white (all channels high).

    Returns float32 array in range roughly [-255, 510], normalised to [0, 1].
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)

    # Primary: red minus blue captures the JET hot-to-cold axis
    rb_diff = r - b  # range [-255, 255]

    # Boost for orange/yellow (high R, moderate-high G, low B)
    orange_boost = np.clip((r - 120) / 135.0, 0, 1) * np.clip((g - 60) / 120.0, 0, 1) * np.clip((120 - b) / 120.0, 0, 1)

    # Boost for near-white (all channels high → very hot in JET)
    white_boost = np.clip((r - 200) / 55.0, 0, 1) * np.clip((g - 200) / 55.0, 0, 1) * np.clip((b - 200) / 55.0, 0, 1)

    # Combine: rb_diff is the backbone, boosts add up to ~1.0 extra
    score = rb_diff + orange_boost * 80.0 + white_boost * 100.0

    # Normalise to [0, 1]
    s_min, s_max = score.min(), score.max()
    if s_max - s_min < 1e-6:
        return np.zeros_like(score)
    return (score - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Region scoring
# ---------------------------------------------------------------------------

def _score_region(cnt: np.ndarray, thermal_score: np.ndarray, img_shape: Tuple) -> float:
    """
    Score a candidate region by multiple factors.

    Factors (all normalised 0-1, weighted sum):
      - peak_score:    max thermal score in region (0.30)
      - mean_score:    mean thermal score in region (0.25)
      - area_score:    log-normalised area (0.20)
      - compactness:   4π·area/perimeter² (0.10)
      - interior_score: penalise regions hugging image borders (0.15)

    Returns float score (higher = more important anomaly).
    """
    h, w = img_shape[:2]
    total = h * w

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    region_vals = thermal_score[mask == 255]
    if len(region_vals) == 0:
        return 0.0

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    peak = float(np.max(region_vals))
    mean_v = float(np.mean(region_vals))

    # Area score: log scale, 0.5% of image = 0.5, 5% = 1.0
    area_pct = area / total
    area_score = float(np.clip(np.log10(max(area_pct * 100, 0.01) + 1) / np.log10(6), 0, 1))

    # Compactness (circle = 1.0, thin strip → 0)
    if perimeter > 0:
        compactness = float(np.clip(4 * np.pi * area / (perimeter ** 2), 0, 1))
    else:
        compactness = 0.0

    # Interior score: penalise regions whose bounding box touches the image border
    x, y, bw, bh = cv2.boundingRect(cnt)
    border_margin = 5  # pixels
    touches_border = (x <= border_margin or y <= border_margin or
                      x + bw >= w - border_margin or y + bh >= h - border_margin)
    # If it touches the border AND is thin (aspect ratio extreme), penalise heavily
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if touches_border and aspect > 4:
        interior_score = 0.1  # strong penalty for thin border strips
    elif touches_border:
        interior_score = 0.6  # mild penalty for border-touching but compact
    else:
        interior_score = 1.0

    score = (
        peak * 0.30 +
        mean_v * 0.25 +
        area_score * 0.20 +
        compactness * 0.10 +
        interior_score * 0.15
    )
    return score


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_thermal_features(img_bgr: np.ndarray) -> Dict:
    """
    Extract thermal anomaly features from a color-mapped thermal image.

    Pipeline:
      1. Compute per-pixel thermal score from color channels
      2. Adaptive threshold: top N% of thermal scores
      3. Morphological cleanup
      4. Filter candidates by minimum area and minimum thermal score
      5. Score each region with multi-factor scoring
      6. Merge nearby regions
      7. Keep top 5 by score, discard junk
      8. Compute aggregate features

    Returns dict with all features needed by downstream services.
    """
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Step 1: thermal score map
    thermal_score = _compute_thermal_score(img_bgr)

    # Step 1b: spatial concentration metrics
    concentration_info = _compute_concentration_score(thermal_score)

    # Step 2: adaptive threshold
    # Use the top 8% of thermal scores as the hotspot candidate zone.
    # This is robust: if the image has a strong hotspot it will be in the top 8%;
    # if the image is uniformly warm the top 8% will still be the warmest part
    # but the contrast filter below will reject it.
    threshold_pct = np.percentile(thermal_score, 92)

    # Safety: if the image has very low dynamic range (nearly uniform),
    # raise the bar so we don't flag everything
    score_range = thermal_score.max() - thermal_score.min()
    if score_range < 0.15:
        # Very uniform image — require top 2% only
        threshold_pct = np.percentile(thermal_score, 98)

    thresh_mask = (thermal_score >= threshold_pct).astype(np.uint8) * 255

    # Step 3: morphological cleanup — close small gaps, remove tiny noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Step 4: find contours
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: filter by minimum area and minimum mean thermal score
    min_area = 0.002 * total_pixels  # 0.2% minimum
    # Contrast threshold: region mean must be meaningfully above image mean
    global_mean_score = float(thermal_score.mean())
    min_contrast = 0.10  # region mean must be at least 0.10 above global mean (on 0-1 scale)

    candidate_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        region_vals = thermal_score[mask == 255]
        if len(region_vals) == 0:
            continue

        region_mean = float(np.mean(region_vals))
        contrast = region_mean - global_mean_score

        if contrast < min_contrast:
            continue

        importance = _score_region(cnt, thermal_score, img_bgr.shape)
        candidate_regions.append({
            "contour": cnt,
            "area": area,
            "region_mean": region_mean,
            "region_max": float(np.max(region_vals)),
            "contrast": contrast,
            "importance": importance,
        })

    # Step 6: merge nearby regions (dilate + re-find)
    if len(candidate_regions) > 1:
        merge_mask = np.zeros((h, w), dtype=np.uint8)
        for r in candidate_regions:
            cv2.drawContours(merge_mask, [r["contour"]], -1, 255, -1)
        k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        merge_mask = cv2.dilate(merge_mask, k_merge, iterations=1)
        merged_cnts, _ = cv2.findContours(merge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Re-score merged contours
        candidate_regions = []
        for cnt in merged_cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            region_vals = thermal_score[mask == 255]
            if len(region_vals) == 0:
                continue
            region_mean = float(np.mean(region_vals))
            contrast = region_mean - global_mean_score
            if contrast < min_contrast:
                continue
            importance = _score_region(cnt, thermal_score, img_bgr.shape)
            candidate_regions.append({
                "contour": cnt,
                "area": area,
                "region_mean": region_mean,
                "region_max": float(np.max(region_vals)),
                "contrast": contrast,
                "importance": importance,
            })

    # Step 7: rank by importance, keep top 5
    candidate_regions.sort(key=lambda r: r["importance"], reverse=True)
    top_regions = candidate_regions[:5]

    # Step 8: aggregate features
    hotspot_mask = np.zeros((h, w), dtype=np.uint8)
    for r in top_regions:
        cv2.drawContours(hotspot_mask, [r["contour"]], -1, 255, -1)

    hot_pixels = int(np.sum(hotspot_mask == 255))
    hotspot_area_percent = (hot_pixels / total_pixels) * 100.0
    region_count = len(top_regions)

    # Thermal contrast: difference between hotspot mean score and background mean score
    if hot_pixels > 0:
        hotspot_scores = thermal_score[hotspot_mask == 255]
        background_scores = thermal_score[hotspot_mask == 0]
        hotspot_mean_score = float(np.mean(hotspot_scores))
        background_mean_score = float(np.mean(background_scores))
        # Scale contrast to a 0-100 range for downstream compatibility
        thermal_contrast = float((hotspot_mean_score - background_mean_score) * 100.0)
        peak_score = float(np.max(hotspot_scores))
    else:
        thermal_contrast = 0.0
        peak_score = 0.0

    # Estimated temperature (proxy from peak thermal score)
    actual_temp = MIN_PANEL_TEMP_C + peak_score * (MAX_PANEL_TEMP_C - MIN_PANEL_TEMP_C)
    estimated_temp_delta = actual_temp - MIN_PANEL_TEMP_C

    # Build detection image with ranked bounding boxes
    detection_img = img_bgr.copy()
    for i, r in enumerate(top_regions):
        x, y, bw, bh = cv2.boundingRect(r["contour"])
        # Color by rank: #1 = red, #2 = orange, rest = yellow
        color = [(0, 0, 255), (0, 128, 255), (0, 200, 255), (0, 255, 255), (0, 255, 200)][i]
        cv2.rectangle(detection_img, (x, y), (x + bw, y + bh), color, 2)

    return {
        "hotspot_area_percent": round(hotspot_area_percent, 2),
        "region_count": region_count,
        "thermal_contrast": round(thermal_contrast, 2),
        "estimated_temp_delta": round(estimated_temp_delta, 2),
        "max_intensity": round(peak_score * 255, 1),  # kept for API compat
        "contours": [r["contour"] for r in top_regions],
        "ranked_regions": top_regions,
        "thresh": hotspot_mask,
        "detection_img": detection_img,
        "actual_temp": round(actual_temp, 2),
        "thermal_score_map": thermal_score,
        "adaptive_threshold": round(float(threshold_pct), 3),
        # Spatial concentration metrics
        "concentration": concentration_info["concentration"],
        "isolation": concentration_info["isolation"],
        "peak_sharpness": concentration_info["peak_sharpness"],
        "is_concentrated": concentration_info["is_concentrated"],
        "top1_area_pct": concentration_info["top1_area_pct"],
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_fault_type(features: Dict) -> Tuple[str, float]:
    """
    Classify fault type from extracted features.

    Uses both region-level features (area, contrast, count) AND
    spatial concentration metrics to distinguish genuine hotspots
    from normal thermal gradients.

    HONEST CONFIDENCE: When features fall in the ambiguous zone where
    C/H/S classes overlap, confidence is capped at 0.65 to reflect
    the genuine uncertainty of rule-based classification on this dataset.

    Returns (fault_type, confidence).
    """
    area = features["hotspot_area_percent"]
    contrast = features["thermal_contrast"]
    region_count = features["region_count"]
    is_concentrated = features.get("is_concentrated", False)
    concentration = features.get("concentration", 0.0)
    isolation = features.get("isolation", 0.0)
    peak_sharpness = features.get("peak_sharpness", 1.0)

    # ── Normal: no meaningful anomaly ────────────────────────────────────────
    if region_count == 0:
        return "normal", 0.92

    if area < 1.5 and contrast < 8.0:
        return "normal", 0.85

    # ── Severe: requires strong multi-factor evidence ─────────────────────────
    # High area + high contrast + multiple regions + concentrated pattern
    if area >= 15.0 and contrast >= 25.0 and region_count >= 3 and is_concentrated:
        confidence = min(0.75 + (area / 100.0) * 0.12, 0.88)
        return "severe_thermal_anomaly", confidence

    if area >= 25.0 and contrast >= 20.0 and is_concentrated:
        return "severe_thermal_anomaly", 0.82

    # ── Hotspot: concentrated anomaly present ────────────────────────────────
    if is_concentrated and contrast >= 15.0:
        base = 0.68
        area_boost = min((area / 15.0) * 0.12, 0.15)
        contrast_boost = min((contrast / 40.0) * 0.08, 0.08)
        concentration_boost = min(concentration * 0.5, 0.05)
        confidence = min(base + area_boost + contrast_boost + concentration_boost, 0.88)
        return "hotspot", confidence

    # ── Ambiguous zone: features present but not concentrated ─────────────────
    # This is where C/H/S classes heavily overlap.
    # We still report the most likely class but cap confidence.
    if area >= 5.0 and contrast >= 20.0:
        # Likely hotspot but not definitively concentrated
        confidence = min(0.55 + (concentration * 0.3) + (isolation * 0.2), 0.65)
        return "hotspot", confidence

    if area >= 2.0 and contrast >= 10.0:
        confidence = min(0.50 + (concentration * 0.2), 0.60)
        return "hotspot", confidence

    # ── Default: normal with low confidence ──────────────────────────────────
    return "normal", 0.65


def classify_severity(features: Dict, fault_type: str) -> str:
    """
    Classify severity: low / medium / high.

    Severity score (0-100) uses:
      - area (45%): normalised to 20% area = max
      - contrast (35%): normalised to 35 contrast units = max
      - region count (20%): normalised to 5 regions = max
    """
    if fault_type == "normal":
        return "low"

    area = features["hotspot_area_percent"]
    contrast = features["thermal_contrast"]
    region_count = features["region_count"]

    area_score = min((area / 20.0) * 100, 100)
    contrast_score = min((contrast / 35.0) * 100, 100)
    region_score = min((region_count / 5.0) * 100, 100)

    severity_score = (
        area_score * 0.45 +
        contrast_score * 0.35 +
        region_score * 0.20
    )

    if severity_score < 30:
        return "low"
    elif severity_score < 65:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rule_based_inference(img_bgr: np.ndarray, quality_penalty: float = 0.0) -> Dict:
    """
    Complete rule-based inference pipeline.

    Args:
        img_bgr: OpenCV BGR image
        quality_penalty: confidence reduction from image quality assessment (0.0-0.5)

    Returns dict with fault_type, severity, confidence, features, and
    is_ambiguous flag indicating when the result has low certainty.
    """
    features = extract_thermal_features(img_bgr)
    fault_type, confidence = classify_fault_type(features)
    severity = classify_severity(features, fault_type)

    # Apply quality penalty to confidence
    final_confidence = max(round(confidence - quality_penalty, 2), 0.30)

    # Flag ambiguous results so the UI can show appropriate caveats
    is_ambiguous = (
        final_confidence < 0.65 or
        not features.get("is_concentrated", False) and fault_type != "normal"
    )

    return {
        "fault_type": fault_type,
        "severity": severity,
        "confidence": final_confidence,
        "features": features,
        "is_ambiguous": is_ambiguous,
    }
