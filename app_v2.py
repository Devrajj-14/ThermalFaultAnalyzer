"""
ThermalFaultAnalyzer Flask Application (Production-Ready Refactor)

Multi-class solar panel thermal fault analysis system with:
- Multi-class classification (normal, hotspot, severe_thermal_anomaly)
- Severity classification (low, medium, high)
- Explainable AI (XAI) with visual heatmaps
- GenAI-generated grounded summaries
- Timeline recommendations
- Robust fallback behavior
"""

import os
import logging
import cv2
from flask import Flask, request, jsonify, send_from_directory

from config.settings import TEMP_UPLOAD_DIR, XAI_OUTPUT_DIR
from utils.validation import validate_upload, sanitize_filename
from utils.image_utils import encode_image, crop_fault_regions, safe_read_image
from utils.response_formatter import build_success_response, build_error_response, compute_score
from services.inference_service import run_inference, compute_performance_metrics, derive_panel_health
from services.xai_service import generate_xai_output
from services.timeline_service import get_timeline
from services.genai_service import generate_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Ensure required directories exist
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)


@app.route("/")
def home():
    """Serve the main dashboard."""
    return app.send_static_file("index.html")


@app.route("/static/generated/<path:subpath>/<filename>")
def serve_generated(subpath, filename):
    """Serve generated files (XAI images, etc.)."""
    directory = os.path.join("static", "generated", subpath)
    return send_from_directory(directory, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    
    Accepts: multipart/form-data with 'image' file
    Returns: JSON with complete analysis results
    """
    # ── Step 1: Validate upload ────────────────────────────────────────────────
    if "image" not in request.files:
        return build_error_response("No image uploaded", 400)
    
    image_file = request.files["image"]
    is_valid, error_msg = validate_upload(image_file)
    if not is_valid:
        return build_error_response(error_msg, 400)
    
    # ── Step 2: Save uploaded file safely ──────────────────────────────────────
    filename = sanitize_filename(image_file.filename)
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"temp_{filename}")
    
    try:
        image_file.save(temp_path)
        logger.info("Image uploaded: %s", filename)
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        return build_error_response("Failed to save uploaded file", 500)
    
    try:
        # ── Step 3: Read and validate image ────────────────────────────────────
        img_bgr = safe_read_image(temp_path)
        if img_bgr is None:
            raise ValueError("Could not read image. File may be corrupted or invalid format.")
        
        # ── Step 4: Run inference ───────────────────────────────────────────────
        inference_result = run_inference(img_bgr)
        
        fault_type = inference_result["fault_type"]
        severity = inference_result["severity"]
        confidence = inference_result["confidence"]
        features = inference_result["features"]
        inference_mode = inference_result.get("inference_mode", "rule_based")
        quality_report = inference_result.get("quality_report", {"warnings": [], "quality_level": "good", "confidence_penalty": 0.0})
        is_ambiguous = inference_result.get("is_ambiguous", False)
        
        logger.info(
            "Inference complete: %s | %s | %.2f | mode=%s",
            fault_type, severity, confidence, inference_mode
        )
        
        # ── Step 5: Derive panel health ─────────────────────────────────────────
        panel_health = derive_panel_health(fault_type, severity, confidence)
        
        # ── Step 6: Compute performance metrics ─────────────────────────────────
        metrics = compute_performance_metrics(features, fault_type, severity)
        
        # ── Step 7: Generate XAI output ─────────────────────────────────────────
        xai_output = generate_xai_output(img_bgr, features, fault_type, severity)
        
        # ── Step 8: Get timeline recommendations ────────────────────────────────
        timeline = get_timeline(fault_type, severity)
        
        # ── Step 9: Compute composite score ─────────────────────────────────────
        score = compute_score(features, fault_type, severity)
        
        # ── Step 10: Prepare GenAI payload ──────────────────────────────────────
        # All values come from the single metrics dict — single source of truth.
        genai_payload = {
            "fault_type": fault_type,
            "severity": severity,
            "confidence": confidence,
            "panel_health": panel_health,
            "action_timeline": timeline["action_timeline"],
            "risk_timeline": timeline["risk_timeline"],
            "hotspot_area_percent": metrics["hotspot_area_percent"],
            "region_count": metrics["region_count"],
            "thermal_contrast": metrics["thermal_contrast"],
            "estimated_temp_delta": metrics["estimated_temp_delta"],
            "power_drop_estimate": metrics["power_drop_estimate"],
        }
        
        # ── Step 11: Generate GenAI summary ─────────────────────────────────────
        genai_summary = generate_summary(genai_payload)
        
        # ── Step 12: Encode images ──────────────────────────────────────────────
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(features["detection_img"])
        fault_parts = crop_fault_regions(img_bgr, features.get("contours", []))

        # Build ranked_regions metadata for the UI (area%, importance, contrast, bbox)
        ranked_meta = []
        total_px = img_bgr.shape[0] * img_bgr.shape[1]
        for r in features.get("ranked_regions", []):
            x, y, bw, bh = cv2.boundingRect(r["contour"])
            ranked_meta.append({
                "area_pct": round(r["area"] / total_px * 100, 2),
                "importance": round(r["importance"], 3),
                "contrast": round(r["contrast"], 3),
                "x": int(x), "y": int(y), "bw": int(bw), "bh": int(bh),
            })

        images = {
            "input_image": input_encoded,
            "fault_detection": detection_encoded,
            "fault_parts": fault_parts,
        }
        
        # ── Step 13: Build response ─────────────────────────────────────────────
        response = build_success_response(
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            panel_health=panel_health,
            score=score,
            timeline=timeline,
            metrics=metrics,
            xai=xai_output,
            genai=genai_summary,
            images=images,
            inference_mode=inference_mode,
            ranked_regions=ranked_meta,
            quality_report=quality_report,
            is_ambiguous=is_ambiguous,
        )
        
        logger.info("Prediction successful: %s", fault_type)
        return jsonify(response)
    
    except Exception as exc:
        logger.exception("Prediction failed")
        return build_error_response(f"Prediction failed: {str(exc)}", 500)
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as exc:
                logger.warning("Failed to remove temp file: %s", exc)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "ThermalFaultAnalyzer"})


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return build_error_response("File too large. Maximum size is 10 MB.", 413)


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error("Internal server error: %s", error)
    return build_error_response("Internal server error", 500)


if __name__ == "__main__":
    logger.info("Starting ThermalFaultAnalyzer Flask app...")
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
