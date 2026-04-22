import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="static")

def encode_image(img):
    """Encode a CV2 BGR image as a base64 string in PNG format."""
    if img is None:
        return ""
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    temp_path = os.path.join("data", f"temp_{image_file.filename}")
    image_file.save(temp_path)

    try:
        # ----- Image Processing & Fault Detection -----
        img_bgr = cv2.imread(temp_path)
        if img_bgr is None:
            raise ValueError("Could not read image. Is it valid?")

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect hot regions
        thr_value = 200
        _, thresh = cv2.threshold(img_gray, thr_value, 255, cv2.THRESH_BINARY)
        
        # Use morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours representing fault areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out tiny contours (noise) — only keep areas > 0.1% of image size
        min_area = 0.001 * img_gray.shape[0] * img_gray.shape[1]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        detection_img = img_bgr.copy()
        fault_parts = []
        fault_temperatures = []  # list for average intensity in each fault region
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw a bounding box around each fault
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Crop the fault region
            crop_img = img_bgr[y:y+h, x:x+w]
            fault_parts.append(encode_image(crop_img))
            # Compute the average intensity in the fault region for dynamic measurement.
            local_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            local_avg = np.mean(local_gray)
            fault_temperatures.append(local_avg)
        
        # Compute confidence as ratio of hot pixels in the entire image.
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        hot_pixels = np.sum(thresh == 255)
        confidence = (hot_pixels / total_pixels) * 100.0
        
        prediction_text = "Hotspot Detected" if confidence >= 5 else "No Fault (Normal)"
        
        # ---------------------------------------------------------------------------------
        # Derive a dynamic "actual_temp" based on fault areas rather than the full image.
        # If fault regions are found, take the maximum average intensity among them.
        if fault_temperatures:
            max_fault_intensity = max(fault_temperatures)
        else:
            # If no faults are detected, fallback to the entire image average.
            max_fault_intensity = np.mean(img_gray)
            
        # Map intensity to realistic panel temperature range: 25°C (min) to 85°C (max)
        min_panel_temp = 25.0
        max_panel_temp = 85.0
        actual_temp = min_panel_temp + (max_fault_intensity / 255.0) * (max_panel_temp - min_panel_temp)
        
        # ---------------------------------------------------------------------------------
        # Now modify the performance metrics to depend on the derived temperature.
        # For example, assume that the expected output decreases when temperature exceeds 25°C.
        base_expected_output = 1684.88   # baseline expected output (kWh)
        if actual_temp > 25:
            temp_diff = actual_temp - 25
            dynamic_expected = base_expected_output - (temp_diff * 5)  # lose 5 kWh per degree above 25
        else:
            dynamic_expected = base_expected_output

        # actual_output scales with fault severity: more faults = lower output
        # confidence represents % of hot pixels; scale output between 95% and 100% of expected
        fault_factor = 1.0 - (confidence / 100.0) * 0.15  # max 15% reduction at 100% confidence
        actual_output = dynamic_expected * fault_factor
        
        # ----- Dynamic Metrics Calculation -----
        power_drop = ((dynamic_expected - actual_output) / dynamic_expected) * 100.0

        # Performance Ratio (PR)
        # PR = actual_output / (installed_power_kWp * irradiance_kWh/m2/day * days)
        # Units must be consistent: actual_output in kWh, theoretical_max in kWh
        # For a 1 kWp system over 1 day with 5.5 kWh/m²/day irradiance:
        installed_power = 1  # kWp (matching the scale of actual_output ~1420-1500 kWh/year per kWp)
        irradiance = 5.5     # kWh/m²/day (peak sun hours)
        days = 365
        theoretical_max = installed_power * irradiance * days  # ~2007.5 kWh/year per kWp
        pr = (actual_output / theoretical_max) * 100.0

        # Annual Degradation Rate over a 5-year period
        years = 5
        degradation_rate = ((dynamic_expected - actual_output) / (dynamic_expected * years)) * 100.0

        # Temperature Loss/Gain Calculation using the derived temperature
        stc_temp = 25          # Standard Test Condition temperature (°C)
        rated_power = 1000     # W (1 kWp panel)
        temp_coeff = -0.0045   # -0.45%/°C typical for crystalline silicon
        temp_delta = actual_temp - stc_temp
        temperature_loss = temp_coeff * temp_delta * rated_power  # positive = loss, negative = gain
        temp_loss_label = "Temperature Gain" if temperature_loss < 0 else "Temperature Loss"
        # ---------------------------------------------------------------------------------

        # Build the analysis dictionary
        analysis = {
            "power_output_drop": {
                "observed_value": round(power_drop, 2),
                "standard_threshold": "<10%",
                "status": "🔴" if power_drop > 10 else "✅",
                "inference": ("Significant power drop. Inspection recommended."
                              if power_drop > 10 else "Normal")
            },
            "performance_ratio": {
                "observed_value": round(pr, 2),
                "standard_threshold": "≥75%",
                "status": "🔴" if pr < 75 else "✅",
                "inference": ("PR < 75%. Substantial inefficiencies."
                              if pr < 75 else "Normal")
            },
            "degradation_rate": {
                "observed_value": round(degradation_rate, 2),
                "standard_threshold": "<0.7%/year",
                "status": "🔴" if degradation_rate > 0.7 else "✅",
                "inference": ("Degradation is high. Consider replacement or remediation."
                              if degradation_rate > 0.7 else "Normal")
            },
            "temperature_loss": {
                "observed_value": round(abs(temperature_loss), 2),
                "inference": f"{temp_loss_label}: {round(abs(temperature_loss), 2)} W at {round(actual_temp, 2)}°C panel temp."
            },
            # For debugging or display purposes:
            "derived_temperature": {
                "value": round(actual_temp, 2),
                "units": "°C"
            }
        }

        # Encode the original and detection images
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(detection_img)
        
        # Clean up temporary file
        os.remove(temp_path)

        # Return complete response with images, prediction, and dynamic analysis metrics
        return jsonify({
            "prediction": prediction_text,
            "confidence": round(confidence, 2),
            "input_image": input_encoded,
            "fault_detection": detection_encoded,
            "fault_parts": fault_parts,
            "analysis": analysis
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
