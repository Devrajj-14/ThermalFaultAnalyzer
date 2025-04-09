import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="static")

def encode_image(img):
    """Encode a CV2 BGR image as base64 string in PNG format."""
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
        # Read the image
        img_bgr = cv2.imread(temp_path)
        if img_bgr is None:
            raise ValueError("Could not read image. Is it valid?")

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Example threshold detection
        thr_value = 200
        _, thresh = cv2.threshold(img_gray, thr_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Contours for bounding boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detection_img = img_bgr.copy()
        fault_parts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = img_bgr[y:y+h, x:x+w]
            fault_parts.append(encode_image(crop_img))

        # Confidence as ratio of hot pixels
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        hot_pixels = np.sum(thresh == 255)
        confidence = (hot_pixels / total_pixels) * 100.0
        
        prediction_text = "Hotspot Detected" if confidence >= 5 else "No Fault (Normal)"

        # ---------------------------------------------------------------------------------
        # ❗ Here, we derive "actual_temp" from the grayscale image
        #    For demonstration, compute average pixel intensity => approximate temperature
        avg_intensity = float(np.mean(img_gray))  # e.g. 0 to 255
        # Suppose 0 → 0°C, 255 → 80°C => 1 pixel intensity ~ 80/255 = ~0.3147°C
        # This is purely hypothetical
        max_temp_range = 80.0
        actual_temp = (avg_intensity / 255.0) * max_temp_range  # ~ range 0–80°C
        # ---------------------------------------------------------------------------------

        # Now define other metrics based on that derived temperature.
        # Perhaps "expected_output" or "actual_output" is also derived or retrieved from a DB?
        # We'll do a simple demonstration:

        # Let's say the "expected_output" correlates inversely with temperature above 25°C:
        # e.g. for each 1°C above 25°, we lose 5 kWh from the expected 1684.88 baseline.
        base_expected_output = 1684.88
        if actual_temp > 25:
            temp_diff = (actual_temp - 25)
            # lose 5kWh per deg above 25
            dynamic_expected = base_expected_output - (temp_diff * 5)
        else:
            dynamic_expected = base_expected_output

        # We'll pretend "actual_output" is 1420 if the threshold is big enough, else 1500
        actual_output = 1420.0 if confidence >= 5 else 1500.0

        # Example metrics again
        power_drop = ((dynamic_expected - actual_output) / dynamic_expected) * 100.0
        # Performance ratio (fake formula, ignoring real irradiance/time inputs)
        # We'll say installed power = 1000, irradiance=5.5, days=365 => same logic:
        installed_power = 1000
        irradiance = 5.5
        days = 365
        theoretical_max = installed_power * irradiance * days  # 2,007,500 kWh
        pr = (actual_output / theoretical_max) * 100.0

        # Suppose we define an "initial_output" as the dynamic_expected for the first year
        # and "current_output" as the actual output. Over 5 years:
        initial_output = dynamic_expected
        current_output = actual_output
        years = 5
        degradation_rate = ((initial_output - current_output) / (initial_output * years)) * 100.0

        # Temperature coefficient approach
        # We'll keep stc_temp=25, rated_power=1000, coefficient=-0.0045
        stc_temp = 25
        rated_power = 1000
        temp_coeff = -0.0045
        temperature_loss = temp_coeff * (actual_temp - stc_temp) * rated_power

        # Build the final analysis dictionary
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
                "observed_value": round(temperature_loss, 2),
                "inference": f"Loss due to temperature: {round(temperature_loss, 2)} kW"
            },
            # Show derived actual_temp for debugging
            "derived_temperature": {
                "value": round(actual_temp, 2),
                "units": "°C"
            }
        }

        # Encode images
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(detection_img)
        
        # Remove temporary file
        os.remove(temp_path)

        # Return JSON
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
