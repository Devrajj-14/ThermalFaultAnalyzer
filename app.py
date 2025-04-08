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

        # Convert image to grayscale for thresholding
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect bright/hot regions
        thr_value = 200
        _, thresh = cv2.threshold(img_gray, thr_value, 255, cv2.THRESH_BINARY)
        
        # Use morphological opening to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours of the fault areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes for each fault area on a copy
        detection_img = img_bgr.copy()
        fault_parts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = img_bgr[y:y+h, x:x+w]
            fault_parts.append(encode_image(crop_img))
        
        # Confidence can be based on the ratio of "hot" pixels to total pixels
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        hot_pixels = np.sum(thresh == 255)
        confidence = (hot_pixels / total_pixels) * 100.0

        if confidence < 5:
            prediction_text = "No Fault (Normal)"
        else:
            prediction_text = "Hotspot Detected"

        # ----- Sample Calculation Steps -----
        # 1️⃣ Power Output Drop (%)
        expected_output = 1684.88
        actual_output = 1420.0
        power_drop = ((expected_output - actual_output) / expected_output) * 100.0  # ≈ 15.7%

        # 2️⃣ Performance Ratio (PR)
        # Given: Irradiance = 5.5 kWh/m²/day, Time = 365 days, Installed Power = 1000 kWp
        theoretical_max = 1000 * 5.5 * 365  # = 2007,500 kWh = 2007.5 MWh
        pr = (actual_output / theoretical_max) * 100.0  # ≈ 70.7%

        # 3️⃣ Annual Degradation Rate
        years = 5
        degradation_rate = ((expected_output - actual_output) / (expected_output * years)) * 100.0  # ≈ 3.14% per year

        # 4️⃣ Temperature Loss Calculation
        # Given: Temp Coefficient = -0.0045, Actual Temp = 40°C, STC = 25°C, Rated Power = 1000 kWp
        temperature_loss = -0.0045 * (40 - 25) * 1000  # = -67.5 kW
        
        # Overall Inference Table (as a nested dict)
        analysis = {
            "power_output_drop": {
                "observed_value": round(power_drop, 2),
                "standard_threshold": "<10%",
                "status": "🔴",
                "inference": "Power drop is 15.7%. Since >10% loss is significant, modules should be closely inspected."
            },
            "performance_ratio": {
                "observed_value": round(pr, 2),
                "standard_threshold": "≥75%",
                "status": "🔴",
                "inference": "PR has dropped below 75%, a critical threshold, signaling substantial system inefficiencies."
            },
            "degradation_rate": {
                "observed_value": round(degradation_rate, 2),
                "standard_threshold": "<0.7%/year",
                "status": "🔴",
                "inference": "Degradation is high (ideal is <0.7%/year). Replacement or remediation is necessary."
            },
            "temperature_loss": {
                "observed_value": round(temperature_loss, 2),
                "inference": "Operating temperature causes a loss of 67.5 kW, which contributes to long-term wear."
            }
        }
        
        # ----- Encoding Images -----
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(detection_img)

        # Clean up temporary file
        os.remove(temp_path)

        # Return a complete JSON response including images and analysis metrics
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
