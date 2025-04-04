import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="static")

def encode_image(img):
    """Encode a CV2 BGR image as base64 string in PNG format."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    temp_path = f"data/temp_{image_file.filename}"
    image_file.save(temp_path)

    try:
        # 1) Read image (BGR) & convert to grayscale
        img_bgr = cv2.imread(temp_path)
        if img_bgr is None:
            raise ValueError("Could not read image. Is it valid?")

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2) Threshold to find "hot" (bright) regions
        #    Adjust 'thr_value' if your images are darker/lighter
        thr_value = 200
        _, thresh = cv2.threshold(img_gray, thr_value, 255, cv2.THRESH_BINARY)

        # 3) Morphological filtering to remove noise (adjust kernel or iterations)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 4) Find external contours of hot areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5) Create "Fault Detection" image by drawing bounding boxes
        detection_img = img_bgr.copy()
        all_xmin, all_xmax = [], []
        all_ymin, all_ymax = [], []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw bounding box in red
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0,0,255), 2)
            # Track min/max for "Fault Part"
            all_xmin.append(x)
            all_xmax.append(x+w)
            all_ymin.append(y)
            all_ymax.append(y+h)

        # 6) Create "Fault Part" image
        if len(contours) > 0:
            x_min = min(all_xmin)
            x_max = max(all_xmax)
            y_min = min(all_ymin)
            y_max = max(all_ymax)
            fault_part_img = detection_img[y_min:y_max, x_min:x_max]
        else:
            # No faults => small black image
            fault_part_img = np.zeros((50, 50, 3), dtype=np.uint8)

        # 7) Compute confidence = fraction of hot pixels
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        hot_pixels = np.sum(thresh == 255)
        confidence = (hot_pixels / total_pixels) * 100.0

        # 8) Basic label
        if confidence < 5:
            prediction = "No Fault (Normal)"
        else:
            prediction = "Hotspot Detected"

        # 9) Encode 3 images: 
        #    - Input (original BGR)
        #    - Fault Detection
        #    - Fault Part
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(detection_img)
        fault_encoded = encode_image(fault_part_img)

        os.remove(temp_path)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "input_image": input_encoded,
            "fault_detection": detection_encoded,
            "fault_part": fault_encoded
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
