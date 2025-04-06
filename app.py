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
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    temp_path = os.path.join("data", f"temp_{image_file.filename}")
    image_file.save(temp_path)

    try:
        # Read the image from disk
        img_bgr = cv2.imread(temp_path)
        if img_bgr is None:
            raise ValueError("Could not read image. Is it valid?")
        
        # Convert image to grayscale for thresholding
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to detect hot (bright) regions
        thr_value = 200
        _, thresh = cv2.threshold(img_gray, thr_value, 255, cv2.THRESH_BINARY)

        # Use morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours of the fault areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image for drawing the detection boxes
        detection_img = img_bgr.copy()
        fault_parts = []

        # Process each contour: draw the bounding box and crop the region
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = img_bgr[y:y+h, x:x+w]
            fault_parts.append(encode_image(crop_img))

        # Calculate confidence as the ratio of hot pixels to total pixels
        total_pixels = img_gray.shape[0] * img_gray.shape[1]
        hot_pixels = np.sum(thresh == 255)
        confidence = (hot_pixels / total_pixels) * 100.0

        if confidence < 5:
            prediction = "No Fault (Normal)"
        else:
            prediction = "Hotspot Detected"

        # Encode the original and detection images for display
        input_encoded = encode_image(img_bgr)
        detection_encoded = encode_image(detection_img)

        # Clean up the temporary file
        os.remove(temp_path)

        # Return all images and prediction results
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "input_image": input_encoded,
            "fault_detection": detection_encoded,
            "fault_parts": fault_parts
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
