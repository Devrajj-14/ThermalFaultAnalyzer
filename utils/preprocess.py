import cv2
import base64
import os

def detect_faults(image_path):
    """
    Given an image path, detect multiple hotspots/faulty areas.
    Returns a dictionary with:
        - 'prediction'
        - 'confidence'
        - 'input_image' (base64-encoded original image)
        - 'fault_detection' (base64-encoded image with bounding boxes)
        - 'fault_parts' (list of base64-encoded cropped images)
    """
    # (1) Read the original image (as BGR by default in OpenCV)
    original_img = cv2.imread(image_path)

    # (2) Define bounding boxes (x, y, width, height).
    #     In real usage, these come from your detection model.
    bboxes = [
        (50, 50, 100, 100),
        (200, 80, 100, 100),
        (350, 120, 80, 80),
        (500, 60, 80, 100)
    ]

    # (3) Create a copy for drawing bounding boxes
    detection_img = original_img.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # (4) Save the detection image (optional) so we can encode it
    fault_detection_path = image_path + "_fault.png"
    cv2.imwrite(fault_detection_path, detection_img)

    # (5) Crop each bounding box to create separate images
    fault_parts_base64 = []
    for (x, y, w, h) in bboxes:
        # Crop the region of interest
        crop = original_img[y:y+h, x:x+w]
        fault_parts_base64.append(encode_cv2_to_base64(crop))

    # (6) Build the result dictionary
    result = {
        "prediction": "No Fault (Normal)",  # or "Hotspot Detected"
        "confidence": 1.92,                # example confidence
        "input_image": encode_file_to_base64(image_path),
        "fault_detection": encode_file_to_base64(fault_detection_path),
        "fault_parts": fault_parts_base64
    }

    return result

def encode_file_to_base64(file_path):
    """Reads a file from disk and returns its Base64-encoded string."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_cv2_to_base64(cv2_image):
    """Encodes an OpenCV image (NumPy array) to Base64 (PNG format)."""
    success, buffer = cv2.imencode('.png', cv2_image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')
