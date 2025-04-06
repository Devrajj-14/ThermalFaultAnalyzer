import cv2
import base64
import os

def detect_faults(image_path):
    """
    Given an image path, detect multiple hotspots/faulty areas, 
    return dict with:
      - 'prediction'
      - 'confidence'
      - 'input_image' (base64)
      - 'fault_detection' (base64)
      - 'fault_parts' (array of base64)
    """

    # (1) Read the original image
    original_img = cv2.imread(image_path)

    # (2) For demonstration, define fake bounding boxes 
    #     (x, y, width, height). In a real scenario, 
    #     these come from your model's detection.
    bboxes = [
        (50, 50, 100, 100),
        (200, 80, 100, 100),
        (350, 120, 80, 80),
        (500, 60, 80, 100)
    ]

    # (3) Draw bounding boxes on a copy of the original image
    #     for the "fault_detection" overlay
    detection_img = original_img.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # (4) Save the detection image to a file (optional)
    fault_detection_path = image_path + "_fault.png"
    cv2.imwrite(fault_detection_path, detection_img)

    # (5) Crop each bounding box to create a separate "fault part" image
    fault_parts_base64 = []
    for (x, y, w, h) in bboxes:
        crop = original_img[y:y+h, x:x+w]
        fault_parts_base64.append(encode_cv2_image_to_base64(crop))

    # (6) Return the results in a dictionary
    return {
        "prediction": "Hotspot Detected",
        "confidence": 16.84,  # or your real confidence score
        "input_image": encode_image(image_path),
        "fault_detection": encode_image(fault_detection_path),
        "fault_parts": fault_parts_base64
    }

def encode_image(path):
    """
    Read an image file from disk and return its Base64 string
    """
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_cv2_image_to_base64(image):
    """
    Convert an OpenCV image (NumPy array) to PNG and then Base64
    """
    success, buffer = cv2.imencode('.png', image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')
