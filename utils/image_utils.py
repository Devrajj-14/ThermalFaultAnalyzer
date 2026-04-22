"""
Image utility functions: encoding, cropping, and preprocessing.
"""

import base64
import cv2
import numpy as np
from typing import List


def encode_image(img: np.ndarray) -> str:
    """Encode a CV2 BGR image as a base64 PNG string."""
    if img is None:
        return ""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def crop_fault_regions(img_bgr: np.ndarray, contours: list) -> List[str]:
    """
    Crop each fault region as a square-padded thumbnail.

    Each crop is expanded to a square with 10% padding around the bounding
    box, then resized to 128×128. This avoids the thin-strip artefact that
    occurs when bounding boxes span the full image width.

    Only crops that are at least 20×20 pixels (after padding) are included.
    """
    if img_bgr is None or not contours:
        return []

    h, w = img_bgr.shape[:2]
    crops = []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # 10% padding on each side
        pad_x = max(int(bw * 0.10), 4)
        pad_y = max(int(bh * 0.10), 4)

        x1 = max(bx - pad_x, 0)
        y1 = max(by - pad_y, 0)
        x2 = min(bx + bw + pad_x, w)
        y2 = min(by + bh + pad_y, h)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Skip degenerate crops (thin strips wider than 8× their height)
        if crop_w < 20 or crop_h < 20:
            continue
        aspect = max(crop_w, crop_h) / max(min(crop_w, crop_h), 1)
        if aspect > 6:
            continue

        crop = img_bgr[y1:y2, x1:x2]

        # Pad to square
        side = max(crop_w, crop_h)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        off_x = (side - crop_w) // 2
        off_y = (side - crop_h) // 2
        square[off_y:off_y + crop_h, off_x:off_x + crop_w] = crop

        # Resize to 128×128 thumbnail
        thumb = cv2.resize(square, (128, 128), interpolation=cv2.INTER_AREA)
        crops.append(encode_image(thumb))

    return crops


def safe_read_image(path: str) -> np.ndarray:
    """
    Read image from path. Returns None if unreadable.
    
    Args:
        path: file path
    
    Returns:
        BGR image array or None
    """
    img = cv2.imread(path)
    return img
