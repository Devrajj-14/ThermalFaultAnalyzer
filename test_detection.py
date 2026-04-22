#!/usr/bin/env python3
"""Quick test script to verify detection algorithm."""

import cv2
import numpy as np
from services.rule_based_service import extract_thermal_features, classify_fault_type, classify_severity

# Test with a few images
test_images = [
    'data/thermal_images/C1.jpeg',
    'data/thermal_images/C37.jpeg', 
    'data/thermal_images/C100.jpeg',
    'data/thermal_images/C500.jpeg',
    'data/temp_C37.jpeg'
]

print("Testing thermal detection algorithm...\n")
print("=" * 80)

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f'\n{img_path}: Could not read')
        continue
    
    features = extract_thermal_features(img)
    fault_type, confidence = classify_fault_type(features)
    severity = classify_severity(features, fault_type)
    
    print(f'\n{img_path}:')
    print(f'  Classification: {fault_type.upper()} | Severity: {severity.upper()} | Confidence: {confidence:.2f}')
    print(f'  Hotspot Area: {features["hotspot_area_percent"]:.2f}%')
    print(f'  Region Count: {features["region_count"]}')
    print(f'  Thermal Contrast: {features["thermal_contrast"]:.1f}')
    print(f'  Adaptive Threshold: {features["adaptive_threshold"]:.1f}')
    print(f'  Max Intensity: {features["max_intensity"]:.1f}')
    print(f'  Estimated Temp: {features["actual_temp"]:.1f}°C')

print("\n" + "=" * 80)
print("Test complete!")
