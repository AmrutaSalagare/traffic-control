"""
Test preprocessing to see actual values
"""
import cv2
import numpy as np
import os
from onnx_inference import ONNXYOLODetector

def test_preprocessing():
    """Test preprocessing values"""
    
    # Initialize detector
    vehicle_model_path = "optimized_models/yolo11n_optimized.onnx"
    vehicle_detector = ONNXYOLODetector(vehicle_model_path, conf_thres=0.3)
    
    # Open video and get a frame
    video_path = "videos/Untitled video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Skip to frame 10
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            return
    
    print(f"Frame shape: {frame.shape}")
    print(f"Model input size: {vehicle_detector.input_width}x{vehicle_detector.input_height}")
    
    # Test preprocessing
    img_preprocessed, ratio, pad = vehicle_detector.preprocess(frame)
    
    print(f"Preprocessing results:")
    print(f"  Ratio: {ratio}")
    print(f"  Pad (dx, dy): {pad}")
    print(f"  Preprocessed shape: {img_preprocessed.shape}")
    
    # Test coordinate conversion manually
    img_height, img_width = frame.shape[:2]
    dx, dy = pad
    
    print(f"\nManual calculations:")
    print(f"  Original image: {img_width}x{img_height}")
    print(f"  Input size: {vehicle_detector.input_width}x{vehicle_detector.input_height}")
    print(f"  Ratio: {ratio}")
    print(f"  Padding: dx={dx}, dy={dy}")
    
    # Test a sample coordinate conversion
    # Let's say we have a detection at center of model input (320, 320)
    model_x, model_y = 320, 320
    
    # Convert back to original coordinates
    orig_x = (model_x - dx) / ratio
    orig_y = (model_y - dy) / ratio
    
    print(f"\nCoordinate conversion test:")
    print(f"  Model coordinate (320, 320) -> Original ({orig_x:.1f}, {orig_y:.1f})")
    
    cap.release()

if __name__ == "__main__":
    test_preprocessing()
