"""
Test script for ambulance detection on static image
"""
import cv2
import numpy as np
from onnx_inference import ONNXAmbulanceDetector

def test_image_detection():
    """Test ambulance detection on a static image"""
    
    # Initialize ambulance detector
    print("Initializing ambulance detector...")
    try:
        ambulance_detector = ONNXAmbulanceDetector(
            "optimized_models/indian_ambulance_yolov11n_best_optimized.onnx", 
            conf_thres=0.1
        )
        print("Ambulance detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Create a test image (you can replace this with an actual image path)
    # For now, let's create a random test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Processing test image...")
    
    # Run detection
    try:
        detections = ambulance_detector.detect(test_image)
        print(f"Detection completed. Found {len(detections)} ambulances.")
        
        if len(detections) > 0:
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"  Detection {i+1}: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], confidence={conf:.3f}")
        
        # Draw detections on image
        display_image = test_image.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"Ambulance {conf:.2f}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save result image
        cv2.imwrite("test_detection_result.jpg", display_image)
        print("Detection result saved as 'test_detection_result.jpg'")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_detection()
