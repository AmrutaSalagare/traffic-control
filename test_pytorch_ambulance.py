"""
Test original PyTorch ambulance model vs ONNX
"""
import cv2
import os
from ultralytics import YOLO

def test_pytorch_ambulance():
    print("="*60)
    print("TESTING PYTORCH AMBULANCE MODEL")
    print("="*60)
    
    # Check if original PyTorch model exists
    pytorch_model_path = "models/indian_ambulance_yolov11n_best.pt"
    
    if not os.path.exists(pytorch_model_path):
        print(f"PyTorch model not found: {pytorch_model_path}")
        return
    
    try:
        # Load PyTorch model
        model = YOLO(pytorch_model_path)
        print(f"PyTorch model loaded: {pytorch_model_path}")
        
        # Load test frame
        cap = cv2.VideoCapture("videos/ammuvid.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 250)  # Frame where ambulance should be
        ret, frame = cap.read()
        
        if not ret:
            print("Could not read frame")
            return
        
        print(f"Frame shape: {frame.shape}")
        
        # Test different confidence levels
        confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        for conf in confidence_levels:
            print(f"\nTesting PyTorch model with confidence: {conf}")
            results = model(frame, conf=conf, verbose=False)
            
            if len(results[0].boxes) > 0:
                print(f"PyTorch detections: {len(results[0].boxes)}")
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    area = (x2 - x1) * (y2 - y1)
                    print(f"  Detection {i+1}: conf={confidence:.4f}, area={area:.0f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            else:
                print(f"PyTorch detections: 0")
        
        cap.release()
        
    except Exception as e:
        print(f"Error testing PyTorch model: {e}")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("The ONNX model shows 0 detections at all confidence levels")
    print("Check PyTorch model results above to see if the issue is with:")
    print("1. ONNX conversion process")  
    print("2. Model preprocessing differences")
    print("3. The specific ambulance in this video")

if __name__ == "__main__":
    test_pytorch_ambulance()
