"""
Scan through video to find when ambulance appears
"""
import cv2
from ultralytics import YOLO
import os

def scan_video_for_ambulance():
    print("="*60)
    print("SCANNING VIDEO FOR AMBULANCE")
    print("="*60)
    
    # Load PyTorch model
    pytorch_model_path = "models/indian_ambulance_yolov11n_best.pt"
    
    if not os.path.exists(pytorch_model_path):
        print(f"PyTorch model not found: {pytorch_model_path}")
        return
    
    model = YOLO(pytorch_model_path)
    print(f"Model loaded: {pytorch_model_path}")
    
    # Open video
    cap = cv2.VideoCapture("videos/ammuvid.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
    
    # Test frames at different intervals
    test_frames = [
        50,   # 2 seconds
        100,  # 3.3 seconds  
        200,  # 6.7 seconds
        300,  # 10 seconds
        400,  # 13.3 seconds
        500,  # 16.7 seconds
        600,  # 20 seconds
        700,  # 23.3 seconds
        800,  # 26.7 seconds
        900,  # 30 seconds
        total_frames - 100  # Near end
    ]
    
    ambulance_found = False
    
    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        time_seconds = frame_num / fps
        print(f"\nTesting frame {frame_num} (time: {time_seconds:.1f}s)")
        
        # Test with very low confidence
        results = model(frame, conf=0.01, verbose=False)
        
        if len(results[0].boxes) > 0:
            print(f"🚨 AMBULANCE FOUND at frame {frame_num}! ({len(results[0].boxes)} detections)")
            ambulance_found = True
            
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                area = (x2 - x1) * (y2 - y1)
                print(f"  Detection {i+1}: conf={confidence:.4f}, area={area:.0f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        else:
            print(f"No ambulance at frame {frame_num}")
    
    cap.release()
    
    if not ambulance_found:
        print("\n" + "="*60)
        print("❌ NO AMBULANCE DETECTED IN ANY TEST FRAMES")
        print("="*60)
        print("Possible reasons:")
        print("1. The ambulance model was not trained on this type of ambulance")
        print("2. The ambulance appears in frames we didn't test")
        print("3. The ambulance in this video doesn't match the training data")
        print("4. The model needs different preprocessing")
        
        # Let's also test with the working system's approach
        print("\nLet's check what the working system (final_tracking_detection.py) detects...")
        print("That system uses a dedicated ambulance detector with different approach")
    
    else:
        print(f"\n✅ Found ambulance in video! Check frames around the detected times.")

if __name__ == "__main__":
    scan_video_for_ambulance()
