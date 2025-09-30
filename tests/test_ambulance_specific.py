"""
Test ambulance detection on ambulance-specific video
"""
import cv2
import numpy as np
import os
from onnx_inference import ONNXAmbulanceDetector

def test_ambulance_detection():
    """Test ambulance detection on ambulance video"""
    
    # Initialize ambulance detector
    print("Initializing ambulance detector...")
    
    try:
        ambulance_model_path = "optimized_models/indian_ambulance_yolov11n_best_optimized.onnx"
        ambulance_detector = ONNXAmbulanceDetector(ambulance_model_path, conf_thres=0.05)  # Very low threshold
        print("Ambulance detector initialized!")
        
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Test on ambulance video
    video_path = "test_ambulance_Ambulance Running.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    print("Processing ambulance video frames...")
    
    frame_count = 0
    total_ambulance_detections = 0
    
    # Process first 30 frames
    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
            
        frame_count += 1
        
        # Run ambulance detection
        try:
            print(f"\n--- Frame {frame_count} ---")
            
            ambulance_detections = ambulance_detector.detect(frame)
            print(f"Ambulance detections: {len(ambulance_detections)}")
            
            if len(ambulance_detections) > 0:
                total_ambulance_detections += len(ambulance_detections)
                for i, det in enumerate(ambulance_detections):
                    bbox = det['bbox']
                    conf = det['confidence']
                    print(f"  Ambulance {i+1}: confidence={conf:.2f} bbox={bbox}")
                
                # Save frame with detections every 10 frames
                if frame_count % 10 == 0:
                    display_frame = frame.copy()
                    
                    # Draw ambulance detections in red
                    for det in ambulance_detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        conf = det['confidence']
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(display_frame, f"AMBULANCE {conf:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add frame info
                    cv2.putText(display_frame, f"Frame {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Ambulances: {len(ambulance_detections)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save the frame
                    output_file = f"ambulance_test_frame_{frame_count}.jpg"
                    cv2.imwrite(output_file, display_frame)
                    print(f"Saved: {output_file}")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
    
    cap.release()
    
    print(f"\n{'='*50}")
    print("AMBULANCE DETECTION SUMMARY:")
    print(f"Frames processed: {frame_count}")
    print(f"Total ambulance detections: {total_ambulance_detections}")
    print(f"Average ambulances per frame: {total_ambulance_detections/frame_count:.1f}")
    print("="*50)

if __name__ == "__main__":
    test_ambulance_detection()
