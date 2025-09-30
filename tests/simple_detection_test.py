"""
Simple detection test to debug what's happening
"""
import cv2
import numpy as np
import os
from onnx_inference import ONNXYOLODetector, ONNXAmbulanceDetector

def simple_test():
    """Simple test to see what's being detected"""
    
    # Initialize detectors
    print("Initializing detectors...")
    
    try:
        # Initialize vehicle detector
        vehicle_model_path = "optimized_models/yolo11n_optimized.onnx"
        vehicle_detector = ONNXYOLODetector(vehicle_model_path, conf_thres=0.1)  # Lower threshold
        print("Vehicle detector initialized!")
        
        # Initialize ambulance detector  
        ambulance_model_path = "optimized_models/indian_ambulance_yolov11n_best_optimized.onnx"
        ambulance_detector = ONNXAmbulanceDetector(ambulance_model_path, conf_thres=0.05)  # Very low threshold
        print("Ambulance detector initialized!")
        
    except Exception as e:
        print(f"Error initializing detectors: {e}")
        return
    
    # Open video
    video_path = "videos/Untitled video.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    print("Processing video frames...")
    
    frame_count = 0
    total_vehicle_detections = 0
    total_ambulance_detections = 0
    
    # Process first 50 frames
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
            
        frame_count += 1
        
        # Run detections
        try:
            print(f"\n--- Frame {frame_count} ---")
            
            # Vehicle detection
            vehicle_detections = vehicle_detector.detect(frame)
            print(f"Vehicle detections: {len(vehicle_detections)}")
            
            if len(vehicle_detections) > 0:
                total_vehicle_detections += len(vehicle_detections)
                for i, det in enumerate(vehicle_detections):
                    bbox = det['bbox']
                    conf = det['confidence']
                    class_name = det.get('class_name', 'unknown')
                    print(f"  Vehicle {i+1}: {class_name} confidence={conf:.2f} bbox={bbox}")
            
            # Ambulance detection (every frame for this test)
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
                
                # Draw vehicle detections in green
                for det in vehicle_detections:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    conf = det['confidence']
                    class_name = det.get('class_name', 'vehicle')
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{class_name} {conf:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
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
                cv2.putText(display_frame, f"Vehicles: {len(vehicle_detections)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Ambulances: {len(ambulance_detections)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save the frame
                output_file = f"simple_test_frame_{frame_count}.jpg"
                cv2.imwrite(output_file, display_frame)
                print(f"Saved: {output_file}")
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
    
    cap.release()
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"Frames processed: {frame_count}")
    print(f"Total vehicle detections: {total_vehicle_detections}")
    print(f"Total ambulance detections: {total_ambulance_detections}")
    print(f"Average vehicles per frame: {total_vehicle_detections/frame_count:.1f}")
    print(f"Average ambulances per frame: {total_ambulance_detections/frame_count:.1f}")
    print("="*50)

if __name__ == "__main__":
    simple_test()
