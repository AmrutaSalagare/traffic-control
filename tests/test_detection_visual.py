"""
Test script to visualize detection results and save to file
"""
import cv2
import numpy as np
import os
from final_tracking_onnx import ONNXTrafficDetector

def test_detection_on_video():
    """Test detection on video and save frames to see results"""
    
    # Initialize detector
    print("Initializing traffic detector...")
    try:
        detector = ONNXTrafficDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Open video file
    video_path = "videos/Untitled video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    print("Processing video frames...")
    
    frame_count = 0
    saved_frames = 0
    max_frames_to_save = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        frame_count += 1
        
        # Process frame
        try:
            processed_frame = detector.process_frame(frame)
            
            # Save some frames to see results
            if saved_frames < max_frames_to_save and frame_count % 30 == 0:  # Save every 30th frame
                output_filename = f"detection_result_frame_{frame_count}.jpg"
                cv2.imwrite(output_filename, processed_frame)
                print(f"Saved detection result: {output_filename}")
                saved_frames += 1
                
                # Print current statistics
                print(f"  Frame {frame_count}: Vehicle count: {detector.vehicle_count}, FPS: {detector.fps:.1f}")
                print(f"  Ambulance detected: {detector.ambulance_detected}")
                print(f"  Tracked objects: {len(detector.tracker.objects)}")
            
            # Process first 300 frames then stop
            if frame_count >= 300:
                print(f"Processed {frame_count} frames")
                break
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            break
    
    # Final statistics
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total vehicles counted: {detector.vehicle_count}")
    print(f"Final FPS: {detector.fps:.1f}")
    print(f"Ambulance detected: {detector.ambulance_detected}")
    print(f"Active tracked objects: {len(detector.tracker.objects)}")
    print("="*50)
    
    cap.release()
    print("Detection test completed!")

if __name__ == "__main__":
    test_detection_on_video()
