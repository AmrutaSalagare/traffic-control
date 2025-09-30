"""
Enhanced visualization test for the traffic detection system
"""
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
from final_tracking_onnx import ONNXTrafficDetector

def draw_enhanced_ui(frame, detector, fps, frame_count):
    """Draw enhanced UI elements on the frame"""
    
    # Colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_ORANGE = (0, 165, 255)
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Create overlay for semi-transparent backgrounds
    overlay = frame.copy()
    
    # Draw main info panel (top-left)
    panel_width = 300
    panel_height = 150
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "TRAFFIC MONITORING SYSTEM", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    # System stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
    cv2.putText(frame, f"Frame: {frame_count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    cv2.putText(frame, f"Resolution: {w}x{h}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    
    # Vehicle count panel (top-right)
    count_panel_width = 200
    count_panel_height = 80
    cv2.rectangle(overlay, (w - count_panel_width - 10, 10), 
                  (w - 10, 10 + count_panel_height), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "VEHICLE COUNT", (w - count_panel_width, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)
    cv2.putText(frame, f"{detector.vehicle_count}", (w - count_panel_width + 20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GREEN, 3)
    
    # Ambulance alert (center-top) - only show if ambulance detected
    if detector.ambulance_detected:
        alert_width = 250
        alert_height = 60
        alert_x = (w - alert_width) // 2
        alert_y = 20
        
        # Flashing red background
        flash_color = COLOR_RED if (frame_count // 10) % 2 == 0 else (0, 0, 150)
        cv2.rectangle(frame, (alert_x, alert_y), 
                      (alert_x + alert_width, alert_y + alert_height), flash_color, -1)
        cv2.rectangle(frame, (alert_x, alert_y), 
                      (alert_x + alert_width, alert_y + alert_height), COLOR_WHITE, 2)
        
        cv2.putText(frame, "🚨 AMBULANCE DETECTED! 🚨", (alert_x + 10, alert_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    # Detection statistics panel (bottom-left)
    stats_panel_width = 250
    stats_panel_height = 100
    cv2.rectangle(overlay, (10, h - stats_panel_height - 10), 
                  (10 + stats_panel_width, h - 10), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "DETECTION STATS", (20, h - stats_panel_height + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)
    
    # Count active detections
    active_objects = len(detector.tracker.objects)
    cv2.putText(frame, f"Active Objects: {active_objects}", (20, h - stats_panel_height + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)
    cv2.putText(frame, f"Total Crossed: {detector.vehicle_count}", (20, h - stats_panel_height + 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)
    
    # Controls panel (bottom-right)
    controls_text = ["Controls:", "Q - Quit", "R - Reset Count", "S - Screenshot"]
    for i, text in enumerate(controls_text):
        cv2.putText(frame, text, (w - 150, h - 60 + i * 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_WHITE, 1)

def draw_enhanced_detections(frame, tracked_objects, detector):
    """Draw enhanced bounding boxes and trajectories"""
    
    # Vehicle class colors
    CLASS_COLORS = {
        0: (0, 255, 0),      # car - green
        1: (0, 165, 255),    # motorcycle - orange  
        2: (255, 255, 0),    # bus - cyan
        3: (0, 0, 255),      # truck - red
        4: (255, 0, 0),      # auto-rickshaw - blue
        'ambulance': (0, 0, 255)  # ambulance - red
    }
    
    CLASS_NAMES = {
        0: 'Car', 1: 'Motorcycle', 2: 'Bus', 3: 'Truck', 4: 'Auto', 'ambulance': 'Ambulance'
    }
    
    # Draw trajectories first (so they appear behind boxes)
    detector.tracker.draw_trajectories(frame)
    
    # Draw detection line
    line_y = detector.count_line_y
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 3)
    cv2.putText(frame, "COUNTING LINE", (frame.shape[1] // 2 - 80, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw tracked objects with enhanced styling
    for obj_id, obj in tracked_objects.items():
        bbox = obj['bbox']
        class_name = obj['class']
        confidence = obj['confidence']
        
        # Get appropriate color
        if class_name == 'ambulance':
            color = CLASS_COLORS['ambulance']
            display_name = 'AMBULANCE'
        else:
            class_id = int(class_name) if class_name.isdigit() else 0
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            display_name = CLASS_NAMES.get(class_id, f'Vehicle')
        
        # Draw bounding box with enhanced styling
        x1, y1, x2, y2 = map(int, bbox)
        
        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Corner markers for better visibility
        corner_length = 15
        corner_thickness = 3
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        # Top-right corner  
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Label background
        label_text = f"{display_name} {confidence:.2f} [ID:{obj_id}]"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                      (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)

def test_enhanced_visualization():
    """Test the enhanced visualization system"""
    
    # Initialize detector
    print("Initializing traffic detection system...")
    try:
        detector = ONNXTrafficDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Open video source
    video_path = "videos/rushing.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print("Starting enhanced visualization test...")
    print("Controls:")
    print("  Q - Quit")
    print("  R - Reset vehicle count")
    print("  S - Save screenshot")
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    fps_history = deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video - restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue
        
        frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        if frame_count > 1:
            fps = 1.0 / (current_time - frame_time)
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)
        else:
            avg_fps = 0
        frame_time = current_time
        
        # Process frame with detector
        processed_frame = detector.process_frame(frame)
        
        # Get tracked objects
        tracked_objects = detector.tracker.objects
        
        # Draw enhanced UI elements
        draw_enhanced_ui(processed_frame, detector, avg_fps, frame_count)
        
        # Draw enhanced detections
        draw_enhanced_detections(processed_frame, tracked_objects, detector)
        
        # Display the frame
        cv2.imshow("Enhanced Traffic Detection System", processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.vehicle_count = 0
            detector.tracker.crossed_ids.clear()
            print("Vehicle count reset!")
        elif key == ord('s'):
            screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, processed_frame)
            print(f"Screenshot saved: {screenshot_name}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("FINAL STATISTICS:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Average FPS: {frame_count/total_time:.1f}")
    print(f"Final vehicle count: {detector.vehicle_count}")
    print(f"Ambulance detections: {'Yes' if detector.ambulance_detected else 'No'}")
    print("="*50)

if __name__ == "__main__":
    test_enhanced_visualization()
