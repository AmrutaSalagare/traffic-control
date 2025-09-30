"""
ByteTrack-style tracker with homography-based counting for high accuracy vehicle tracking
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from scipy.optimize import linear_sum_assignment
import time


@dataclass
class ByteTrackedVehicle:
    """Tracked vehicle using ByteTrack methodology"""
    track_id: int
    class_name: str
    score: float

    # State tracking
    state: str = "tracked"  # "tracked", "lost", "removed"
    tracklet_len: int = 0
    start_frame: int = 0

    # Position and motion
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))

    # History for appearance features (simple)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    score_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Kalman filter
    kalman_filter: Optional[Any] = None

    def __post_init__(self):
        """Initialize Kalman filter after creation"""
        self._init_kalman()

    def _init_kalman(self):
        """Initialize Kalman filter for motion prediction"""
        # Simple constant velocity model
        # State: [x, y, s, r, dx, dy, ds, dr] where s=scale, r=aspect_ratio
        self.kalman_filter = KalmanBoxTracker()

    def predict(self):
        """Predict next state"""
        if self.kalman_filter:
            self.mean = self.kalman_filter.predict()
        return self.mean

    def update(self, bbox: List[float], score: float):
        """Update track with new detection"""
        self.score = score
        self.tracklet_len += 1
        self.state = "tracked"

        if self.kalman_filter:
            self.kalman_filter.update(bbox)
            self.mean = self.kalman_filter.get_state()

        self.bbox_history.append(bbox)
        self.score_history.append(score)

    def mark_missed(self):
        """Mark track as missed/lost"""
        self.state = "lost"
        if self.kalman_filter:
            self.kalman_filter.update(None)


class KalmanBoxTracker:
    """Kalman filter for bounding box tracking (similar to SORT)"""

    count = 0

    def __init__(self, bbox=None):
        """Initialize Kalman filter for a bounding box"""
        from filterpy.kalman import KalmanFilter

        # State: [x, y, s, r, dx, dy, ds, dr]
        # where x,y is center, s is scale (area), r is aspect ratio w/h
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        if bbox is not None:
            self.kf.x[:4] = self._convert_bbox_to_z(bbox).reshape(-1, 1)
        else:
            self.kf.x[:4] = np.zeros((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update the state with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if bbox is not None:
            self.kf.update(self._convert_bbox_to_z(bbox).reshape(-1, 1))

    def predict(self):
        """Predict the current state"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return current bounding box estimate"""
        return self._convert_x_to_bbox(self.kf.x)

    def _convert_bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x, score=None):
        """Convert [x,y,s,r] to [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return [x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]


class ByteTracker:
    """ByteTrack implementation for robust multi-object tracking"""

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracked_stracks = []  # type: list[ByteTrackedVehicle]
        self.lost_stracks = []     # type: list[ByteTrackedVehicle]
        self.removed_stracks = []  # type: list[ByteTrackedVehicle]

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        # For counting
        self.track_id_count = 0
        self.counted_ids = set()
        # Track IDs that have been confirmed as real vehicles
        self.confirmed_vehicles = set()
        self.min_frames_for_confirmation = 5  # Minimum frames to confirm a vehicle

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections"""
        self.frame_id += 1

        # Convert detections to required format
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) > 0:
            # Separate high and low confidence detections
            scores = [det['confidence'] for det in detections]

            # First association with high confidence detections
            inds_high = [i for i, s in enumerate(
                scores) if s >= self.track_thresh]
            inds_low = [i for i, s in enumerate(
                scores) if s < self.track_thresh]

            dets_high = [detections[i] for i in inds_high]
            dets_low = [detections[i] for i in inds_low]

            if len(dets_high) > 0:
                # Predict tracked tracks
                for track in self.tracked_stracks:
                    track.predict()

                # Associate high confidence detections
                matches, u_track, u_detection = self._associate(
                    self.tracked_stracks, dets_high, self.match_thresh)

                # Update matched tracks
                for m in matches:
                    track = self.tracked_stracks[m[0]]
                    det = dets_high[m[1]]
                    track.update(det['bbox'], det['confidence'])
                    activated_starcks.append(track)

                # Handle unmatched tracks
                for it in u_track:
                    track = self.tracked_stracks[it]
                    if track.state != "lost":
                        track.mark_missed()
                        lost_stracks.append(track)

                # Initialize new tracks for unmatched high-conf detections
                for inew in u_detection:
                    track = self._initiate_track(dets_high[inew])
                    if track.score >= self.track_thresh:
                        activated_starcks.append(track)

                # Second association with low confidence detections
                if len(dets_low) > 0:
                    r_tracked_stracks = [
                        t for t in lost_stracks if t.tracklet_len >= 2]
                    matches, u_track, u_detection = self._associate(
                        r_tracked_stracks, dets_low, 0.5)

                    for m in matches:
                        track = r_tracked_stracks[m[0]]
                        det = dets_low[m[1]]
                        track.update(det['bbox'], det['confidence'])
                        activated_starcks.append(track)
                        lost_stracks.remove(track)
                        refind_stracks.append(track)

        # Update state lists
        for track in self.lost_stracks:
            if self.frame_id - track.start_frame > self.max_time_lost:
                track.state = "removed"
                removed_stracks.append(track)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == "tracked"]
        self.tracked_stracks.extend(activated_starcks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks = [
            t for t in self.lost_stracks if t.state != "removed"]
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Update confirmed vehicles count - only count vehicles that have been tracked for minimum frames
        for track in self.tracked_stracks:
            if track.tracklet_len >= self.min_frames_for_confirmation and track.track_id not in self.confirmed_vehicles:
                self.confirmed_vehicles.add(track.track_id)

        # Prepare output
        output_stracks = [
            track for track in self.tracked_stracks if track.tracklet_len >= 2]
        return self._tracks_to_detections(output_stracks)

    def _associate(self, tracks, detections, thresh):
        """Associate tracks and detections using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            track_bbox = track.kalman_filter.get_state(
            ) if track.kalman_filter else track.bbox_history[-1]
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calc_iou(track_bbox, det['bbox'])

        # Hungarian algorithm assignment
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > thresh).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_tracks = []
        for i, _ in enumerate(tracks):
            if len(matched_indices) == 0 or (matched_indices[:, 0] != i).all():
                unmatched_tracks.append(i)

        unmatched_detections = []
        for j, _ in enumerate(detections):
            if len(matched_indices) == 0 or (matched_indices[:, 1] != j).all():
                unmatched_detections.append(j)

        # Filter matches by threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < thresh:
                unmatched_tracks.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Initialize new track"""
        self.track_id_count += 1
        track = ByteTrackedVehicle(
            track_id=self.track_id_count,
            class_name=detection['class'],
            score=detection['confidence'],
            start_frame=self.frame_id
        )
        track.kalman_filter = KalmanBoxTracker(detection['bbox'])
        track.update(detection['bbox'], detection['confidence'])
        return track

    def _calc_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _tracks_to_detections(self, tracks):
        """Convert tracks back to detection format"""
        detections = []
        for track in tracks:
            if track.kalman_filter:
                bbox = track.kalman_filter.get_state()
                detections.append({
                    'bbox': bbox,
                    'confidence': track.score,
                    'class': track.class_name,
                    'track_id': track.track_id
                })
        return detections


class HomographyCounter:
    """Homography-based counting for accurate vehicle counting"""

    def __init__(self):
        self.homography_matrix = None
        self.count_lines = []  # List of lines in world coordinates
        self.crossed_tracks = set()
        self.total_count = 0
        self.count_history = deque(maxlen=100)

    def set_homography(self, src_points, dst_points):
        """Set homography matrix from image to world coordinates"""
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        self.homography_matrix = cv2.findHomography(src_points, dst_points)[0]

    def add_count_line(self, p1, p2, direction="both"):
        """Add a counting line in world coordinates"""
        self.count_lines.append({
            'p1': p1,
            'p2': p2,
            'direction': direction,
            'crossed_ids': set()
        })

    def update_counts(self, tracked_detections):
        """Update counts based on line crossings"""
        if not self.homography_matrix or not self.count_lines:
            return

        for detection in tracked_detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue

            # Get vehicle center in image coordinates
            bbox = detection['bbox']
            center_img = np.array(
                [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2, 1])

            # Transform to world coordinates
            center_world = self.homography_matrix @ center_img
            center_world = center_world[:2] / center_world[2]

            # Check line crossings
            for i, line in enumerate(self.count_lines):
                if track_id not in line['crossed_ids']:
                    if self._check_line_crossing(center_world, line):
                        line['crossed_ids'].add(track_id)
                        self.total_count += 1
                        self.count_history.append({
                            'track_id': track_id,
                            'line_id': i,
                            'timestamp': datetime.now(),
                            'class': detection['class']
                        })

    def _check_line_crossing(self, point, line):
        """Check if point crossed the counting line"""
        # Simple implementation - in practice you'd track previous positions
        # and check for actual line crossings
        p1, p2 = np.array(line['p1']), np.array(line['p2'])

        # Calculate distance from point to line
        line_vec = p2 - p1
        point_vec = point - p1
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return False

        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)

        # Check if projection is on line segment
        if 0 <= proj_length <= line_len:
            proj_point = p1 + proj_length * line_unitvec
            dist = np.linalg.norm(point - proj_point)
            return dist < 50  # 50 pixel threshold in world coordinates

        return False

    def get_count_stats(self):
        """Get counting statistics"""
        return {
            'total_count': self.total_count,
            'recent_crossings': list(self.count_history)[-10:],
            'lines_configured': len(self.count_lines)
        }


class AccurateVehicleTracker:
    """Combined ByteTrack + Homography counter for maximum accuracy"""

    def __init__(self):
        self.byte_tracker = ByteTracker(
            track_thresh=0.4,
            track_buffer=30,
            match_thresh=0.7
        )
        self.homography_counter = HomographyCounter()

        # Simple line-based fallback counter
        self.line_counter_y = None
        self.track_positions = {}
        self.line_crossed_ids = set()
        self.line_count = 0

        # For compatibility with standard tracker interface
        self.objects = []

        # Trajectory visualization support (like VehicleTracker)
        self.trajectory_points = {}  # {track_id: deque of center points}
        self.max_trajectory_length = 20  # Default length
        self.show_trajectories = True
        self.track_colors = {}
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (255, 20, 147), (34, 139, 34), (255, 140, 0)
        ]

    def set_count_line(self, y_ratio=0.55):
        """Set simple horizontal counting line"""
        self.line_counter_y = y_ratio

    def update(self, detections, frame_shape=None):
        """Update tracking and counting"""
        # ByteTrack update
        tracked_detections = self.byte_tracker.update(detections)

        # Update objects list for compatibility
        self.objects = self.byte_tracker.tracked_stracks

        # Homography counting if configured
        if self.homography_counter.homography_matrix:
            self.homography_counter.update_counts(tracked_detections)

        # Simple line counting fallback
        if self.line_counter_y and frame_shape:
            self._update_line_count(tracked_detections, frame_shape)

        return tracked_detections

    def get_confirmed_detections(self):
        """Get only confirmed vehicle detections for visual display"""
        confirmed_tracks = [
            track for track in self.byte_tracker.tracked_stracks
            if track.track_id in self.byte_tracker.confirmed_vehicles
        ]
        return self.byte_tracker._tracks_to_detections(confirmed_tracks)

    def _update_line_count(self, detections, frame_shape):
        """Update simple line-based counting"""
        h, w = frame_shape[:2]
        line_y = int(h * self.line_counter_y)

        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            bbox = det['bbox']
            center_y = (bbox[1] + bbox[3]) / 2

            # Check for line crossing
            prev_y = self.track_positions.get(track_id)
            if prev_y is not None and track_id not in self.line_crossed_ids:
                if (prev_y < line_y <= center_y) or (prev_y > line_y >= center_y):
                    self.line_crossed_ids.add(track_id)
                    self.line_count += 1

            self.track_positions[track_id] = center_y

    def get_count(self):
        """Get unique vehicle count for compatibility with standard tracker"""
        return self.byte_tracker.track_id_count

    def get_counts(self):
        """Get all counting metrics"""
        stats = {
            'bytetrack_active': len(self.byte_tracker.tracked_stracks),
            # Use confirmed vehicles instead of raw track count
            'bytetrack_total': len(self.byte_tracker.confirmed_vehicles),
            # Keep raw count for debugging
            'bytetrack_raw_total': self.byte_tracker.track_id_count,
            'line_count': self.line_count,
            'homography_count': self.homography_counter.total_count
        }
        return stats

    def update_trajectory(self, track_id: int, bbox: list):
        """Update trajectory points for a tracked vehicle (compatibility with VehicleTracker)"""
        from collections import deque

        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        if track_id not in self.trajectory_points:
            self.trajectory_points[track_id] = deque(
                maxlen=self.max_trajectory_length)
            # Assign color to new track
            color_idx = track_id % len(self.color_palette)
            self.track_colors[track_id] = self.color_palette[color_idx]

        self.trajectory_points[track_id].append((center_x, center_y))

    def draw_trajectories(self, frame):
        """Draw vehicle trajectories on the frame (compatibility with VehicleTracker)"""
        import cv2

        if not self.show_trajectories:
            return frame

        for track_id, points in self.trajectory_points.items():
            if len(points) < 2:
                continue

            # Get color for this track
            color = self.track_colors.get(track_id, (0, 255, 0))

            # Draw trajectory lines
            for i in range(1, len(points)):
                if points[i-1] is None or points[i] is None:
                    continue

                # Draw line with decreasing thickness/opacity for older points
                thickness = max(1, int(3 * (i / len(points))))
                cv2.line(frame, points[i-1], points[i], color, thickness)

            # Draw current position as a circle
            if points:
                cv2.circle(frame, points[-1], 4, color, -1)

        return frame

    def get_track_color(self, track_id: int):
        """Get consistent color for a track ID (compatibility with VehicleTracker)"""
        return self.track_colors.get(track_id, (0, 255, 0))
