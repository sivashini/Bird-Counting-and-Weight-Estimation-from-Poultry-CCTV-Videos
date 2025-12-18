"""
Bird detection and tracking module using YOLOv8 and ByteTrack
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ByteTracker:
    """Simple ByteTrack implementation for object tracking"""
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracks with new detections
        detections: List of [x1, y1, x2, y2, confidence]
        Returns: List of [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age existing tracks
            dead_tracks = []
            for track_id in self.tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.track_buffer:
                    dead_tracks.append(track_id)
            for track_id in dead_tracks:
                del self.tracks[track_id]
            return []
        
        # Filter high confidence detections
        high_conf_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d[4] < self.track_thresh]
        
        # Match high confidence detections to existing tracks
        matched_tracks, unmatched_dets, unmatched_tracks = self._match(
            high_conf_dets, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_id in matched_tracks:
            det = high_conf_dets[det_idx]
            self.tracks[track_id]['box'] = det[:4]
            self.tracks[track_id]['confidence'] = det[4]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
        
        # Try to match low confidence detections to unmatched tracks
        if len(low_conf_dets) > 0 and len(unmatched_tracks) > 0:
            unmatched_track_dict = {tid: self.tracks[tid] for tid in unmatched_tracks}
            matched_tracks_low, _, remaining_unmatched = self._match(
                low_conf_dets, unmatched_track_dict
            )
            
            for det_idx, track_id in matched_tracks_low:
                det = low_conf_dets[det_idx]
                self.tracks[track_id]['box'] = det[:4]
                self.tracks[track_id]['confidence'] = det[4]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
            
            unmatched_tracks = remaining_unmatched
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_conf_dets[det_idx]
            self.tracks[self.next_id] = {
                'box': det[:4],
                'confidence': det[4],
                'age': 0,
                'hits': 1,
                'start_frame': self.frame_count
            }
            self.next_id += 1
        
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
        
        # Remove dead tracks
        dead_tracks = [tid for tid, t in self.tracks.items() 
                      if t['age'] > self.track_buffer]
        for track_id in dead_tracks:
            del self.tracks[track_id]
        
        # Return active tracks
        results = []
        for track_id, track in self.tracks.items():
            if track['age'] == 0:  # Only return tracks updated this frame
                box = track['box']
                results.append([box[0], box[1], box[2], box[3], track_id])
        
        return results
    
    def _match(self, detections, tracks):
        """Match detections to tracks using IoU"""
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), list(tracks.keys())
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        track_ids = list(tracks.keys())
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                iou_matrix[i, j] = self._iou(det[:4], tracks[track_id]['box'])
        
        # Greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()
        
        while len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
            # Find max IoU
            max_iou = 0
            max_det = -1
            max_track = -1
            
            for i in unmatched_dets:
                for j, tid in enumerate(unmatched_tracks):
                    track_idx = track_ids.index(tid)
                    if iou_matrix[i, track_idx] > max_iou:
                        max_iou = iou_matrix[i, track_idx]
                        max_det = i
                        max_track = tid
            
            if max_iou < self.match_thresh:
                break
            
            matched.append((max_det, max_track))
            unmatched_dets.remove(max_det)
            unmatched_tracks.remove(max_track)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class BirdDetectorTracker:
    """Bird detection and tracking pipeline"""
    
    def __init__(self, conf_thresh=0.25, iou_thresh=0.45):
        """Initialize detector and tracker"""
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load YOLOv8 model (using nano for speed)
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # Initialize tracker
        self.tracker = ByteTracker(
            track_thresh=conf_thresh,
            track_buffer=30,
            match_thresh=0.7
        )
    
    def process_video(self, video_path: str, fps_sample: int = 5) -> Dict:
        """
        Process video and return tracking results
        
        Args:
            video_path: Path to input video
            fps_sample: Process every Nth frame
            
        Returns:
            Dictionary containing counts and tracks
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames at {fps} FPS")
        logger.info(f"Processing every {fps_sample} frame(s)")
        
        counts = []
        all_tracks = defaultdict(lambda: {
            'boxes': [],
            'confidences': [],
            'frames': []
        })
        
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % fps_sample == 0:
                # Detect birds
                detections = self._detect_birds(frame)
                
                # Update tracker
                tracks = self.tracker.update(detections)
                
                # Count birds
                bird_count = len(tracks)
                
                # Calculate timestamp
                timestamp = self._frame_to_timestamp(frame_idx, fps)
                
                counts.append({
                    'timestamp': timestamp,
                    'count': bird_count,
                    'frame': frame_idx
                })
                
                # Store track information
                for track in tracks:
                    track_id = int(track[4])
                    box = track[:4]
                    
                    # Find confidence from original detection
                    conf = 0.0
                    for det in detections:
                        if self._iou(box, det[:4]) > 0.5:
                            conf = det[4]
                            break
                    if hasattr(box, "tolist"):
                        box = box.tolist()

                    all_tracks[track_id]['boxes'].append(box)

                    all_tracks[track_id]['confidences'].append(float(conf))
                    all_tracks[track_id]['frames'].append(frame_idx)
                
                processed_frames += 1
                
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames, current count: {bird_count}")
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Processing complete: {processed_frames} frames processed")
        logger.info(f"Total tracks: {len(all_tracks)}")
        
        return {
            'counts': counts,
            'tracks': dict(all_tracks),
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_frames
            }
        }
    
    def _detect_birds(self, frame):
        """Detect birds in frame"""
        # Run detection
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for bird class (class 14 in COCO)
                if int(box.cls) == 14:  # bird class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    detections.append([x1, y1, x2, y2, conf])
        
        return detections
    
    def _iou(self, box1, box2):
        """Calculate IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _frame_to_timestamp(self, frame_idx, fps):
        """Convert frame index to timestamp string"""
        seconds = frame_idx / fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def create_annotated_video(self, video_path, tracks, counts, output_path, fps_sample=5):
        """Create annotated video with bounding boxes and tracking IDs"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / fps_sample, (width, height))
        
        # Create frame-to-tracks mapping
        frame_tracks = defaultdict(list)
        for track_id, track_data in tracks.items():
            for i, frame_idx in enumerate(track_data['frames']):
                frame_tracks[frame_idx].append({
                    'id': track_id,
                    'box': track_data['boxes'][i],
                    'conf': track_data['confidences'][i]
                })
        
        # Create frame-to-count mapping
        frame_counts = {c['frame']: c['count'] for c in counts}
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % fps_sample == 0:
                # Draw tracks
                if frame_idx in frame_tracks:
                    for track in frame_tracks[frame_idx]:
                        box = track['box']
                        track_id = track['id']
                        conf = track['conf']
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        color = self._get_color(track_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"ID:{track_id} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw count overlay
                count = frame_counts.get(frame_idx, 0)
                timestamp = self._frame_to_timestamp(frame_idx, fps)
                
                cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Count: {count}", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {timestamp}", (20, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Annotated video saved to {output_path}")
    
    def _get_color(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))