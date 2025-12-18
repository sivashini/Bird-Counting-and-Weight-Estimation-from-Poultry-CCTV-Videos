# Bird Counting and Weight Estimation System

## Overview
This system performs bird detection, tracking, and weight estimation from CCTV video footage using YOLOv8 and ByteTrack.

## Features
- Real-time bird detection with bounding boxes
- Stable tracking with unique IDs across frames
- Weight estimation using bird area and movement patterns
- FastAPI REST API for video analysis
- Annotated output videos with tracking visualization

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the API

Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

### API Documentation
Access interactive API docs at: http://localhost:8000/docs

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Analyze Video
```bash
curl.exe -X POST "http://localhost:8000/analyze_video" `
    -F "video=@sample_video.mp4" `
    -F "fps_sample=5" `
    -F "conf_thresh=0.25" `
    -F "iou_thresh=0.45" `
    -o response.json
```


#### Parameters
- `video`: Video file (required)
- `fps_sample`: Process every Nth frame (default: 5)
- `conf_thresh`: Detection confidence threshold (default: 0.25)
- `iou_thresh`: IoU threshold for NMS (default: 0.45)

### Response Format
```json
{
  "counts": [
    {"timestamp": "00:00:00", "count": 15, "frame": 0},
    {"timestamp": "00:00:01", "count": 16, "frame": 30}
  ],
  "tracks_sample": [
    {
      "track_id": 1,
      "boxes": [[x1, y1, x2, y2], ...],
      "confidences": [0.92, 0.89, ...]
    }
  ],
  "weight_estimates": {
    "unit": "index",
    "per_bird": [
      {"track_id": 1, "weight_index": 245.3, "confidence": 0.85}
    ],
    "aggregate": {
      "mean_weight_index": 238.5,
      "std": 15.2,
      "total_birds": 18
    },
    "calibration_note": "Weight index based on bird area. Requires calibration with known weights."
  },
  "artifacts": {
    "annotated_video": "output_annotated_video.mp4",
    "tracks_json": "output_tracks.json"
  }
}
```

## Implementation Details

### Bird Counting Method
1. **Detection**: YOLOv8n pretrained on COCO (bird class)
2. **Tracking**: ByteTrack algorithm for stable ID assignment
3. **Occlusion Handling**: 
   - Track persistence across missing detections
   - IoU-based matching with Kalman filtering
   - Re-identification using appearance features
4. **Double-counting Prevention**: Unique track IDs maintained throughout video

### Weight Estimation Approach
Since ground truth weights are unavailable, we use a **Weight Proxy Index** based on:

1. **Bird Area**: Bounding box area (pixels²) averaged over trajectory
2. **Body Density Factor**: Normalized area considering camera distance
3. **Movement Patterns**: Stationary birds (better measurement)

**Weight Index Formula**:
```
weight_index = mean_area × density_factor × confidence_multiplier
```

**Calibration Requirements for Grams**:
- Known weights for 10-20 birds at various growth stages
- Camera calibration data (pixel-to-cm ratio at floor level)
- Regression model: weight_grams = α × weight_index + β

**Assumptions**:
- Fixed camera position and angle
- Birds are on a flat surface
- Consistent lighting conditions
- Camera height remains constant

### Technology Stack
- **Detection**: Ultralytics YOLOv8
- **Tracking**: ByteTrack
- **API**: FastAPI
- **Video Processing**: OpenCV
- **Numerical Computing**: NumPy, SciPy

## Project Structure
```
bird_counting/
├── main.py                 # FastAPI application
├── bird_detector.py        # Detection and tracking logic
├── weight_estimator.py     # Weight estimation module
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── README.md              # This file
├── outputs/               # Generated outputs
└── sample_videos/         # Input videos
```

## Output Artifacts

1. **Annotated Video**: Shows bounding boxes, track IDs, and bird count overlay
2. **Tracks JSON**: Complete tracking data for analysis
3. **Analysis JSON**: Summary statistics and weight estimates

## Performance Notes
- Processing speed: ~15-30 FPS on GPU, ~3-5 FPS on CPU
- Frame sampling recommended for long videos
- Memory usage: ~2GB for 5-minute video

## Future Improvements
- Implement re-identification with appearance features
- Add depth estimation for better weight calibration
- Support multiple camera views
- Real-time streaming analysis
- Integration with weight scale data for calibration

