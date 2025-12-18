"""
FastAPI application for bird counting and weight estimation
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import logging

from bird_detector import BirdDetectorTracker
from weight_estimator import WeightEstimator
import utils


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Bird Counting and Weight Estimation API",
    description="Analyze poultry CCTV videos for bird counting and weight estimation",
    version="1.0.0"
)

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "message": "Service is running"}

@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    fps_sample: Optional[int] = Form(5, description="Process every Nth frame"),
    conf_thresh: Optional[float] = Form(0.25, description="Detection confidence threshold"),
    iou_thresh: Optional[float] = Form(0.45, description="IoU threshold for NMS")
):
    """
    Analyze video for bird counting and weight estimation
    
    Parameters:
    - video: Video file (MP4, AVI, etc.)
    - fps_sample: Process every Nth frame (default: 5)
    - conf_thresh: Detection confidence threshold (default: 0.25)
    - iou_thresh: IoU threshold for NMS (default: 0.45)
    
    Returns:
    - counts: Time series of bird counts
    - tracks_sample: Sample tracking data
    - weight_estimates: Weight proxy indices
    - artifacts: Paths to generated files
    """
    temp_video_path = None
    
    try:
        # Validate parameters
        if fps_sample < 1:
            raise HTTPException(status_code=400, detail="fps_sample must be >= 1")
        if not 0 < conf_thresh < 1:
            raise HTTPException(status_code=400, detail="conf_thresh must be between 0 and 1")
        if not 0 < iou_thresh < 1:
            raise HTTPException(status_code=400, detail="iou_thresh must be between 0 and 1")
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name
            shutil.copyfileobj(video.file, temp_file)
        
        logger.info(f"Processing video: {video.filename}")
        logger.info(f"Parameters - fps_sample: {fps_sample}, conf_thresh: {conf_thresh}, iou_thresh: {iou_thresh}")
        
        # Initialize detector and tracker
        detector_tracker = BirdDetectorTracker(
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        
        # Process video
        results = detector_tracker.process_video(
            temp_video_path,
            fps_sample=fps_sample
        )
        
        # Generate output filenames
        base_name = Path(video.filename).stem
        annotated_video_path = OUTPUT_DIR / f"{base_name}_annotated.mp4"
        tracks_json_path = OUTPUT_DIR / f"{base_name}_tracks.json"
        
        # Create annotated video
        logger.info("Creating annotated video...")
        detector_tracker.create_annotated_video(
            temp_video_path,
            results['tracks'],
            results['counts'],
            str(annotated_video_path),
            fps_sample=fps_sample
        )
        
        # Estimate weights
        logger.info("Estimating weights...")
        weight_estimator = WeightEstimator()
        weight_estimates = weight_estimator.estimate_weights(results['tracks'])
        
        # Save tracks to JSON
        utils.save_json(tracks_json_path, results['tracks'])
        
        # Prepare tracks sample (first 5 tracks with summary)
        tracks_sample = []
        for track_id, track_data in list(results['tracks'].items())[:5]:
            tracks_sample.append({
                "track_id": int(track_id),
                "boxes": track_data['boxes'][:10],  # First 10 boxes
                "confidences": track_data['confidences'][:10],
                "total_detections": len(track_data['boxes'])
            })
        
        # Prepare response
        response = {
            "counts": results['counts'],
            "tracks_sample": tracks_sample,
            "weight_estimates": weight_estimates,
            "artifacts": {
                "annotated_video": str(annotated_video_path),
                "tracks_json": str(tracks_json_path)
            },
            "video_info": {
                "filename": video.filename,
                "total_frames_processed": len(results['counts']),
                "total_tracks": len(results['tracks'])
            }
        }
        
        logger.info("Analysis complete")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Close uploaded file
        await video.close()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bird Counting and Weight Estimation API",
        "endpoints": {
            "health": "/health",
            "analyze_video": "/analyze_video (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)