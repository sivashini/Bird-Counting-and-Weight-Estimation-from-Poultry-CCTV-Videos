"""
Utility functions for the bird counting system
"""
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict


def save_json(filepath: Path, data: Dict):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert_to_serializable)


def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def get_box_center(box):
    """Get center point of bounding box"""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def get_box_area(box):
    """Calculate area of bounding box"""
    return (box[2] - box[0]) * (box[3] - box[1])