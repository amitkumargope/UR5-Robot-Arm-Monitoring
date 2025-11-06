"""
Utility functions for the Robot Gripper Detection project.
"""

import cv2
import os


def get_video_info(video_path):
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing fps, width, height, and frame count
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }


def resize_frame(frame, scale_factor=0.6):
    """
    Resize a video frame by a scale factor.
    
    Args:
        frame: Input frame from video
        scale_factor (float): Factor to scale the frame by (default 0.6)
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(frame, (new_width, new_height))


def ensure_directory_exists(path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str): Path to the directory
    """
    os.makedirs(path, exist_ok=True)


def validate_video_source(source):
    """
    Validate if the video source is accessible.
    
    Args:
        source: Path to video file or camera index
        
    Returns:
        bool: True if source is valid, False otherwise
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        # It's a camera index
        cap = cv2.VideoCapture(int(source))
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    elif isinstance(source, str):
        # It's a file path
        return os.path.exists(source)
    else:
        return False