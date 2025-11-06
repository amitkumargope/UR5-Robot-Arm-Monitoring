"""
Main entry point for the Robot Gripper Detection project.

This script provides a unified interface to run:
1. Training of the YOLOv8 segmentation model
2. Testing/inference on video input
3. Evaluation of model performance
"""

import argparse
import os
import sys

# Add the src directory to the Python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.train_robot_seg_yolov8 import main as train_model
from models.test import main as run_test


def main():
    parser = argparse.ArgumentParser(description="Robot Gripper Detection System")
    parser.add_argument(
        "mode", 
        choices=["train", "test", "evaluate"], 
        help="Choose the mode: train, test, or evaluate"
    )
    parser.add_argument("--model-path", type=str, help="Path to model for testing")
    parser.add_argument("--video-path", type=str, help="Path to video for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        # For now, we'll call the test function directly
        # In a more complex system, we'd pass parameters from args
        run_test()
    elif args.mode == "evaluate":
        print("Evaluation mode coming soon...")
        

if __name__ == "__main__":
    main()