# flamingoDetection
Detect and Track flamingoes and people in a zoo using YOLOv8/YOLOv5 and DeepSORT

## Overview
This project implements flamingo and people detection and tracking in video streams using YOLOv8/YOLOv5 for object detection and DeepSORT for object tracking. It includes special handling for varying lighting conditions (shadows and sunny areas) to improve detection accuracy.

## Features
- Detects flamingos and people in video frames using YOLOv8 (object_detection_2.py) or YOLOv5 (object_detection.py)
- Enhanced detection in varying lighting conditions (shadows and sunny areas)
- Tracks detected objects across frames using DeepSORT
- Different colors for flamingoes (green) and people (red) for easy visual distinction
- Proximity detection and alerts when people get too close to flamingoes (object_detection_with_log.py)
- Side panel log display showing real-time detection information (object_detection_with_log.py)
- Detection log file generation with timestamps, object types, positions, and alerts (object_detection_with_log.py)
- Multiple fallback options if YOLOv5 is not available:
  - Torch Hub YOLOv5
  - OpenCV DNN with MobileNet SSD
  - Simulated detections

## Requirements
Install the required dependencies:
```bash
pip install -r pre_requirements.txt
```

## Usage
1. Place your video file in the `data` directory
2. Update the `video_path` variable in the script to point to your video file
3. Run one of the scripts:

For enhanced detection with YOLOv8 (recommended for varying lighting conditions):
```bash
cd scripts
python object_detection_2.py
```

For advanced features including proximity alerts and logging:
```bash
cd scripts
python object_detection_with_log.py
```

For the original implementation with YOLOv5 or simulated detections:
```bash
cd scripts
python object_detection.py
```

## Implementation Details
- YOLOv8/YOLOv5 is used for object detection, focusing on the 'bird' class (ID 14) as a proxy for flamingos and 'person' class (ID 0)
- Adaptive image preprocessing is applied to improve detection in varying lighting conditions:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) is used to enhance visibility in both shadows and sunny areas
  - Detection is performed on both original and enhanced frames to maximize detection accuracy
  - Duplicate detections are filtered using IoU (Intersection over Union)
- DeepSORT is used for tracking the detected objects across frames
- The output video will be saved in the `data` directory

## Customization
- To use a custom YOLOv8/YOLOv5 model trained specifically for flamingos, update the model loading line in the script
- Adjust detection thresholds in the code to fine-tune detection sensitivity
- Modify the `preprocess_frame` function to customize the image enhancement for your specific lighting conditions

## Handling Varying Lighting Conditions
The script uses several techniques to improve detection in challenging lighting conditions:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast in both dark (shadows) and bright (sunny) areas
2. **Dual-detection approach**: Runs object detection on both original and enhanced frames to maximize detection rate
3. **Lower confidence threshold**: Uses a slightly lower confidence threshold (0.4) to detect objects in challenging lighting
4. **Duplicate removal**: Implements IoU-based filtering to avoid counting the same object twice

These techniques significantly improve the detection of flamingos and people in areas with varying lighting conditions, such as zoos with both sunny and shaded areas.

## Proximity Detection and Alert System
The `object_detection_with_log.py` script includes a proximity detection system that:

1. **Monitors distances** between detected people and flamingoes
2. **Triggers alerts** when a person gets too close to a flamingo (configurable threshold)
3. **Visually highlights** proximity events with orange bounding boxes and connecting lines
4. **Logs alerts** to both the side panel and the log file with timestamps

This feature is particularly useful for zoo monitoring to ensure visitors maintain a safe distance from the animals.

## Logging and Visualization
The advanced logging features in `object_detection_with_log.py` include:

1. **Side panel display** showing real-time detection information including:
   - Timestamps
   - Object types (Flamingo/Person)
   - Object IDs
   - Positions
   - Confidence scores
   - Proximity alerts

2. **Log file generation** with detailed information:
   - CSV format for easy analysis
   - Complete timestamp for each detection
   - Object type and tracking ID
   - Position coordinates
   - Confidence scores
   - Alert status

The log file can be used for post-processing analysis, such as tracking visitor patterns, animal behavior, or security monitoring.
