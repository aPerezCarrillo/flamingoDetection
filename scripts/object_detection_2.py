from collections import defaultdict

import cv2
import numpy as np
import time

# ... (imports and video setup) ...
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
# model = YOLO('yolov8n.pt') # 'n' for nano, a small and fast model. Use 'yolov8s.pt' for slightly better accuracy.
model = YOLO('../runs/detect/train2/weights/best.pt')
# model = YOLO('yolov8s.pt')
# Identify the class ID for 'bird' in COCO dataset (usually 14 for YOLOv8/COCO)
# You can check model.names to confirm class IDs
# e.g., print(model.names)
BIRD_CLASS_ID = 14 # Confirm this from model.names
PERSON_CLASS_ID = 0 # Person class ID in COCO dataset

# Define a function to apply adaptive image preprocessing
def preprocess_frame(frame):
    """
    Apply adaptive image preprocessing to improve detection in varying lighting conditions.

    Args:
        frame: Input frame from video

    Returns:
        Preprocessed frame with improved contrast and brightness
    """
    # Convert to HSV color space for better handling of lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the V channel
    # This helps with both dark and bright areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)

    # Merge the channels back
    hsv_clahe = cv2.merge([h, s, v_clahe])

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    return enhanced_frame

# --- Placeholder for Object Detection Model ---
# In a real scenario, you'd load your pre-trained Mask R-CNN, YOLO, etc. here.
# For simplicity, we'll simulate detections or use a very generic pre-trained model if available.

# Example: If using a simple pre-trained OpenCV DNN model (e.g., MobileNet SSD)
# This model might detect 'bird' or 'animal' but not specifically 'flamingo'.
# Replace these paths with your actual model files if you have them.
# net = cv2.dnn.readNetFromCaffe("path/to/MobileNetSSD_deploy.prototxt", "path/to/MobileNetSSD_deploy.caffemodel")
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# DESIRED_CLASSES = ["bird"] # Or "flamingo" if your model supports it

# --- Placeholder for DeepSORT Tracker ---
# We'll use the 'deep_sort_realtime' library.
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Video Stream Configuration ---
video_path = '../data/Flamingo_30sec_excerpt.avi'  # <<< IMPORTANT: Change this to your video file path
output_video_path = '../data/flamingo_tracking_output.mp4'  # Using .mp4 extension for QuickTime compatibility

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# XVID is a common codec, you might need 'mp4v' or 'MJPG' depending on your system
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize DeepSORT tracker
# The `max_iou_distance` and `n_init` parameters can be tuned.
# `max_iou_distance`: maximum cosine distance or IOU distance between track and detection.
# `n_init`: number of consecutive detections before a track is confirmed.
tracker = DeepSort(max_iou_distance=0.7, n_init=3,
                   nms_max_overlap=1.0)  # nms_max_overlap=1.0 to disable NMS within DeepSORT

# --- Trajectory Storage ---
# Stores {track_id: [(x1, y1), (x2, y2), ...]}
# Using defaultdict will automatically create an empty list for new track_ids
trajectories = defaultdict(list)
max_trajectory_length = 100 # How many past points to store for drawing the tail

frame_count = 0
start_time = time.time()


frame_count = 0
start_time = time.time()

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_count += 1

    # Create a copy of the original frame for display
    display_frame = frame.copy()

    # Apply preprocessing to enhance the frame for better detection
    enhanced_frame = preprocess_frame(frame)

    # Run detection on both original and enhanced frames
    results_original = model(frame, verbose=False)[0]
    results_enhanced = model(enhanced_frame, verbose=False)[0]

    # Combine detections from both frames
    detections_for_tracker = []

    # Process original frame detections
    for box in results_original.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        # Track both birds (flamingoes) and people
        if (class_id == BIRD_CLASS_ID or class_id == PERSON_CLASS_ID) and confidence > 0.4:
            w = x2 - x1
            h = y2 - y1
            detections_for_tracker.append(([x1, y1, w, h], confidence, model.names[class_id]))

    # Process enhanced frame detections, avoiding duplicates
    for box in results_enhanced.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        # Track both birds (flamingoes) and people
        if (class_id == BIRD_CLASS_ID or class_id == PERSON_CLASS_ID) and confidence > 0.4:
            w = x2 - x1
            h = y2 - y1

            # Check if this detection overlaps significantly with any existing detection
            is_duplicate = False
            for existing_det in detections_for_tracker:
                ex_x, ex_y, ex_w, ex_h = existing_det[0]
                # Calculate IoU (Intersection over Union)
                x_left = max(x1, ex_x)
                y_top = max(y1, ex_y)
                x_right = min(x1 + w, ex_x + ex_w)
                y_bottom = min(y1 + h, ex_y + ex_h)

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w * h
                    area2 = ex_w * ex_h
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > 0.5:  # If IoU > 0.5, consider it a duplicate
                        is_duplicate = True
                        # Keep the detection with higher confidence
                        if confidence > existing_det[1]:
                            detections_for_tracker.remove(existing_det)
                            is_duplicate = False
                        break

            if not is_duplicate:
                detections_for_tracker.append(([x1, y1, w, h], confidence, model.names[class_id]))


    # --- Update DeepSORT Tracker ---
    # `update_tracks` takes a list of detections (as specified above)
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # --- Draw Detections and Tracks on Frame ---
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Calculate center point for trajectory
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Store current center point in trajectory
        trajectories[track_id].append((center_x, center_y))

        # Limit trajectory length to keep it from getting too long
        if len(trajectories[track_id]) > max_trajectory_length:
            trajectories[track_id].pop(0)  # Remove the oldest point



        # Draw bounding box with different colors for different classes
        if any(det[2] == "bird" for det in detections_for_tracker if det[0][0] == x1 and det[0][1] == y1):
            color = (0, 255, 0)  # Green for birds/flamingoes
            class_name = "Flamingo"
        elif any(det[2] == "person" for det in detections_for_tracker if det[0][0] == x1 and det[0][1] == y1):
            color = (0, 0, 255)  # Red for people
            class_name = "Person"
        else:
            color = (255, 255, 0)  # Cyan for other tracked objects
            class_name = "Object"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put track ID and class text
        cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Draw Trajectory ---
        # Draw lines connecting the past points
        for i in range(1, len(trajectories[track_id])):
            pt1 = trajectories[track_id][i - 1]
            pt2 = trajectories[track_id][i]
            # Use a slightly different color or transparency for trajectory
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue line for trajectory

    # Add information about preprocessing to the frame
    cv2.putText(frame, "Enhanced Detection for Varying Light Conditions", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display frame
    cv2.imshow('Flamingo Tracking', frame)

    # Write the frame to the output video
    out.write(frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print(f"Average FPS: {frame_count / (end_time - start_time):.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing finished.")
