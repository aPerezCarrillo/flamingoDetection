import cv2
import numpy as np
import time

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
# Try mp4v (H.264) first, then fall back to avc1 if mp4v is not available
# Both codecs are compatible with QuickTime Player
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec, compatible with QuickTime
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Check if VideoWriter was successfully initialized
if not out.isOpened():
    print("Warning: Could not initialize VideoWriter with mp4v codec. Trying avc1...")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Alternative H.264 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Warning: Could not initialize VideoWriter with avc1 codec. Trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec, widely compatible
        output_video_path = output_video_path.replace('.mp4', '.avi')  # Change extension to .avi for XVID
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Could not initialize VideoWriter with any codec. Exiting.")
            cap.release()
            exit()

# Initialize DeepSORT tracker
# The `max_iou_distance` and `n_init` parameters can be tuned.
# `max_iou_distance`: maximum cosine distance or IOU distance between track and detection.
# `n_init`: number of consecutive detections before a track is confirmed.
tracker = DeepSort(max_iou_distance=0.7, n_init=3,
                   nms_max_overlap=1.0)  # nms_max_overlap=1.0 to disable NMS within DeepSORT

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

    # --- Object Detection Step (Placeholder) ---
    # In a real application, you would run your Mask R-CNN or YOLO model here.
    # The output should be a list of detections in the format:
    # detections = [([x, y, w, h], confidence, class_name), ...]

    # For demonstration, let's simulate some detections or use a very basic one
    # If you have an actual model, integrate it here.

    # Example using a dummy detection for a "flamingo"
    # Replace this with actual model inference
    dummy_detections = []
    # if frame_count % 30 == 0: # Simulate detections every 30 frames
    #     # Example: A flamingo detected around the center, slightly offset each time
    #     dummy_detections.append(([frame_width // 2 - 50, frame_height // 2 - 100, 100, 200], 0.95, "flamingo"))
    #     dummy_detections.append(([frame_width // 3 - 50, frame_height // 3 - 100, 100, 200], 0.88, "flamingo"))

    # For a more realistic placeholder, we'll assume a pre-trained detector is run
    # and gives us a list of bounding boxes and confidences for flamingos.
    # Let's imagine 'detect_flamingos(frame)' is your function for this.
    # It should return a list of (x, y, w, h, confidence, class_id) tuples or similar.

    # --- SIMULATED DETECTIONS (Replace with your actual model inference) ---
    # This is where you would integrate your actual object detection model.
    # If you have a YOLO model, you'd run model(frame) and parse its output.
    # If you have Mask R-CNN, you'd run model.detect(frame) and process results.

    # For now, let's make up some detections if we haven't integrated a model yet.
    # The 'deep_sort_realtime' library expects detections as:
    # `[([left, top, width, height], confidence, class_name), ...]`

    # Placeholder: Generate more realistic bounding boxes for flamingos
    # This will generate smaller, more reasonable boxes in the bottom half of the screen
    detections_for_tracker = []
    if frame_count % 5 == 0:  # Simulate detection every 5 frames
        num_flamingos_to_sim = np.random.randint(1, 4)  # Limit to 3 flamingos max
        for _ in range(num_flamingos_to_sim):
            # Generate smaller, more reasonable bounding boxes
            x = np.random.randint(50, frame_width - 100)
            y = np.random.randint(frame_height // 2, frame_height - 100)
            w = np.random.randint(40, 80)  # Smaller width
            h = np.random.randint(80, 160)  # Smaller height
            confidence = np.random.uniform(0.7, 0.99)
            detections_for_tracker.append(([x, y, w, h], confidence, "flamingo"))

    # Draw the raw detections (before tracking) in a different color
    for det in detections_for_tracker:
        bbox, conf, class_name = det
        x, y, w, h = [int(v) for v in bbox]
        # Draw detection box in blue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # Add confidence text
        cv2.putText(frame, f"{class_name}: {conf:.2f}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- Update DeepSORT Tracker ---
    # `update_tracks` takes a list of detections (as specified above)
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # --- Draw Detections and Tracks on Frame ---
    # Add a legend to explain the colors
    cv2.putText(frame, "Blue: Raw Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "Colored: Tracked Flamingos", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw tracks (confirmed objects being tracked)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Draw bounding box with a unique color based on track_id
        # Generate a unique color for each track_id to make them distinguishable
        color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)

        # Draw rectangle outline with thicker lines for tracked objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw a more visible track ID text with background
        text = f"Flamingo ID: {track_id}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)  # Background for text
        cv2.putText(frame, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text

        # Draw a dot at the center of the bounding box for better visibility
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, color, -1)

    # Display frame
    cv2.imshow('Flamingo Tracking', frame)

    # Process frame before writing to video
    # This helps prevent color space issues that might cause a green screen
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Check if frame is a color image
        # Make a copy to avoid modifying the displayed frame
        output_frame = frame.copy()

        # For mp4v/avc1 codecs on macOS/QuickTime, we need to convert BGR to RGB
        # This is crucial to fix the green screen issue
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the output video
        out.write(output_frame)
    else:
        # Handle grayscale or other formats if needed
        print("Warning: Unexpected frame format. Converting to RGB before writing.")
        # Convert to RGB before writing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        out.write(rgb_frame)

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
