import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = '../runs/detect/train2/weights/best.pt'
INPUT_VIDEO_PATH = '../data/Flamingo_30sec_excerpt.avi'
OUTPUT_VIDEO_PATH = '../data/flamingo_tracking_output.mp4'
# ---------------------

# 1. Load your fine-tuned YOLO model
model = YOLO(MODEL_PATH)

# 2. Open the input video file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
    exit()

# 3. Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 4. Define the codec and create VideoWriter object for the output file
# The 'mp4v' codec is commonly used for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print("Processing video... This may take a while.")

# 5. Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Run YOLO prediction on the frame
        results = model(frame)

        # The '.plot()' method automatically draws the bounding boxes on the frame
        annotated_frame = results[0].plot(labels=False, conf=False)

        # Write the annotated frame to the output video
        out.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# 6. Release everything when the job is finished
print(f"Processing complete. Video saved to {OUTPUT_VIDEO_PATH}")
cap.release()
out.release()
cv2.destroyAllWindows()