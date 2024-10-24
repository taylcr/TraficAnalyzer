import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Assuming you have the SORT tracker in a separate file called sort.py
import time

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")  # Change this path if needed

# # Initialize video capture (0 or 1 for webcam, or provide a video file path)
# cap = cv2.VideoCapture(0)  # Use '0' for default webcam, or a file path for video


# Initialize video capture from video file 'cars.mp4'
cap = cv2.VideoCapture("cars.mp4")  # Load the video file

# Initialize the SORT tracker
tracker = Sort()

# Class names for YOLO (COCO dataset classes)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Variables to keep track of cars
car_count = 0
line_position = 400  # The position of the virtual line for counting cars
cars_passed = set()  # To store IDs of cars that passed the line

while True:
    success, img = cap.read()
    if not success:
        break

    # YOLO detection
    results = model(img, stream=True)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if classNames[cls] == "car":  # Only track cars
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to np.array format required for SORT
    detections = np.array(detections)

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Draw tracking results
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_y = (y1 + y2) // 2  # Calculate the center y position of the car

        # Draw bounding box and track ID
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'Car {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Check if car has passed the counting line
        if center_y > line_position and track_id not in cars_passed:
            car_count += 1
            cars_passed.add(track_id)

    # Display car count on the screen
    cv2.putText(img, f'Car Count: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # Draw the counting line
    cv2.line(img, (0, line_position), (img.shape[1], line_position), (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Traffic Analysis", img)

    # Exit if 'q' is pressed or video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
