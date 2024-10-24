import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QRect
from ultralytics import YOLO
from sort import Sort

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Traffic Analysis')

        # Set up video display label and button layout
        self.label = QLabel(self)
        self.label.setFixedSize(1280, 720)  # Set the window size to match the video

        self.run_button = QPushButton('Run Model', self)
        self.run_button.clicked.connect(self.run_model)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

        # Video capture and processing
        self.cap = cv2.VideoCapture("cars.mp4")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Drawing variables
        self.drawing = False
        self.areas_of_interest = []
        self.current_area_start = None
        self.current_area_end = None
        self.model_running = False
        self.video_paused = False

        # Load YOLO and SORT
        self.model = YOLO("../Yolo-Weights/yolov8n.pt")
        self.tracker = Sort()

        # Class names for YOLO (COCO dataset)
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

        # Start the video display
        self.timer.start(30)

    def paintEvent(self, event):
        """Paint event to handle drawing areas of interest."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw current area while dragging
        if self.drawing and self.current_area_start and self.current_area_end:
            pen = QPen(Qt.green, 3, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(self.current_area_start[0], self.current_area_start[1],
                         self.current_area_end[0] - self.current_area_start[0],
                         self.current_area_end[1] - self.current_area_start[1])
            painter.drawRect(rect)

        # Draw saved areas
        for area in self.areas_of_interest:
            pen = QPen(Qt.red, 3, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(area[0][0], area[0][1], area[1][0] - area[0][0], area[1][1] - area[0][1])
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        """Capture the first point of the rectangle."""
        if not self.model_running:
            self.drawing = True
            self.video_paused = True  # Pause the video while drawing
            self.timer.stop()  # Stop the timer to pause video updates
            self.current_area_start = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        """Capture the second point and update the rectangle."""
        if self.drawing:
            self.current_area_end = (event.x(), event.y())
            self.update()  # Update the drawing

    def mouseReleaseEvent(self, event):
        """Finalize the area of interest."""
        if self.drawing:
            self.drawing = False
            self.current_area_end = (event.x(), event.y())
            self.areas_of_interest.append([self.current_area_start, self.current_area_end])
            self.current_area_start = None
            self.current_area_end = None
            self.update()  # Finalize drawing
            self.timer.start(30)  # Resume video updates
            self.video_paused = False

    def run_model(self):
        """Run the YOLO and SORT model when the button is clicked."""
        self.model_running = True
        self.run_button.setDisabled(True)  # Disable the button once the model is running

    def update_frame(self):
        """Update the video frame and perform detection when model is running."""
        if not self.video_paused:  # Only update video if not paused for drawing
            ret, frame = self.cap.read()
            if not ret:
                return

            # Perform YOLO detection and tracking if the model is running
            if self.model_running:
                results = self.model(frame, stream=True)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if cls < len(self.classNames) and self.classNames[cls] == "car":
                            detections.append([x1, y1, x2, y2, conf])

                detections = np.array(detections)
                tracked_objects = self.tracker.update(detections)

                # Draw tracked objects on the frame
                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = track
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Car {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Check if the car is in any area of interest
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    for area in self.areas_of_interest:
                        if area[0][0] <= center_x <= area[1][0] and area[0][1] <= center_y <= area[1][1]:
                            cv2.putText(frame, f'In Area {self.areas_of_interest.index(area)}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # Display the video frame in PyQt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))

# Main application
app = QApplication(sys.argv)
window = VideoWidget()
window.show()
sys.exit(app.exec_())
