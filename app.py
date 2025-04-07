import os
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, Response
import threading

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model (update the path to your best.pt model)
model_path = "my_model/train/weights/best.pt"  # Path to your YOLO model
model = YOLO(model_path, task='detect')  # Load the YOLO model
labels = model.names  # Get labels for YOLO classes

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam (you can change this if you have a different camera source)

# Define frame size (optional, can be set to your desired resolution)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Set bounding box colors (for displaying on detected objects)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Function to process and detect objects in frames
def process_frame():
    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            break
        
        # Run YOLO detection on the frame
        results = model(frame)

        # Extract detections
        detections = results[0].boxes
        for i in range(len(detections)):
            # Extract bounding box coordinates and confidence score
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            # Only draw bounding boxes for detections with confidence > 0.5
            if conf > 0.5:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Encode the frame as JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame = jpeg.tobytes()  # Convert to bytes for sending over HTTP
        else:
            break
        
        # Send frame back to the browser for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the index page (you can add HTML or just display a simple message)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
