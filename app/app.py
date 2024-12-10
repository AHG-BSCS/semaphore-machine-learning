import io
import numpy as np
import cv2
import os
import time
import base64
import webbrowser
from flask import Flask, jsonify, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from PIL import ImageOps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

class Detection:
    def __init__(self):
        self.model = YOLO(r"model\best.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img

detection = Detection()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Read and resize the image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Failed to read the image.")

            img = cv2.resize(img, (640, 640))

            # Process the image
            img = detection.detect_from_image(img)

            # Encode the processed image to base64
            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.remove(file_path)

        # Return the base64 string
        return jsonify({"result_img": img_base64}), 200

@app.route('/live_video.html')
def live_video():
    return render_template('live_video.html')

@app.route('/img_classification.html')
def img_classification():
    return render_template('img_classification.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 640))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    webbrowser.open("http://localhost:8000/")
    time.sleep(1)
    app.run(host="0.0.0.0", port=8000)
