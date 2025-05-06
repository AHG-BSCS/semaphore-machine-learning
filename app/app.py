import numpy as np
import cv2
import os
import time
import base64
import threading
import atexit
import webbrowser
import signal
import sys
import re
from flask import Flask, jsonify, request, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class Detection:
    def __init__(self):
        self.model = YOLO("model/yolov12.pt")
        self.latest_detection = "No detections"

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, vid=False):
        results = self.predict(img, classes, conf=conf)
        detection_info = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detection_info.append({'label': label, 'confidence': confidence})
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), rectangle_thickness)
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 30), 
                              (int(box.xyxy[0][0]) + 120, int(box.xyxy[0][1]) - 5), (0, 255, 0), -1)
                cv2.putText(img, f"{label} {confidence:.2f}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), text_thickness)
                if vid:
                    frame_height, frame_width = img.shape[:2]
                    text = label
                    (text_width, text_height), _ = cv2.getTextSize("W", cv2.FONT_HERSHEY_PLAIN, 3, 3)
                    x = frame_width - text_width - 20
                    y = frame_height // 2 + text_height // 2
                    cv2.rectangle(img, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        self.latest_detection = detection_info[0]['label'] if detection_info else "No detections"
        return img, detection_info

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img

detection = Detection()

class AsyncDetector(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640, 640))
            processed_frame, _ = detection.predict_and_detect(frame)
            with self.lock:
                self.frame = processed_frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            _, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()

    def stop(self):
        self.running = False
        self.cap.release()

detector = None

@atexit.register
def cleanup():
    global detector
    if detector is not None and detector.is_alive():
        detector.stop()

@app.route('/start_camera')
def start_camera():
    global detector
    if detector is None or not detector.is_alive():
        detector = AsyncDetector()
        detector.start()
    return jsonify({"status": "Camera started"})

@app.route('/stop_camera')
def stop_camera():
    global detector
    if detector is not None:
        detector.stop()
    return jsonify({"status": "Camera stopped"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-models')
def get_models():
    model_folder = "model/"
    models = [f for f in os.listdir(model_folder) if f.endswith('.pt')]
    return jsonify(models)

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    model_name = request.form.get('model_name')
    if file.filename == '' or not model_name:
        return jsonify({"error": "Invalid input"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))

    try:
        file.save(file_path)
        detection.model = YOLO(os.path.join("model", model_name))
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read the image.")
        img = cv2.resize(img, (640, 640))
        img, _ = detection.predict_and_detect(img)
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        detected_text = detection.latest_detection
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "result_img": img_base64,
        "detected_text": detected_text
    }), 200

@app.route('/live_video.html')
def live_video():
    return render_template('live_video.html')

@app.route('/img_classification.html')
def img_classification():
    return render_template('img_classification.html')

@app.route('/vid_classification.html')
def vid_classification():
    return render_template('vid_classification.html')

@app.route('/get_detection_result')
def get_detection_result():
    return jsonify({'letter': detection.latest_detection})

@app.route('/set_model', methods=['POST'])
def set_model():
    data = request.get_json()
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    try:
        detection.model = YOLO(os.path.join("model", model_name))
        return jsonify({"message": f"Model switched to {model_name}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            frame = detector.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_classification', methods=['POST'])
def video_classification():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')

    try:
        file.save(file_path)

        video_capture = cv2.VideoCapture(file_path)
        if not video_capture.isOpened():
            return jsonify({"error": "Failed to open uploaded video."}), 500

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0 or width == 0 or height == 0:
            video_capture.release()
            return jsonify({"error": "Invalid video properties."}), 500

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                break
            processed_frame, _ = detection.predict_and_detect(frame, vid=True)
            out.write(processed_frame)
            time.sleep(0.03)

        video_capture.release()
        out.release()

        if not os.path.exists(output_path):
            return jsonify({"error": "Processed video not saved."}), 500

        return jsonify({
            "message": "Video processed successfully",
            "video_url": "/uploads/processed_video.mp4"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/uploads/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4', as_attachment=False)

def signal_handler(sig, frame):
    global detector
    if detector is not None and detector.is_alive():
        detector.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    webbrowser.open("http://localhost:8000/")
    time.sleep(1)
    app.run(host="0.0.0.0", port=8000)