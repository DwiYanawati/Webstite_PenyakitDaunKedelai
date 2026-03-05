from flask import Flask, render_template, Response, request
import os
import cv2
from ultralytics import YOLO
from pyngrok import ngrok

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

# Folder
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Kamera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perbaiki kamera terbalik (ROTATE 180)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = model(frame)
        frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    results = model(img)
    result_img = results[0].plot()

    out_path = os.path.join(RESULT_FOLDER, "result.jpg")
    cv2.imwrite(out_path, result_img)

    return render_template(
        "index.html",
        result_image=out_path
    )

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print("🌍 Akses Publik:", public_url)
    app.run(host="0.0.0.0", port=5000, debug=False)
