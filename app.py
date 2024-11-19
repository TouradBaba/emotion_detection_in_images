from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('Models/model.h5')

# Class labels
class_labels = ['Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture object
camera = None


def detect_faces_and_emotions(image):
    """
    Detect faces in the image, predict their emotions, and draw bounding boxes.
    Returns the processed image and the detected emotion.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No face detected"

    detected_emotion = None
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face and preprocess for the model
        face = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (96, 96))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = img_to_array(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0

        # Predict emotion
        prediction = model.predict(face_resized)
        max_index = np.argmax(prediction[0])
        detected_emotion = class_labels[max_index]

        # Add emotion label above the rectangle
        label_position = (x, y - 10)
        cv2.putText(image, detected_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, detected_emotion


@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Handle image upload and display the detected emotion.
    """
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return redirect(url_for('upload'))

        # Read the uploaded file and convert to numpy array
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect faces and emotions
        processed_image, emotion = detect_faces_and_emotions(image)

        # Convert image to base64 for rendering in HTML
        _, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('upload.html', emotion=emotion, image=image_base64)

    return render_template('upload.html', emotion=None)


@app.route('/real_time')
def real_time():
    """
    Render the Real-Time Detection page.
    """
    return render_template('real_time.html', stream=False)


@app.route('/start', methods=['POST'])
def start_detection():
    """
    Start real-time emotion detection.
    """
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    return render_template('real_time.html', stream=True)


@app.route('/video_feed')
def video_feed():
    """
    Generate frames for the video feed.
    """
    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Detect faces and emotions
            frame, _ = detect_faces_and_emotions(frame)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop', methods=['POST'])
def stop_detection():
    """
    Stop real-time emotion detection and release the camera.
    """
    global camera
    if camera:
        camera.release()
        camera = None

    return redirect(url_for('real_time'))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
