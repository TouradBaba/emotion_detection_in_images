import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import warnings
import time

# Suppress specific warning
warnings.filterwarnings("ignore", message="The use_column_width parameter has been deprecated")

# Load the trained model
model = load_model('model.h5')

# The class labels
class_labels = ['Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Streamlit app layout
st.set_page_config(page_title="Emotion Detection App")
st.title("Emotion Detection App")
st.write("This app lets you detect emotions from an uploaded image or real-time camera feed.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ("Upload Image", "Real-Time Detection"))

if page == "Upload Image":
    st.header("Upload an Image")

    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            st.subheader("No face detected in the uploaded image.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (96, 96))
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_resized = img_to_array(face_resized)
                face_resized = np.expand_dims(face_resized, axis=0) / 255.0

                # Predict emotion
                prediction = model.predict(face_resized)
                max_index = np.argmax(prediction[0])
                emotion = class_labels[max_index]

                # Add label text above the image
                label_position = (x, y - 10)
                cv2.putText(image, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

            st.subheader(f"Detected Emotion: {emotion}")
            st.image(image, channels="BGR", use_container_width=True)

else:
    st.header("Real-Time Emotion Detection")
    st.write("Click 'Start Detection' to begin real-time emotion detection. This feature may take time to load.")

    # Button to toggle start/stop detection
    if 'is_detecting' not in st.session_state:
        st.session_state.is_detecting = False

    if st.button("Start/Stop Detection"):
        st.session_state.is_detecting = not st.session_state.is_detecting

    if st.session_state.is_detecting:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        st.write("Real-time detection started. Press the same button to stop.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                # Display a message when no face is detected
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.putText(frame_rgb, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
                frame_placeholder.image(frame_rgb, use_container_width=True)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = frame[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (96, 96))
                    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_resized = img_to_array(face_resized)
                    face_resized = np.expand_dims(face_resized, axis=0) / 255.0

                    # Predict the emotion
                    prediction = model.predict(face_resized)
                    max_index = np.argmax(prediction[0])
                    emotion = class_labels[max_index]

                    # Display the emotion label on the frame
                    label_position = (x, y - 10)
                    cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

                # Display the frame in Streamlit app
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_container_width=True)

            time.sleep(0.05)

            # Stop detection if the button is pressed again
            if not st.session_state.is_detecting:
                break

        cap.release()
        frame_placeholder.empty()
