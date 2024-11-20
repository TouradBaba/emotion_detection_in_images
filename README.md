# **Emotion detection in images**

This repository hosts the code for a **Facial Emotion Recognition** app that identifies emotions (`Happy`, `Sad`, `Surprise`, and `Neutral`) from images and real-time video streams. The app utilizes a **VGG16 model fine-tuned** for emotion classification, **OpenCV** for image processing, **Haar cascades** for face detection, and a **Flask** backend with an **HTML/CSS** frontend.

---

## **Repository Structure**

```
📁 Models         # Trained deep learning model
📁 Notebooks      # Jupyter notebooks for data exploration, model training, and evaluation
📁 static         # Static files (CSS and Icon)
📁 templates      # HTML templates
📄 .gitattributes # Git LFS configuration
📄 .gitignore     # Git ignore configuration
📄 app.py         # Flask application
📄 requirements.txt # Python dependencies
```
[Icon by Freepik](https://www.freepik.com/icon/smile_2383590)

--- 

## **Dataset**

The dataset used for training the model is the **FER-2013 dataset** (Facial Expression Recognition 2013), which is widely used for emotion detection tasks. You can access the dataset on [Kaggle here](https://www.kaggle.com/datasets/msambare/fer2013).

The data consists of **35,887 labeled 48x48-pixel grayscale images of faces**.

For this project, only the following four emotion classes were used: `Happy`, `Sad`, `Surprise`, and `Neutral`.

---

## **Methods and Techniques**

- **Data Augmentation**: Applied random transformations like rotation, flipping, and zooming to improve the model's generalization and diversify the training dataset.

- **Overfitting Prevention**:
  - **Dropout layers**: Added to prevent co-adaptation of neurons and reduce overfitting.
  - **L2 Regularization**: Used in dense layers to penalize large weights.

- **Batch Normalization**: Incorporated to stabilize training and accelerate convergence by normalizing intermediate layer outputs.

- **Callbacks**:
  - **EarlyStopping**: Monitors validation accuracy and halts training when performance stops improving.
  - **ReduceLROnPlateau**: Dynamically adjusts the learning rate to prevent stagnation.
  - **ModelCheckpoint**: Saves the best-performing model during training.

---

## **Model Performance**

The model achieves an overall classification accuracy of 79.9%, with detailed metrics:

| **Emotion/Metric** | **Precision** | **Recall** | **F1-Score** | **Support** |  
|--------------------|---------------|------------|--------------|-------------|  
| **Happy**          | 0.91          | 0.89       | 0.90         | 1774        |  
| **Sad**            | 0.69          | 0.76       | 0.72         | 1247        |  
| **Surprise**       | 0.86          | 0.86       | 0.86         | 831         |  
| **Neutral**        | 0.72          | 0.67       | 0.69         | 1233        |  

---

## **User Interface**

### **Tools Used**:
- **Flask**: A Python web framework used for building the backend of the application. it handles image uploads, real-time video streams, and invokes the emotion recognition model. 
- **HTML/CSS**: These are used for the frontend of the application. HTML structures the web pages and CSS is used to style them, to create a user-friendly interface for uploading images, streaming video, and displaying results.

The UI consists of a **Home Page** from which users can navigate to the following pages:  
  - **Real-Time Video Stream**: For live emotion recognition.  
  - **Image Upload**: To upload an image and receive the detected emotion label.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TouradBaba/emotion_detection_in_images.git
   cd emotion_detection_in_images
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**: 

    Once the Flask application is running, you can access it by navigating to the following address in your web browser:
    
    ```
    http://127.0.0.1:5000
    ```

---