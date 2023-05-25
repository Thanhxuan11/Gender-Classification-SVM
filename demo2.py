import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from joblib import dump, load
model = load('D:/gender-classification-dataset/Model/model2.sav')

def checking(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("D:/gender-classification-dataset/haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Preprocess image
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = cv2.GaussianBlur(face, (5, 5), 0)
        face = face / 255.0
        features = hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)

        # Predict gender
        classes = model.predict(features.reshape(1, -1))
        probability=model.predict_proba(features.reshape(1, -1))  
        if classes[0] < 0.5:
            gender = f"Male ({probability[0][1]*100:.2f}%)" 
        else:
            gender = f"Female  ({probability[0][0]*100:.2f}%)"

        # Draw rectangle around face and display gender prediction
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def webcam():
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier("D:/gender-classification-dataset/haarcascade_frontalface_default.xml")
    Categories = ['female', 'male']
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Predict gender for each face
        for (x, y, w, h) in faces:
            # Preprocess image
            # Preprocess image
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = cv2.GaussianBlur(face, (5, 5), 0)
            face = face / 255.0
            features = hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)

            # Predict gender
            classes = model.predict(features.reshape(1, -1))
            probability=model.predict_proba(features.reshape(1, -1))  
            if classes[0] < 0.5:
                gender = f"Male ({probability[0][1]*100:.2f}%)" 
            else:
                gender = f"Female  ({probability[0][0]*100:.2f}%)"

            # Draw rectangle around face and display gender prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display image
        cv2.imshow('frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()

# Tạo cửa sổ
window = tk.Tk()

# Thiết lập tiêu đề
window.title("Gender Classifier")


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        checking(file_path)


button = tk.Button(window, text="Open Image", command=open_image)
button1 = tk.Button(window, text="Webcam", command=webcam)
# Loop chạy form

button.pack()
button1.pack()
window.mainloop()