import cv2
import numpy as np
from sklearn.cluster import KMeans
from deepface import DeepFace
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk

# Define functions
def preprocess_image(image, target_size=(416, 416)):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def detect_car_color(car_region):
    car_region_rgb = car_region[:, :, ::-1]  # Convert BGR to RGB
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(car_region_rgb.reshape(-1, 3))
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def count_cars(outputs, width, height):
    car_count = 0
    car_regions = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 2 and confidence > 0.5:  # Assuming class_id 2 is 'car'
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                car_regions.append((x, y, w, h))
                car_count += 1
    return car_count, car_regions

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def predict_gender_deepface(face_region):
    analysis = DeepFace.analyze(face_region, actions=['gender'], enforce_detection=False)
    gender = analysis['gender']
    return gender

def object_detection(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs

def count_other_vehicles(outputs, width, height):
    other_vehicle_count = 0
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id != 2 and confidence > 0.5:  # Assuming class_id 2 is 'car'
                other_vehicle_count += 1
    return other_vehicle_count

def load_and_process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Image not loaded. Check the path and file.")
        return None
    processed_image = preprocess_image(image)
    outputs = object_detection(processed_image)
    height, width, _ = image.shape
    car_count, car_regions = count_cars(outputs, width, height)
    car_colors = []
    for (x, y, w, h) in car_regions:
        car_region = image[y:y+h, x:x+w]
        color = detect_car_color(car_region)
        car_colors.append(color)
    faces = detect_faces(image)
    male_count, female_count = 0, 0
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        gender = predict_gender_deepface(face_region)
        if gender == "Man":
            male_count += 1
        else:
            female_count += 1
    other_vehicle_count = count_other_vehicles(outputs, width, height)
    for (x, y, w, h) in car_regions:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        gender = predict_gender_deepface(image[y:y+h, x:x+w])
        label = f"{gender}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image, car_count, car_colors, male_count, female_count, other_vehicle_count

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        image, car_count, car_colors, male_count, female_count, other_vehicle_count = load_and_process_image(file_path)
        display_image(image)
        result_text.set(f"Cars: {car_count}, Males: {male_count}, Females: {female_count}, Other Vehicles: {other_vehicle_count}")

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.configure(image=image)
    panel.image = image

# Load pre-trained models
net = cv2.dnn.readNet("C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 3\\yolov3.weights", "C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 3\\yolov3.cfg")
with open("C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 3\\coco.names", "r") as f:
    classes = f.read().strip().split("\n")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the main window
root = tk.Tk()
root.title("Traffic Analysis")
root.geometry("800x600")

result_text = StringVar()

# Create GUI elements
open_button = Button(root, text="Open Image", command=open_file)
open_button.pack()

panel = Label(root)
panel.pack()

result_label = Label(root, textvariable=result_text)
result_label.pack()

root.mainloop()