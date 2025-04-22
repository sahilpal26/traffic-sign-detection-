import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# === Traffic Sign Labels ===
sign_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No overtaking', 'No overtaking trucks', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children', 'Bicycles', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no overtaking', 'End of no overtaking trucks'
]

# === Load Model ===
model = load_model("traffic_sign_recognition_model.h5")  # Update the path if needed

# === Prepare the image for prediction ===
def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((30, 30))  # Resize to match training size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# === Predict the traffic sign ===
def predict_sign(image_path):
    img = prepare_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_label = sign_names[predicted_class]
    return predicted_label

# === Display result ===
def display_prediction(image_path):
    img = Image.open(image_path)
    predicted_label = predict_sign(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# === Image picker and prediction loop ===
def run_image_picker_loop():
    root = tk.Tk()
    root.withdraw()  # Hide main window

    while True:
        file_path = filedialog.askopenfilename(
            title="Select a traffic sign image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            print("❌ No file selected. Exiting.")
            break

        print(f"📂 Selected: {file_path}")
        display_prediction(file_path)

        again = messagebox.askyesno("Choose Another?", "Do you want to predict another image?")
        if not again:
            break

# === Start ===
run_image_picker_loop()
