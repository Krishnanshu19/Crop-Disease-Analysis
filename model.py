import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define dataset path
DATASET_PATH = "data"  # Change this to your dataset folder

# Define categories (same as your folder names)
CATEGORIES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Image size for CNNs like ResNet
IMG_SIZE = 224

# Store images and labels
data = []
labels = []

# Load images from each category
for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    class_index = CATEGORIES.index(category)  # Assign a label (0,1,2,3)

    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
        img_path = os.path.join(folder_path, img_name)
        try:
            # Read and resize image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            
            # Normalize pixels (0 to 1 range)
            img = img / 255.0

            # Append to dataset
            data.append(img)
            labels.append(class_index)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Convert lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split into Train (70%), Validation (20%), and Test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Data Augmentation for Training Data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

# ✅ Save all data in a Pickle file
with open("data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }, f)

print("✅ Preprocessing Completed & Data Saved in data.pkl!")
