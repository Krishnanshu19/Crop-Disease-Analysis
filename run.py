import os
import kagglehub

# Step 1: Download the dataset
print("Downloading dataset from Kaggle using kagglehub...")
dataset_path = kagglehub.dataset_download("smaranjitghose/corn-or-maize-leaf-disease-dataset")
print("Dataset downloaded to:", dataset_path)

# Step 2: Run model.py
print("Running model.py...")
os.system("python model.py")

# Step 3: Run train_model.py
print("Training the model with train_model.py...")
os.system("python train_model.py")

print("All tasks completed successfully!")
