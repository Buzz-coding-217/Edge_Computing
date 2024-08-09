import os
import re
import numpy as np
from PIL import Image
from skimage import color, feature, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Path to the dataset directory
data_dir = './dataset2'

# Function to extract label from filename
def extract_label(filename):
    # Remove the file extension
    base_name = os.path.splitext(filename)[0]
    # Remove numeric characters
    label = re.sub(r'\d+', '', base_name).strip()
    return label

# Function to extract features from image
def extract_features(image):
    # Resize image to a fixed size
    image = image.resize((224, 224))
    # Convert image to grayscale
    gray_image = color.rgb2gray(np.array(image))
    # Extract Histogram of Oriented Gradients (HOG) features
    hog_features, _ = feature.hog(
        gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True
    )
    return hog_features

# Load images and extract features
features = []
labels = []

for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    try:
        img = Image.open(img_path)
        img_features = extract_features(img)
        features.append(img_features)
        
        # Extract label from the filename
        label = extract_label(img_name)
        labels.append(label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Check if any images were processed
if not features:
    raise ValueError("No images were processed. Check if the dataset directory has images.")

# Split data into training and validation sets
features_train, features_val, labels_train, labels_val = train_test_split(
    features, labels, test_size=0.2, random_state=0
)

# Train model
model = RandomForestClassifier(random_state=0)
model.fit(np.array(features_train), labels_train)

# Test performance
y_pred = model.predict(np.array(features_val))
score = accuracy_score(labels_val, y_pred)

print(f"Validation Accuracy: {score}")

# Save the model
import pickle
with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
