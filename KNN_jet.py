import numpy as np
from PIL import Image
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import time

# Path to the dataset directory
data_dir = './dataset2'

# Function to extract label from filename
def extract_label(filename):
    # Remove the file extension
    base_name = os.path.splitext(filename)[0]
    # Remove numeric characters
    label = ''.join(filter(str.isalpha, base_name)).strip()
    return label

# Function to extract features from image
def extract_features(image):
    # Resize image to a fixed size
    image = image.resize((64, 64))  # Resize to 64x64 pixels
    # Convert image to grayscale
    gray_image = np.array(image.convert('L'))
    # Flatten the image array
    features = gray_image.flatten()
    return features

# Start timing
start = time.time()

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
        print("Error processing {}: {}".format(img_path, e))

# Check if any images were processed
if not features:
    raise ValueError("No images were processed. Check if the dataset directory has images.")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reducing to 50 components
features_pca = pca.fit_transform(features)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
features_train, features_val, labels_train, labels_val = train_test_split(
    features_pca, labels_encoded, test_size=0.2, random_state=0
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=15)
model.fit(features_train, labels_train)

# Test performance
from sklearn.metrics import accuracy_score
y_pred = model.predict(features_val)
score = accuracy_score(labels_val, y_pred)

# End timing
timing = time.time() - start

# Print timing and accuracy
print("Time taken: {:.2f} seconds".format(timing))
print("Validation Accuracy: {:.2f}%".format(score * 100))

# Save the model and label encoder
with open('./model_knn.p', 'wb') as f:
    pickle.dump((model, label_encoder, pca), f)

print("Model saved as 'model_knn.p'")

