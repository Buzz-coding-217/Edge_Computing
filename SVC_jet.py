import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define paths and parameters
input_dir = 'dataset/'  # Set this to your dataset directory
model_path = './model.p'
performance_metrics_path = './performance_metrics.pkl'
categories = ['empty', 'not_empty']

# Function to process and flatten images
def process_image(img_path, size=(15, 15)):
    with Image.open(img_path) as img:
        img = img.convert('RGB')  # Ensure the image is in RGB format
        img = img.resize(size)  # Resize image
        img_array = np.array(img)  # Convert to numpy array
        return img_array.flatten()  # Flatten the array for the model

def prepare_data(input_dir, categories):
    data = []
    labels = []
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            data.append(process_image(img_path))
            labels.append(category_idx)
    return np.asarray(data), np.asarray(labels)

def train_model(x_train, y_train):
    classifier = SVC()
    classifier.fit(x_train, y_train)
    return classifier

def main():
    if os.path.exists(model_path) and os.path.exists(performance_metrics_path):
        # Load the pre-trained model and performance metrics
        with open(model_path, 'rb') as f:
            best_estimator = pickle.load(f)
        with open(performance_metrics_path, 'rb') as f:
            performance_metrics = pickle.load(f)
        print("Model and performance metrics loaded from disk.")
        print("Previous accuracy: {performance_metrics['accuracy']:.2f}%")
    else:
        # Prepare data
        data, labels = prepare_data(input_dir, categories)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        # Train the model
        best_estimator = train_model(x_train, y_train)

        # Test performance
        y_pred = best_estimator.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print('{score * 100:.2f}% of samples were correctly classified')

        # Save the model and performance metrics
        with open(model_path, 'wb') as f:
            pickle.dump(best_estimator, f)
        with open(performance_metrics_path, 'wb') as f:
            pickle.dump({'accuracy': score * 100}, f)
        print("Model and performance metrics saved to disk.")

if __name__ == "__main__":
    main()
