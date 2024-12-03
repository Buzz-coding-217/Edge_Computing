import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time  # Importing time module to track the execution time

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
    classifier = SVC(kernel='sigmoid', C=1.0)
    classifier.fit(x_train, y_train)
    return classifier

def main():
    # Start the timer
    start_time = time.time()

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

    # Saving the model to disk
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_estimator, model_file)

    # Saving performance metrics to disk (in this case, accuracy)
    performance_metrics = {'accuracy': score * 100}
    with open(performance_metrics_path, 'wb') as metrics_file:
        pickle.dump(performance_metrics, metrics_file)

    # Calculate time taken and print the results
    timing = time.time() - start_time  # Elapsed time
    print("Time taken: {:.2f} seconds".format(timing))
    print("Validation Accuracy: {:.2f}%".format(score * 100))

    print("Model and performance metrics saved to disk.")

if __name__ == "__main__":
    main()
