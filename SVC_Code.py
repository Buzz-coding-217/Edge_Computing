import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import logging
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For standardizing the data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths and parameters
input_dir = 'dataset/'  # Set this to your dataset directory
model_path = './model.p'
performance_metrics_path = './performance_metrics.pkl'
categories = ['empty', 'not_empty']

# Function to process and flatten images
def process_image(img_path, size=(15, 15)):
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')  # Ensure the image is in RGB format
            img = img.resize(size)  # Resize image
            img_array = np.array(img)  # Convert to numpy array
            return img_array.flatten()  # Flatten the array for the model
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

# Function to prepare the dataset
def prepare_data(input_dir, categories):
    data = []
    labels = []
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        if not os.path.exists(category_path):
            logging.warning(f"Category path {category_path} does not exist.")
            continue
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            processed_img = process_image(img_path)
            if processed_img is not None:
                data.append(processed_img)
                labels.append(category_idx)
    return np.asarray(data), np.asarray(labels)

# Function to train model
def train_model(x_train, y_train):
    classifier = SVC(kernel='sigmoid', C=1.0)
    classifier.fit(x_train, y_train)
    return classifier

# Function for hyperparameter tuning using GridSearchCV
def tune_hyperparameters(x_train, y_train):
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'sigmoid', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Function to standardize data (important for SVM)
def standardize_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

# Main execution function
def main():
    start_time = time.time()
    
    # Prepare data
    logging.info("Preparing the data...")
    data, labels = prepare_data(input_dir, categories)
    
    if len(data) == 0:
        logging.error("No data found. Exiting.")
        return
    
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Standardizing the data
    logging.info("Standardizing the data...")
    x_train, x_test = standardize_data(x_train, x_test)

    # Optional: Use hyperparameter tuning for better performance
    # logging.info("Tuning hyperparameters...")
    # best_estimator = tune_hyperparameters(x_train, y_train)

    # Train the model
    logging.info("Training the model...")
    best_estimator = train_model(x_train, y_train)

    # Test performance
    logging.info("Testing the model...")
    y_pred = best_estimator.predict(x_test)
    score = accuracy_score(y_test, y_pred)

    # Saving the model to disk
    logging.info(f"Saving the model to {model_path}...")
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_estimator, model_file)

    # Saving performance metrics to disk (in this case, accuracy)
    performance_metrics = {'accuracy': score * 100}
    logging.info(f"Saving performance metrics to {performance_metrics_path}...")
    with open(performance_metrics_path, 'wb') as metrics_file:
        pickle.dump(performance_metrics, metrics_file)

    # Calculate time taken and print the results
    timing = time.time() - start_time
    logging.info(f"Time taken: {timing:.2f} seconds")
    logging.info(f"Validation Accuracy: {score * 100:.2f}%")

    logging.info("Model and performance metrics saved to disk.")

if __name__ == "__main__":
    main()
