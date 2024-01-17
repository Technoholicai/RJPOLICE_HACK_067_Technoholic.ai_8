import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(128, 128)):
    X = []
    y = []

    for label, category in enumerate(['real', 'fake']):
        category_path = os.path.join(data_dir, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            img = cv2.resize(img, img_size)  # Resize images to a consistent size
            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess_data(X, y):
    X = X / 255.0  # Normalize pixel values to the range [0, 1]
    y = to_categorical(y, num_classes=2)  # Convert labels to one-hot encoding

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
