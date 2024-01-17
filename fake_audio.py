# Import necessary libraries
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

# Function to extract MFCC features from audio files
def extract_mfcc(audio_path):
    wave, sr = librosa.load(audio_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, hop_length=256, n_fft=512)
    return mfcc

# Load and process the dataset
def load_and_process_dataset(data_dir):
    mfcc_features = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                mfcc = extract_mfcc(file_path)
                mfcc_features.append(mfcc)
                labels.append(label)

    # Convert labels to numerical representation
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Pad or truncate MFCC features to a fixed length (optional)
    max_pad_len = max([mfcc.shape[1] for mfcc in mfcc_features])
    mfcc_features = np.array([np.pad(mfcc, pad_width=((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant') for mfcc in mfcc_features])

    # Convert the data to numpy arrays
    encoded_labels = to_categorical(encoded_labels)

    return mfcc_features, encoded_labels

# Set the path to your dataset
data_directory = "C:/Users/vansh/Desktop/DEEP/audiodata-20240114T182524Z-001"

# Load and process the dataset
mfcc_features, labels = load_and_process_dataset(data_directory)

# For binary classification, we assume that the dataset has only two classes (0 and 1)
# If you have more than two classes, modify this accordingly
num_classes = 2

# Convert the labels to binary (0 or 1)
binary_labels = np.argmax(labels, axis=1)
binary_labels = np.where(binary_labels == 0, 0, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfcc_features, binary_labels, test_size=0.2, random_state=42)

# Build the model for binary classification
input_shape = (X_train.shape[1], X_train.shape[2])  # Adjust input shape based on your data

model = Sequential()
model.add(LSTM(128, input_shape=input_shape))
model.add(Dense(1, activation='sigmoid'))  # Use sigmoid activation for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Use binary crossentropy for binary classification

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=5, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy: {accuracy}")

# Save the model
model.save('audio_model_binary.h5')
