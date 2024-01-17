import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Function to extract frames from videos
def extract_frames(video_path, num_frames=30, resize_dim=(128, 128)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if ret and i in frame_indices:
            frame = cv2.resize(frame, resize_dim)
            frames.append(frame.flatten())  # Flatten the frame before stacking

    cap.release()
    return np.stack(frames)

# Load video data and labels
def load_data(data_dir='C:/Users/vansh/Desktop/DEEP/dfdc/dfdc_train_part_46', num_videos=2200):
    videos = []
    labels = []

    for label in ['real', 'fake']:
        label_dir = os.path.join(data_dir, label)
        for video_file in os.listdir(label_dir)[:num_videos // 2]:
            video_path = os.path.join(label_dir, video_file)
            frames = extract_frames(video_path)
            videos.append(frames)
            labels.append(1 if label == 'fake' else 0)

    videos = np.stack(videos)
    labels = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(videos, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test