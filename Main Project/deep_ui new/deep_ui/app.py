from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import librosa
from tensorflow.keras.applications.resnet50 import preprocess_input
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)

# Load the pre-trained image model
image_model = tf.keras.models.load_model('C:/Users/vansh/Desktop/deep_ui new/deep_ui new/deep_ui/cnn_model.h5')

# Load the pre-trained video model (replace 'lstm_model.h5' with your actual video model file)
video_model = tf.keras.models.load_model('C:/Users/vansh/Desktop/deep_ui new/deep_ui new/deep_ui/lstm_model.h5')

audio_model = tf.keras.models.load_model('C:/Users/vansh/Desktop/deep_ui new/deep_ui new/deep_ui/audio_model_binary.h5')  # Replace with your audio model file

# Define the allowed file extensions for images and videos
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav'}

# Function to check if the file extension is allowed for images
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Function to check if the file extension is allowed for videos
def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Function to check if the file extension is allowed for audio
def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def extract_mfcc(audio_path):
    wave, sr = librosa.load(audio_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, hop_length=256, n_fft=512)
    return mfcc

def process_audio(file_path, threshold=0.65):
    try:
        mfcc = extract_mfcc(file_path)
        max_pad_len = 112581  # Update with the correct value

        print("Original MFCC shape:", mfcc.shape)  # Add this line for debugging

        # Normalize MFCC features
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')

        print("Processed MFCC shape:", mfcc.shape)  # Add this line for debugging

        reshaped_mfcc = np.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))

        # Make prediction
        if reshaped_mfcc.shape[2] != 112581:  # Check if reshaped_mfcc has the expected size
            raise ValueError("Unexpected size for reshaped_mfcc. Check the value of max_pad_len.")

        prediction = audio_model.predict(reshaped_mfcc)
        predicted_probability = prediction[0, 0]  # Probability of being class 0

        print("Raw prediction values:", prediction)  # Add this line for debugging

        return "This Audio is Real" if predicted_probability < threshold else "This Audio is Fake"
    except Exception as e:
        print("Error processing audio file:", str(e))  # Print the error details
        return 'Error'
    
# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for image prediction
@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Handle the POST request for image prediction

        if 'file' not in request.files:
            return jsonify({'result': 'No file selected'})

        file = request.files['file']

        if not allowed_image_file(file.filename):
            return jsonify({'result': 'Invalid image file extension'})

        img = Image.open(file)
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)

        prediction = image_model.predict(np.expand_dims(img_array, axis=0))
        result = 'This Image is Real' if prediction[0][0] > 0.5 else 'This Image is Fake'
        
         # Convert the image to base64 for display
        _, img_encoded = cv2.imencode('.jpg', img_array)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({'result': result, 'image_data': img_base64})


    # If the request method is GET, you can return a default response or render the HTML page.
    return render_template('result_image.html')

@app.route('/predict_video', methods=['GET', 'POST'])
def predict_video():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'result': 'No file selected'})

            file = request.files['file']

            if not allowed_video_file(file.filename):
                return jsonify({'result': 'Invalid video file extension'})

            # Save the video file to a temporary directory
            video_filename = secure_filename(file.filename)
            video_path = os.path.join('C:/Users/vansh/Desktop/deep_ui new/deep_ui new/deep_ui/uploads', video_filename)
            file.save(video_path)

            # Use OpenCV for video processing
            cap = cv2.VideoCapture(video_path)
            frames = []

            # Read only the first 30 frames
            max_frames = 30
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the frame to (128, 128) and preprocess using ResNet50 preprocessing
                frame = cv2.resize(frame, (128, 128))
                frame = preprocess_input(frame)
                frames.append(frame)

            cap.release()

            # Convert frames to a single numpy array
            frames_array = np.array(frames)

            # Reshape the frames to match the expected input shape of the LSTM model
            frames_reshaped = frames_array.reshape((1, 30, 49152))

            # Predict using the LSTM model
            video_prediction = video_model.predict(frames_reshaped)

            # Calculate the percentage of frames classified as 'Real'
            threshold_percentage = 0.65
            real_frame_percentage = np.sum(video_prediction > 0.5) / len(video_prediction)

            # Compare the percentage with the threshold
            result = 'This Video is Real' if real_frame_percentage > threshold_percentage else 'This Video is Fake'

            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'result': f'Error: {str(e)}'})

    return render_template('result_video.html', result=None)

# Route for audio prediction
@app.route('/predict_audio', methods=['GET', 'POST'])
def predict_audio():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})

            file = request.files['file']
            print("Received file:", file.filename)  # Add this line for debugging

            if file.filename == '':
                return jsonify({'error': 'No selected file'})

            # Save the uploaded file to the upload folder
            audio_filename = secure_filename(file.filename)
            audio_path = os.path.join('C:/Users/vansh/Desktop/deep_ui new/deep_ui new/deep_ui/uploads', audio_filename)
            file.save(audio_path)  # Fix: Use audio_path instead of file_path

            # Process the audio file and get the result
            result = process_audio(audio_path)  # Fix: Use audio_path instead of file_path

            # Optionally, you can remove the uploaded file after processing
            os.remove(audio_path)  # Fix: Use audio_path instead of file_path

            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'error': str(e)})
    return render_template('result_audio.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
