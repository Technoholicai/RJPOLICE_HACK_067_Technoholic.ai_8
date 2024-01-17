from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import librosa
import numpy as np

app = Flask(__name__)

# Function to extract MFCC features from audio files
def extract_mfcc(audio_path):
    wave, sr = librosa.load(audio_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, hop_length=256, n_fft=512)
    return mfcc

# Load the trained model
model = load_model('C:/Users/vansh/Desktop/d/deep_ui/audio_model_binary.h5')  # Replace with your model file

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

        prediction = model.predict(reshaped_mfcc)
        predicted_probability = prediction[0, 0]  # Probability of being class 0

        print("Raw prediction values:", prediction)  # Add this line for debugging

        return "Real Audio" if predicted_probability < threshold else "Fake Audio"
    except Exception as e:
        print("Error processing audio file:", str(e))  # Print the error details
        return 'Error'

@app.route('/')
def index():
    return render_template('result_audio.html')

# Route to handle file upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        print("Received file:", file.filename)  # Add this line for debugging

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            result = process_audio(file)
            if result == 'Error':
                return jsonify({'error': 'Error processing the audio file'})
            else:
                return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
