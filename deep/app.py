
import os
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, VIDEOS
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Configure the file uploads
photos = UploadSet("photos", IMAGES)
videos = UploadSet("videos", VIDEOS)

app.config["UPLOADED_PHOTOS_DEST"] = "uploads/photos"
app.config["UPLOADED_VIDEOS_DEST"] = "uploads/videos"

configure_uploads(app, (photos, videos))

# Load the pre-trained deep fake detection model
# Replace 'path_to_model' with the actual path to your model
model = tf.keras.models.load_model("path_to_model")

def detect_deep_fake(image_path):
    # Add your deep fake detection logic here
    # This is a placeholder, and you may need to modify it based on your model
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize input image based on your model requirements
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values

    prediction = model.predict(image)
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        photo_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], photo.filename)
        photo.save(photo_path)
        result = detect_deep_fake(photo_path)
        os.remove(photo_path)  # Remove the uploaded file after processing
        return render_template("result.html", result=result)

    if request.method == "POST" and "video" in request.files:
        video = request.files["video"]
        video_path = os.path.join(app.config["UPLOADED_VIDEOS_DEST"], video.filename)
        video.save(video_path)
        # Add video deep fake detection logic here
        # You may need to use a library like OpenCV to process the video
        result = "Video detection result"
        os.remove(video_path)  # Remove the uploaded file after processing
        return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
