from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocesser import ImagePreprocessor
from PIL import Image

app = Flask(__name__)

# Load your Keras model
model = load_model('ResNet50.keras')

# Define the folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded image
    imagefile = request.files['imagefile']
    
    # Save the image in the 'static/uploads' directory
    filename = imagefile.filename
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagefile.save(image_path)

    # Initialize the image preprocessor
    preprocessor = ImagePreprocessor()

    # Preprocess the image (resize, normalize, augment)
    img_array = preprocessor.preprocess(image_path)

    # Ensure the image is in the shape [batch_size, height, width, channels] for Keras
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes [1, height, width, 3]

    # Make prediction using the model
    prediction = model.predict(img_array)  # Get the prediction from the model

    # Get the predicted class (0 or 1)
    predicted_class = (prediction > 0.5).astype("int32").flatten()[0]

    # Get the confidence of the prediction
    confidence = prediction[0][0]  # For binary classification, assuming output is (1,1)

    # Print the result
    if predicted_class == 0:
        result = f"There is no tumor ({(1-confidence) * 100:.2f}% confidence)"
    else:
        result = f"There is a tumor ({confidence * 100:.2f}% confidence)"

    # Return the result to the HTML template
    return render_template('index.html', prediction=result, image=filename)
#just comment
if __name__ == '__main__':
    app.run(port=3000, debug=True)
