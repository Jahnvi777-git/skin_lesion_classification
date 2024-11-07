from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet import preprocess_input
from heatmap import save_and_display_gradcam,make_gradcam_heatmap


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FEEDBACK_FILE'] = 'feedback.json'  # For storing feedback data

# Load the model
MODEL_PATH = 'C:\\Users\\Admin\\OneDrive\\Desktop\\study\\major_project_deployment\\models\\model.h5'
model = load_model(MODEL_PATH)

# Class dictionary for predictions
class_dict = {
    0: "Actinic Keratosis and Intraepithelial Carcinoma (Bowen's disease)",  # akiec
    1: "Basal Cell Carcinoma",  # bcc
    2: "Benign Keratosis (solar lentigines / seborrheic keratoses / lichen planus-like keratoses)",  # bkl
    3: "Dermatofibroma",  # df
    4: "Melanoma",  # mel
    5: "Melanocytic Nevus",  # nv
    6: "Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage)"  # vasc
}


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


def model_predict(img_path, model):
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to target size
    img = cv2.resize(img, (224, 224))
    # Preprocess the image (normalize, expand dimensions, and apply model-specific preprocessing)
    img = preprocess_input(img)  # Ensure you're using the right preprocess_input
    img = np.expand_dims(img, axis=0)
    # Model prediction
    preds = model.predict(img)[0]

    # Format the predictions with class labels and probabilities
    prediction = sorted(
        [(class_dict[i], round(j * 100, 2)) for i, j in enumerate(preds)],
        reverse=True,
        key=lambda x: x[1],
    )
    return prediction, img

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    f = request.files['file']

        # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    file_name=os.path.basename(file_path)
    
    
    # Make a prediction

    pred, img = model_predict(file_path, model)
    last_conv_layer_name = "conv_pw_13"
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
    fname=save_and_display_gradcam(file_path, heatmap)

    # Render prediction template with feedback form
    return render_template('predict_temp.html',file_name=file_name, heatmap_file=fname,result=pred,class_dict=class_dict)


@app.route('/feedback', methods=['POST'])
def feedback():
    # Get the prediction feedback
    feedback_data = request.json

    # Load existing feedback data
    if os.path.exists(app.config['FEEDBACK_FILE']):
        with open(app.config['FEEDBACK_FILE'], 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Append new feedback data
    existing_data.append(feedback_data)

    # Save updated feedback data
    with open(app.config['FEEDBACK_FILE'], 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(feedback_data)

    # Call retraining if the feedback indicates misclassification
    if feedback_data.get('feedback') == 'incorrect':
        retrain_model(feedback_data, MODEL_PATH)

    return 'Feedback received, retraining in progress', 200

def retrain_model(feedback_data, model_path):
    # Load the model
    model = load_model(model_path)

    # Extract the correct class from the feedback data
    correct_class_name = feedback_data.get('correct_class')

    # Check if correct class is provided
    if correct_class_name:
        # Preprocess the image for retraining
        file_name = feedback_data.get("file_name")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Get the correct label
        label_index = list(class_dict.values()).index(correct_class_name)
        labels = to_categorical([label_index], num_classes=len(class_dict))

        # Use data augmentation to increase dataset size
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Use callbacks for early stopping and learning rate reduction
        callbacks = [
            EarlyStopping(monitor='loss', patience=2, verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
        ]

        # Compile the model with a stable learning rate
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Retrain with augmented data
        datagen.fit(img)  # Augment the single image
        augmented_data = datagen.flow(img, labels, batch_size=1)

        # Retrain with augmented dataset
        model.fit(
            augmented_data,
            epochs=50,
            callbacks=callbacks,
            steps_per_epoch=50  # Adjust this to get more augmented samples
        )

        # Save the updated model with a unique identifier
        new_model_path = f'retrained_models\\model_retrained_{int(time.time())}.h5'
        model.save(new_model_path)

        print(f"Model retrained and saved to {new_model_path}")


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
