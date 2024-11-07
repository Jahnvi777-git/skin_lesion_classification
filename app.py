"""
Skin cancer detection web app

designed and developed by Wisdom ML
"""
# Importing essential python libraries

from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet import preprocess_input
from heatmap import save_and_display_gradcam,make_gradcam_heatmap



os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')


app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\Admin\\OneDrive\\Desktop\\study\\major_project_deployment\\models\\model.h5'

model = load_model(MODEL_PATH)
print("Summary model")
print(model.summary())

        # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

class_labels = ['akeic', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


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



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        file_name=os.path.basename(file_path)
        # Make prediction
        pred,img = model_predict(file_path, model)
        
        last_conv_layer_name = "conv_pw_13"
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
        fname=save_and_display_gradcam(file_path, heatmap)
        
        
        
    return render_template('predict.html',file_name=file_name, heatmap_file=fname,result=pred)




    #this section is used by gunicorn to serve the app on Azure and localhost
if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
