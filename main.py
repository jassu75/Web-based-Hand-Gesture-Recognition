from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/tests'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

verbose_name = {
    0: 'No Hand Detected',
    1: 'Open Hand', 
    2: 'Peace',
    3: 'Thumb',
    4: 'Okay'
}

# Load model
model = load_model('hand.h5', compile=False)
model.make_predict_function()

def predict_label(img_path):
    """Predict hand gesture from image path"""
    try:
        # Load and preprocess image
        test_image = image.load_img(img_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = test_image.reshape(1, 224, 224, 3)
        
        # Make prediction
        predict_x = model.predict(test_image)
        classes_x = np.argmax(predict_x, axis=1)
        
        return verbose_name[classes_x[0]]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error in prediction"

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return render_template("index.html", error="No file selected")
        
        img = request.files['my_image']
        
        if img.filename == '':
            return render_template("index.html", error="No file selected")
        
        # Save uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)
        
        # Make prediction
        predict_result = predict_label(img_path)
        
        return render_template("prediction.html", 
                             prediction=predict_result, 
                             img_path=img_path)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)