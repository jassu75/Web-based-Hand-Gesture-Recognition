from flask import Flask, render_template, request
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template


app = Flask(__name__)

verbose_name = {
	       
 0: 'No Hand Detected',
 1: 'Open Hand',
 2: 'Peace',
 3: 'Thumb',
 4: 'Okay'
    
           }
 

 

# Select model
model = load_model('hand.h5', compile=False)

model.make_predict_function()

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]
 


 
@app.route("/", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)
    

if __name__ =='__main__':
	app.run()
