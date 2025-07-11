# Web based Sign Language Recognition 

**Note** : A public URL to access this project does not exist as it is a Machine Learning project. So hosting it on Google Cloud using App Engine is not possible and expensive via Cloud Run. You can access the project and run it on your local system 

## Dataset

The dataset has been taken from Kaggle
[Kaggle Dataset Link](https://www.kaggle.com/datasets/jayaprakashpondy/hand-gesture-dataset)

**The model has already been trained and results stored in hands.h5 file. You will not need it unless you want to retrain the model**


## Steps

1) Clone the repository 
2) Install the dependencies in requirements.txt
3) Download the dataset from kaggle website and add it to your project (If you want to retrain the model)
4) python main.py to run the app

## Model Capabilities

The model can detect 5 sign languages in images
1) No Hand Detected
2) Open Hand
3) Peace
4) Thumb
5) okay

## Screenshots

### Main Page
Upload the image to detect the sign the image is. 

<img width="1860" height="861" alt="image" src="https://github.com/user-attachments/assets/34efab71-fce0-4c74-95f0-ebdea4f08788" />

<img width="1839" height="827" alt="image" src="https://github.com/user-attachments/assets/4f87cd9f-cdf6-494c-b0b8-568d2e9f73b7" />

### Prediction Page
The prediction for the image is displayed

<img width="1806" height="790" alt="image" src="https://github.com/user-attachments/assets/17702d95-a99b-407f-bb8a-9f4b27689a1a" />




