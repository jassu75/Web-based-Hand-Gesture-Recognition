# Web-based Sign Language Recognition

**Note:** A public URL to access this project does not exist as it is a Machine Learning project. Hosting it on Google Cloud using App Engine is not possible and Cloud Run is expensive. You can access the project and run it on your local system.

## Demo Link
[Watch on YouTube](https://youtu.be/MP3bIqmON_k?si=UmykhsyY2FekzWft)

## Dataset

The dataset has been taken from Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/jayaprakashpondy/hand-gesture-dataset)

**The model has already been trained and the results are stored in the `hands.h5` file. You do not need this unless you want to retrain the model.**

## Steps

1. Clone the repository  
2. Install the dependencies listed in `requirements.txt`  
3. Download the dataset from Kaggle and add it to your project (only if you want to retrain the model)  
4. Run the app with:  
   ```bash
   python main.py

## Model Capabilities

The model is capable of recognizing the following 5 hand gestures:
- 👍 Thumbs Up  
- 👌 OK  
- ✌️ Peace  
- ✋ Open Hand  
- 🚫 No Hand 

---

#### 🛠️ Technologies Used:
- Flask  
- Xception Model  
- HTML, CSS  
- JavaScript  
- Bootstrap

---

## Screenshots

### Main Page
Upload the image to detect the sign the image shows.

<img width="1860" height="861" alt="image" src="https://github.com/user-attachments/assets/34efab71-fce0-4c74-95f0-ebdea4f08788" />

----------------------------------------------------------------------------------------------------------------------------------------

<img width="1839" height="827" alt="image" src="https://github.com/user-attachments/assets/4f87cd9f-cdf6-494c-b0b8-568d2e9f73b7" />

### Prediction Page
The prediction for the image is displayed.

<img width="1806" height="790" alt="image" src="https://github.com/user-attachments/assets/17702d95-a99b-407f-bb8a-9f4b27689a1a" />




