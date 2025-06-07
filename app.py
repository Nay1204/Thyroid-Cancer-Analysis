import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import joblib

# Load trained model and label encoder
model = load_model('thyroid_classifier.h5')  # Your trained dense classifier
label_encoder = joblib.load('label_encoder.pkl')  # Save this separately using joblib

# Load VGG16 feature extractor
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=vgg_base.input, outputs=vgg_base.output)

def preprocess_image(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def extract_features(img_array):
    img_batch = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_batch)
    return features.reshape(1, -1)

st.title("Thyroid Image Classifier")
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    img = preprocess_image(uploaded_file)
    features = extract_features(img)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    st.subheader("Prediction:")
    st.success(f"Predicted Category: **{predicted_label[0]}**")