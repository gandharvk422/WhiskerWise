import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import os

# Set the page configuration
st.set_page_config(page_title="WhiskerWise - Know Your Pet Instantly", page_icon=":dog:")

st.header("WhiskerWise")
st.subheader("Know Your Pet Instantly")

# Path to the saved model
MODEL_PATH = "model.h5"

# Load the pretrained model
try:
    if os.path.exists(MODEL_PATH):
        cnn = load_model(MODEL_PATH, compile=False)  # Add compile=False to avoid unnecessary compilation
        st.markdown("Model loaded successfully.")
    else:
        st.error("Model not found. Please upload the model file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown("Now you can test the model by uploading an image")

st.sidebar.markdown("# Upload a pic")
file = st.sidebar.file_uploader(label="Upload a cat or dog pic", type=["png", "jpg", "jpeg"])
if file is not None:
    st.markdown("# Your uploaded pic")
    st.image(file)
    test_image = load_img(file, target_size=(64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    
    prediction = "dog" if result[0][0] >= 0.5 else "cat"
    st.markdown("## Prediction")
    st.markdown(f"### {prediction}")

# Calculate accuracy (assuming you have a dataset ready)
try:
    test_set = ImageDataGenerator(rescale=1/255).flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary"
    )
    y_pred = cnn.predict(test_set)
    accuracy = accuracy_score(test_set.classes, np.round(y_pred)) * 100
    st.sidebar.markdown(f"# Model Accuracy")
    st.sidebar.markdown(f"## &emsp;_{accuracy:.2f}%_")
except Exception as e:
    st.error(f"Error calculating accuracy: {str(e)}")

st.markdown("---")
st.markdown("##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Created by [Gandharv Kulkarni](https://share.streamlit.io/user/gandharvk422)")

st.markdown("&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[![GitHub](https://img.shields.io/badge/GitHub-100000?style=the-badge&logo=github&logoColor=white&logoBackground=white)](https://github.com/gandharvk422) &emsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gandharvk422) &emsp; [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/gandharvk422)")
