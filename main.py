import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

features_list = np.array(pickle.load(open('embeddings.pkl','rb'))) # this is for 2 array
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # standard size of image
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title("Recommendation System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def features_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # loading image from image path
    img_array = image.img_to_array(img)  # converting image to array
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = features_extraction(os.path.join("uploads",uploaded_file.name),model)
        st.text(features)
        indices = recommend(features,features_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("Some error occured in file upload")