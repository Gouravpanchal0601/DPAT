import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

# features_list = pickle.load(open('embeddings.pkl','rb'))
features_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

#here we are calling the resnet50 module from cnn libray
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # standard size of image
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img("sample/red.jpg", target_size=(224, 224))  # loading image from image path
img_array = image.img_to_array(img)  # converting image to array
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(features_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)