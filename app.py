import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # standard size of image
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224,224)) # loading image from image path
    img_array = image.img_to_array(img) # converting image to array
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)
    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))# through this we get the complete path of the file
# and now for every file we have to call the extract_features function

features_list = [] # this 2d array will contain the all path of 44k images and inside this array it will contain the features that is 2048

for file in tqdm(filenames):
    features_list.append(extract_features(file,model))

pickle.dump(features_list,open('embeddings.pkl','wb')) # in this the features_list will fill first then it will fill to these files
pickle.dump(filenames,open('filenames.pkl','wb')) # then we can use the file anywhere

# print(np.array(features_list).shape)
# print(len(filenames))
# print(filenames[:5])
# print(os.listdir("images"))