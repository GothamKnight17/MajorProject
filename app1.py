import tensorflow as tf
from tensorflow import keras
from keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Image Classification Model')
model = load_model('C:\\Users\\ajit7\\OneDrive\\Documents\\Major Project\\Image_classify.keras')
data_cat = ['WithMask', 'WithoutMask']
img_height = 224
img_width = 224
image =st.text_input('Enter Image name','C:\\Users\\ajit7\\OneDrive\\Documents\\Major Project\\gayatri-malhotra-26SGAduvONc-unsplash.png')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.preprocessing.image.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, 0) / 255.0

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write(  data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))