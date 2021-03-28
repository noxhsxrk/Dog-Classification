import loadmodel as lm
import requests
from IPython.display import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
from tensorflow import keras
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


predict_model=lm.loadmd()
#เปลี่ยน path
test_path = ('C:/Users/Noahs/Desktop/12.jpg')
img = keras.preprocessing.image.load_img(
    test_path, target_size=(180, 180)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
y_predictions = predict_model.predict(img_array)
score = tf.nn.softmax(y_predictions[0])
breed_list = ['Chihuahua', 'German_shepherd', 'Great_Dane', 'Labrador_retriever', 'Pug', 'Rottweiler', 'Shiba', 'Shih-Tzu', 'golden_retriever']
y_classes = y_predictions.argmax(axis=-1)

for i in range(10):
    print(score[i])

for i in range(10):
    if score[i]==np.max(score):  
        image = mpimg.imread(test_path)
        plt.imshow(image)
        ax = plt.subplot()
        ax.set_title("Prediction : {} with {:.2f}%.".format(breed_list[i], 100 * np.max(score)))
        plt.show()
        break