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
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator

predict_model=lm.loadmd()
dataset = "C:/Users/Noahs/Desktop/train"
data_dir = pathlib.Path(dataset)

def define_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    return model

batch_size = 32
img_height = 180
img_width = 180
num_classes = 10;

train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # sแบ่งข้อมูล เพื่อ training 80% และ validate 20%
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

predict_model = define_model()

predict_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

his = predict_model.fit(
    train,
    validation_data=val,
    epochs=1000
)

with open('history_model', 'wb') as file:
    p.dump(his.history, file)

filepath = 'model2.h5'
model.save(filepath)
filepath_model = 'model2.json'
filepath_weights = 'weights_model2.h5'
model_json = model.to_json()
with open(filepath_model, "w") as json_file:
    json_file.write(model_json)

    model.save_weights('weights_model2.h5')
    print("Saved model to disk")
