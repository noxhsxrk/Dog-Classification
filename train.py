import tensorflow as tf
import PIL
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import plotly
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import plotly.graph_objs as go
from tensorflow import keras
from tensorflow.keras.models import Sequential
import pathlib

import requests
from IPython.display import Image
from io import BytesIO

dataset = "C:/Users/Noahs/Desktop/train"
data_dir = pathlib.Path(dataset)

batch_size = 32
img_height = 180
img_width = 180

train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # sแบ่งข้อมูล เพื่อ training 80% และ validate 20%
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print(train.class_names)

val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

num_classes = 10
epochs = 1000

model = Sequential([
    layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train.class_names[labels[i]])
        plt.axis("off")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.summary()

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

start = time.time()

his = model.fit(
    train,
    validation_data=val,
    epochs=1000
)
done = time.time()
print("Time", done - start)


# load = keras.models.load_model('C:/Users/Noahs/Desktop/history_model')
with open('history_model', 'wb') as file:
    p.dump(his.history, file)

filepath = 'model1.h5'
model.save(filepath)
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
model_json = model.to_json()
with open(filepath_model, "w") as json_file:
    json_file.write(model_json)

    model.save_weights('weights_model.h5')
    print("Saved model to disk")

with open('history_model', 'rb') as file:
    his = p.load(file)
filepath = 'model1.h5'
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
h1 = go.Scatter(y=his['loss'],
                mode="lines", line=dict(
    width=2,
    color='blue'),
    name="loss"
)
h2 = go.Scatter(y=his['val_loss'],
                mode='lines', line=dict(
    width=2,
    color='red'),
    name="val_loss"
)

data = [h1, h2]
layout1 = go.Layout(title='Loss',
                    xaxis=dict(title='epochs'),
                    yaxis=dict(title=''))
fig1 = go.Figure(data, layout=layout1)

predict_model = load_model(filepath)
predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    print("Loaded model from disk")


# test_path = ('C:/Users/Noahs/Desktop/Degaen.jpg')
# img = keras.preprocessing.image.load_img(
#     test_path, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# y_predictions = predict_model.predict(img_array)
# score = tf.nn.softmax(y_predictions[0])
# y_classes = y_predictions.argmax(axis=-1)
# print(y_classes)
# for i in range(9):
#   print(score[i])
# print(np.max(score))
# #display(Image(filename=test_path,width=180, height=180))
# if score[0]==np.max(score) :
#     breed = "Chi"
# elif score[1]==np.max(score) :
#     breed = "German"
# elif score[2]==np.max(score) :
#     breed = "Greatdane"
# elif score[3]==np.max(score) :
#     breed = "Labrador"
# elif score[4]==np.max(score) :
#     breed = "Pug"
# elif score[5]==np.max(score) :
#     breed = "Rott"
# elif score[6]==np.max(score) :
#     breed = "Shiba"
# elif score[7]==np.max(score) :
#     breed = "Shihtzu"
# elif score[8]==np.max(score) :
#     breed = "Golden"

# print("ทำนายว่าเป็น {} ด้วยความมั่นใจ {:.2f}%.".format(breed, 100 * np.max(score)))
