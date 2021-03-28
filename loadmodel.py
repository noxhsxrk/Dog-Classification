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


def loadmd():
    #เปลี่ยน path
    with open('C:/Users/Noahs/Desktop/Dog Classify/history_model', 'rb') as file:
        his = p.load(file)
    #เปลี่ยน path
    filepath = 'C:/Users/Noahs/Desktop/Dog Classify/model1.h5'
    #เปลี่ยน path
    filepath_model = 'C:/Users/Noahs/Desktop/Dog Classify/model1.json'
    #เปลี่ยน path
    filepath_weights = 'C:/Users/Noahs/Desktop/Dog Classify/weights_model.h5'
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
    #plotly.offline.iplot(fig1, filename="testMNIST")
    predict_model = load_model(filepath)
    #predict_model.summary()
    with open(filepath_model, 'r') as f:
        loaded_model_json = f.read()
        predict_model = model_from_json(loaded_model_json)
        predict_model.load_weights(filepath_weights)
        print("Loaded model from disk")
    return predict_model
