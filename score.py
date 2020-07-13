import json
import numpy as np
import os
from tensorflow import keras
from azureml.core.model import Model

def init():
    global model

    model_dir = Model.get_model_path("anomaly_detect_lstm_ae")
    with open(os.path.join(model_dir, 'lstm_model.json'), 'r') as f:
        model_json = f.read()
    model = keras.models.model_from_json(model_json)
    model.load_weights(os.path.join(model_dir, 'lstm_model.h5'))
    model.compile(optimizer='adam', loss='mae')

def run(input_data):
    threshold = 0.03
    x = np.array(json.loads(input_data)['data'])
    x_hat = model.predict(x, batch_size=1)
    loss = np.mean(np.abs(x - x_hat), axis=1)
    anomaly = loss > threshold
    return anomaly
