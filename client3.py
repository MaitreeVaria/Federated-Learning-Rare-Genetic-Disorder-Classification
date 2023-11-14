
import warnings
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os
import flwr as fl
import tensorflow as tf
import keras


import utils

import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    df_train=pd.read_csv("train_split3.csv",encoding='utf-8')
    # X, y = utils.preprocess(df_train)
    x_subclass, y_subclass = utils.preprocess(df_train)
    X_train, X_test, y_train, y_test = train_test_split(x_subclass, y_subclass)
# Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load model and data (MobileNetV2, CIFAR-10)
    model3 = keras.Sequential()
    model3.add(keras.layers.Dense(100, input_dim = 26, activation = 'relu'))
    model3.add(keras.layers.Dense(48, activation = 'relu'))
    model3.add(keras.layers.Dense(16, activation = 'relu'))
    model3.add(keras.layers.Dense(8, activation = 'relu'))
    model3.add(keras.layers.Dense(9, activation = "softmax"))

    model3.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'Precision', 'Recall'])

    epochs = 75
    batch_size = 25


    # Define Flower client
    class HospitalClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model3.get_weights()

        def fit(self, parameters, config):
            model3.set_weights(parameters)
            model3.fit(X_train, y_train,epochs=epochs, batch_size=batch_size, shuffle=True)
            return model3.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            model3.set_weights(parameters)
            loss, accuracy, _, _ = model3.evaluate(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:3000", client=HospitalClient())