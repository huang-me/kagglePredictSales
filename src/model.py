from tensorflow.keras.layers import Dense
from tensorflow import keras

def getModel(shape):
    inputs = keras.Input(shape=shape)
    dense1 = Dense(128)(inputs)
    dense2 = Dense(256)(dense1)
    outputs = Dense(1)(dense2)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics="acc"
    )
    return model