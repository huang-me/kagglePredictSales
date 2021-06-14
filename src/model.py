from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow import keras
import tensorflow as tf
import math

def getModel(shape):
    inputs = keras.Input(shape=shape)

    bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    bilstm2 = Bidirectional(LSTM(128))(bilstm1)

    outputs = Dense(64,activation='relu')(bilstm2)
    outputs = Dense(32,activation='relu')(outputs)
    outputs = Dense(8,activation='relu')(outputs)
    outputs = Dense(1,activation='sigmoid')(outputs)
    
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["val_loss"]
        #metrics=tf.keras.metrics.MeanSquaredError
    )
    return model