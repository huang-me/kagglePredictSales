from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow import keras

def getModel(shape):
    inputs = keras.Input(shape=shape)
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    bilstm2 = Bidirectional(LSTM(128))(bilstm1)
    outputs = Dense(1)(bilstm2)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["acc"]
    )
    return model