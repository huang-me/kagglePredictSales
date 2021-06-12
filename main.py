from src import load_train, get_train, generator, getModel
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # disbale gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # data
    data = load_train()
    df = get_train(data)
    # train test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    # generators
    bacth_size = 16
    hist = 15
    ranges = [0, 1, 2]
    test_gen = generator(df_test, ranges, bacth_size, hist)
    train_gen = generator(df_train, ranges, bacth_size, hist)
    # get model and train
    model = getModel([hist+len(ranges)])
    model.summary()
    model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(
        train_gen, 
        validation_data=test_gen, 
        # epochs=3, 
        epochs=10, 
        steps_per_epoch=2000, 
        validation_steps=200,
        callbacks=[model_checkpoint]
    )
    # get predict data
    ranges.extend(range(35-hist, 35))
    x_pred = df.iloc[:, ranges]
    result = model.predict(x_pred)
    
    df_submit = pd.read_csv('data/sample_submission.csv')
    df_submit['item_cnt_month'] = result
    df_submit[df_submit < 0] = 0
    df_submit.to_csv('prediction.csv', index=False)