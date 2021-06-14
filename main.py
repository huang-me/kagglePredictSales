# from tensorflow.keras import callbacks
# from tensorflow.keras.callbacks import ModelCheckpoint
from src import load_train, get_train
# from src.generator import generator
# from src.model import getModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    # disbale gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # data
    data = load_train()
    df, df_pred = get_train(data)
    # train test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    x_train = np.array(df_train.iloc[:, -32:-1])
    y_train = np.array(df_train.iloc[:, -1:])
    x_test =  np.array(df_test.iloc[:, -32:-1])
    y_test =  np.array(df_test.iloc[:, -1:])

    model = DecisionTreeRegressor()
    # model = RandomForestRegressor()
    model.fit(x_train, y_train)

    print('train acc:', mean_squared_error(y_train, model.predict(x_train), squared=False))
    print('test acc:', mean_squared_error(y_test, model.predict(x_test), squared=False))

    x_pred = df_pred.iloc[:, -31:]
    result = model.predict(x_pred)
    
    df_submit = pd.read_csv('data/sample_submission.csv')
    df_submit['item_cnt_month'] = result
    df_submit[df_submit < 0] = 0
    df_submit.to_csv('prediction.csv', index=False)