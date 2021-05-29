from src import load_train, get_train
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression

import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = load_train()
    x_train, y_train, x_test, y_test, x_pred = get_train(data)
    # initialize model
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    print('Train set mse:', mean_squared_error(y_train, LR.predict(x_train)))
    print('Test set mse:', mean_squared_error(y_test, LR.predict(x_test)))

    result = LR.predict(x_pred)
    df_submit = pd.read_csv('data/sample_submission.csv')
    df_submit['item_cnt_month'] = result
    df_submit.to_csv('prediction.csv', index=False)