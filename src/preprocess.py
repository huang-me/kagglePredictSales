from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_train(data):
    test = pd.read_csv('data/test.csv')
    data = data.pivot_table(
        index=['shop_id', 'item_id'], 
        columns='date_block_num',
        # values=['cnt/m', 'avg_price'],
        values='cnt/m',
        fill_value=0)
    data.reset_index(inplace=True)

    # merge data and test
    train_data = pd.merge(test, data, on=['shop_id', 'item_id'], how='left')
    train_data = train_data.drop(['ID'], axis=1)
    train_data = train_data.fillna(0)

    X_train = train_data.drop([33], axis=1)
    Y_train = train_data[33].values
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)

    x_pred = train_data.drop([0], axis=1)
    return x_train, y_train, x_test, y_test, x_pred