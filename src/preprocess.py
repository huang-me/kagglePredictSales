import pandas as pd
import numpy as np

def get_train(data):
    test = pd.read_csv('data/test.csv')
    data = data.pivot_table(
        index=['shop_id', 'item_id'], 
        columns='date_block_num',
        values='cnt/m',
        fill_value=0)
    data.reset_index(inplace=True)

    # merge data and test
    train_data = pd.merge(test, data, on=['shop_id', 'item_id'], how='left')
    train_data = train_data.drop(['ID'], axis=1)
    train_data = train_data.fillna(0)

    print('train_data.shape', train_data.shape)

    return train_data