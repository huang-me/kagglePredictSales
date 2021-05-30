import pandas as pd
import numpy as np
import random

def get_train(data):
    test = pd.read_csv('data/test.csv')
    items = pd.read_csv('data/items.csv')
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

    # get item category id
    items = items.drop(['item_name'], axis=1)
    train_data = pd.merge(items, train_data, on=['item_id'])
    train_data = train_data.fillna(-1)

    print('***\ntrain_data\n', train_data.head())

    return train_data

def genX(df, ranges, batch_size=16, hist=10):
    indices = np.arange(df.shape[0])
    out_x, out_y = [], []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            r = ranges.copy()
            start = random.randint(2, 35-hist)
            r.extend(range(start, start+hist))
            out_x.append( np.array(df.iloc[i, r]).astype('int') )
            out_y.append( df.iloc[i, start+hist] )
            if len(out_y) == batch_size:
                return (np.array(out_x), np.array(out_y))