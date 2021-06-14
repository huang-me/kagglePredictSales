import pandas as pd
import numpy as np

def load_train(path='../data/sales_train.csv'):
    '''
    load data from file
    count sold count and mean price of each item in each month
    '''
    train = pd.read_csv(path)
    tmp = train.groupby(['date_block_num', 'shop_id', 'item_id'])
    out = pd.DataFrame({
        'cnt/m': tmp['item_cnt_day'].sum()
    }).reset_index()
    return out

def get_train(data):
    #test = pd.read_csv('../data/test.csv')
    #items = pd.read_csv('../data/items.csv')
    test = pd.read_csv('data/test.csv')
    items = pd.read_csv('data/items.csv')
    data = data.pivot_table(
        index=['shop_id', 'item_id'], 
        columns='date_block_num',
        values='cnt/m',
        fill_value=0)
    data.reset_index(inplace=True)

    # merge data and test
    test_data = pd.merge(test, data, on=['shop_id', 'item_id'], how='left',sort=False)
    test_data = test_data.drop(['ID'], axis=1)
    test_data = test_data.fillna(0)

    # get item category id
    items = items.drop(['item_name'], axis=1)
    train_data = pd.merge(items, data, on=['item_id'])
    train_data = train_data.fillna(-1)

    test_data = test_data.merge(items, on=['item_id'], how='left', sort=False)
    test_data = test_data.fillna(-1)
    cols = list(test_data.columns.values)
    cols = cols[-1:] + cols[:-1]
    test_data = test_data[cols]
    print(test_data.head())

    #print(f'***\ntrain_data {train_data.shape}\n', train_data.head())
    #print(f'***\ntest_data {test_data.shape}\n', test_data.head())

    return train_data, test_data

