import pandas as pd
import os

from pandas.core.frame import DataFrame

def load_train(path='data/sales_train.csv'):
    '''
    load data from file
    count sold count and mean price of each item in each month
    '''
    train = pd.read_csv(path)
    tmp = train.groupby(['date_block_num', 'shop_id', 'item_id'])
    out = DataFrame({
        'cnt/m': tmp['item_cnt_day'].sum()
    }).reset_index()
    return out

