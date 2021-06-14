from tensorflow.keras import callbacks
from src import load_train, get_train,get_shop_max_item
from src.generator import generator
from src.model import getModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression
from tensorflow.keras.callbacks import ModelCheckpoint
import math

import numpy as np
import pandas as pd

def get_shop_max_item(data):
    shop_info=pd.read_csv('data/shops.csv')
    items_id=pd.read_csv('data/items.csv')
    history_shop_value=np.full((len(shop_info['shop_id']),len(items_id['item_id'])),10)

    for iter,row in data.iterrows():
        item_id=int(row['item_id'])
        shop_id=int(row['shop_id'])
        max_count=np.max(np.array(row[3:]))
        if max_count>history_shop_value[shop_id][item_id]:
            history_shop_value[shop_id][item_id]=max_count
    return history_shop_value

if __name__ == "__main__":
    data = load_train()
    df, df_pred = get_train(data)
    history_shop_item=get_shop_max_item(df)
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
        epochs=30, 
        steps_per_epoch=2000, 
        validation_steps=200,
        callbacks=[model_checkpoint]
    )
    # get predict data
    ranges.extend(range(35-hist, 35))
    x_pred = df_pred.iloc[:, ranges]
    result=[]
    for iter,row in x_pred.iterrows():
        shop_id=int(row['shop_id'])
        item_id=int(row['item_id'])
        prediction=model.predict(row)
        prediction=round(prediction*history_shop_item[shop_id][item_id])
        result.append(prediction)
    df_submit = pd.read_csv('data/sample_submission.csv')
    df_submit['item_cnt_month'] = result
    df_submit[df_submit < 0] = 0
    df_submit.to_csv('prediction.csv', index=False)