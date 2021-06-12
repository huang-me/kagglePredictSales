from src import get_train, load_train

import pandas as pd

def fullPreprocess():
    # load data from files
    data = load_train()
    # preprocess datas
    df = get_train(data)
    # output to file
    df.to_csv('processed.csv', index=False)

if __name__ == "__main__":
    fullPreprocess()