from src import get_train, load_train

import pandas as pd

if __name__ == "__main__":
    # load data from files
    data = load_train()
    # preprocess datas
    df = get_train(data)
    # output to file
    df.to_csv('processed.csv', index=False)