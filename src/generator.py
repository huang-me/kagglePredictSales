import numpy as np
import random

def generator(df, ranges, batch_size=16, hist=10):
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
                yield (np.array(out_x), np.array(out_y))
                out_x, out_y = [], []