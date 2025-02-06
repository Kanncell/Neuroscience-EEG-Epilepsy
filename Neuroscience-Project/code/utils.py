import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eeg_from_parquet(parquet_path, FEATS, display = False):
    eeg = pd.read_parquet(parquet_path,columns=FEATS)
    eeg_shape = len(eeg)
    print(eeg_shape)
    offset = (eeg_shape-10_000)//2
    print(offset)
    eeg = eeg.iloc[offset:offset+10_000]
    print(eeg)
    data = np.zeros((10_000,len(FEATS)))
    for j,col in enumerate(FEATS):
        
        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
        else: x[:] = 0
            
        data[:,j] = x
        if display: 
            if j!=0: offset += x.max()
            plt.plot(range(10_000),x-offset,label=col)
            offset -= x.min()
    if display:
        plt.legend()
        name = parquet_path.split('/')[-1]
        name = name.split('.')[0]
        plt.title(f'EEG {name}',size=16)
        plt.show()
        
    return data

