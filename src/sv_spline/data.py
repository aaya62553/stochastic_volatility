import pandas as pd
import numpy as np

def load_data(path,scale=1.0):
    df = pd.read_csv(path, index_col=False)
    df = df.rename(columns={'snapped_at': 'Date', 'price': 'Close'}, errors='ignore')
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = pd.to_datetime(df['Date'])
    df['Returns'] = df['Close'].pct_change() * scale
    df['Log Returns'] = np.log(df['Close']).diff()
    df = df[df["Date"]>="2014-01-01"]
    df = df.dropna().reset_index(drop=True)
    return df

def split_data(df, train_size=0.8, split_date=None):
    if split_date is not None:
        split_index = df[df['Date'] >= split_date].index[0]
    else:
        split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def get_y(df, y_col='Log Returns'):
    return df[y_col].values