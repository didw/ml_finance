import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm


#========================================================
def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))
#========================================================


def tick_bars(df, price_column, m):
    '''
    compute tick bars

    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # returns
        idx: list of indices
    '''
    t = df[price_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def tick_bar_df(df, price_column, m):
    idx = tick_bars(df, price_column, m)
    return df.iloc[idx]


#========================================================
def volume_bars(df, volume_column, m):
    '''
    compute volume bars

    # args
        df: pd.DataFrame()
        column: name for volume data
        m: int(), threshold value for volume
    # returns
        idx: list of indices
    '''
    t = df[volume_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx]


#========================================================
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    return df.iloc[idx]
#========================================================


#@jit(nopython=True)
def numba_isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a), np.fabs(b)), abs_tol)


#@jit(nopython=True)
def bt(p0, p1, bp):
    #if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
    if p0==p1:
        b = bp
        return b
    else:
        b = np.abs(p1-p0)/(p1-p0)
        return b

    
#@jit(nopython=True)
def get_imbalance(t):
    bs = np.ones_like(t)
    for i in tqdm(np.arange(1, bs.shape[0])):
        t_bt = bt(t[i-1], t[i], bs[i-1])
        bs[i] = t_bt
    return bs # remove last value


def numpy_ewma_vectorized(data, window):
    if len(data) == 0:
        return data[0]
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def tib(df, column, exp_num_ticks_init=100, num_prev_bars=1000):
    '''
    compute tick imbalance bars

    # args
        df: pd.DataFrame()
        column: name for price data
    # returns
        idx: list of indices
    '''
    num_ticks_bar = []  # initial value, remove when return
    t = np.array(df[column])
    imbalance_array = get_imbalance(t)
    
    idx = [0]
    cum_theta = 0
    
    exp_num_ticks = exp_num_ticks_init
    expected_imbalance = np.nan

    for i, b in enumerate(tqdm(imbalance_array)):
        cum_theta += b
        
        if np.isnan(expected_imbalance) and i > exp_num_ticks_init:
            expected_imbalance = numpy_ewma_vectorized(np.array(
                imbalance_array[:i]), 
                min(i, exp_num_ticks))[-1]

        if np.abs(cum_theta) >= exp_num_ticks * np.abs(expected_imbalance):
            num_ticks_bar.append(i-idx[-1]+1)
            exp_num_ticks = numpy_ewma_vectorized(np.array(
                num_ticks_bar[-num_prev_bars:]), num_prev_bars)[-1]
            expected_imbalance = numpy_ewma_vectorized(np.array(
                imbalance_array[:i]), 
                min(i, exp_num_ticks*num_prev_bars))[-1]
            idx.append(i)
            cum_theta = 0
            continue
    return idx[1:]


def tib_df(df, column, exp_num_ticks_init=100, num_prev_bars=1000):
    idx = tib(df, column, exp_num_ticks_init, num_prev_bars)
    return df.iloc[idx]


def vib(df, price_column, volume_column, exp_num_ticks_init=100, num_prev_bars=1000):
    '''
    compute tick imbalance bars

    # args
        df: pd.DataFrame()
        price_column: name for price data
        volume_column: name for volume data
    # returns
        idx: list of indices
    '''
    num_ticks_bar = []  # initial value, remove when return
    t = np.array(df[price_column])
    volume = np.array(df[volume_column])
    imbalance_array = get_imbalance(t)
    
    idx = [0]
    cum_theta = 0
    
    exp_num_ticks = exp_num_ticks_init
    expected_imbalance = np.nan

    for i, b in enumerate(tqdm(imbalance_array)):
        if i == 0:
            continue
        v = volume[i]
        cum_theta += b*v
        
        if np.isnan(expected_imbalance) and i > exp_num_ticks_init:
            expected_imbalance = numpy_ewma_vectorized(np.array(
                imbalance_array[1:i]*volume[1:i]), 
                min(i, exp_num_ticks))[-1]

        if np.abs(cum_theta) >= exp_num_ticks * np.abs(expected_imbalance):
            num_ticks_bar.append(i-idx[-1]+1)
            exp_num_ticks = numpy_ewma_vectorized(np.array(
                num_ticks_bar[-num_prev_bars:]), num_prev_bars)[-1]
            expected_imbalance = numpy_ewma_vectorized(np.array(
                imbalance_array[1:i]*volume[1:i]), 
                min(i, exp_num_ticks*num_prev_bars))[-1]
            idx.append(i)
            cum_theta = 0
            continue
    return idx[1:]


def vib_df(df, price_column, volume_column, exp_num_ticks_init=100, num_prev_bars=1000):
    idx = vib(df, price_column, volume_column, exp_num_ticks_init, num_prev_bars)
    return df.iloc[idx]
