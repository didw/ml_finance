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


def tib(df, column):
    '''
    compute tick imbalance bars

    # args
        df: pd.DataFrame()
        column: name for price data
    # returns
        idx: list of indices
    '''
    initial_T = 100
    len_bars = [initial_T]  # initial value, remove when return
    t = np.array(df[column])
    bt = get_imbalance(t)
    bt_bars = bt[:initial_T]  # initial value, remove when return
    
    idx = [0]
    theta_t = 0
    recent_bt = []
    
    e_t = np.mean(len_bars)
    e_bt = numpy_ewma_vectorized(bt_bars, len(bt_bars))[-1]
    
    for i, b in enumerate(tqdm(bt[initial_T:])):
        theta_t += b
        recent_bt.append(b)
        if np.sum(theta_t) >= e_t*e_bt:
            if len_bars[0]==initial_T:  # 초기값 제거, 교체
                len_bars = [i-idx[-1]+1]
                bt_bars = recent_bt
            else:
                len_bars.append(i-idx[-1]+1)
                bt_bars.extend(recent_bt)  # 이전바의 모든 bt값을 넣을까, 바별로 하나의 대표값을 넣을까?
            if len(len_bars)==1:
                e_t = np.mean(len_bars)
            else:
                e_t = numpy_ewma_vectorized(np.array(len_bars), len(len_bars))[-1]
            e_bt = numpy_ewma_vectorized(np.array(bt_bars), len(bt_bars))[-1]
            idx.append(i)
            theta_t = 0
            recent_bt = []
            continue
    return idx[1:]


def tib_df(df, column):
    idx = tib(df, column)
    return df.iloc[idx]


def vib(df, price_column, volume_column):
    '''
    compute volume imbalance bars

    # args
        df: pd.DataFrame()
        price_column: name for price data
        volume_column: name for volume data
    # returns
        idx: list of indices
    '''
    initial_T = 100
    len_bars = [initial_T]  # initial value, remove when return
    t = np.array(df[price_column])
    vs = np.array(df[volume_column])
    bt = get_imbalance(t)
    vbt_bars = vs[1:initial_T]*bt[1:initial_T]  # initial value, remove when return, remove first volume
    
    idx = [0]
    theta_t = 0
    recent_vbt = []
    
    e_t = np.mean(len_bars)
    e_vbt = numpy_ewma_vectorized(vbt_bars, len(vbt_bars))[-1]
    
    print(e_t, e_vbt)
    
    for i, b in enumerate(tqdm(bt[initial_T:])):
        v = vs[i]
        theta_t += b*v
        recent_vbt.append(b*v)
        if np.sum(theta_t) >= e_t*e_vbt:
            if len_bars[0]==initial_T:  # 초기값 제거, 교체
                len_bars = [i-idx[-1]+1]
                vbt_bars = recent_vbt
            else:
                len_bars.append(i-idx[-1]+1)
                vbt_bars.extend(recent_vbt)  # 이전바의 모든 bt값을 넣을까, 바별로 하나의 대표값을 넣을까?
            if len(len_bars)==1:
                e_t = np.mean(len_bars)
            else:
                e_t = numpy_ewma_vectorized(np.array(len_bars), len(len_bars))[-1]
            e_vbt = numpy_ewma_vectorized(np.array(vbt_bars), len(vbt_bars))[-1]
            idx.append(i)
            theta_t = 0
            recent_vbt = []
            continue
    return idx[1:]


def vib_df(df, price_column, volume_column):
    idx = vib(df, price_column, volume_column)
    return df.iloc[idx]