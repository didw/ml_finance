import boto3
import pandas as pd
import numpy as np
import io
import os
from tqdm import tqdm


def get_datelist():
    s3 = boto3.resource('s3')
    contents = s3.meta.client.list_objects(Bucket='schperics.stock')['Contents']
    date_list = []
    for c in contents:
        if 'csv_dp_tick' in c['Key'] and 'csv' in c['Key']:
            date = os.path.basename(c['Key']).split('.')[0]
            date_list.append(date)
    date_list = np.unique(date_list)
    return date_list


def load_data(codes, date='2019-05-10'):
    s3 = boto3.resource('s3')
    key_name = 'csv_dp_tick/{}.csv'.format(date)
    obj = s3.meta.client.get_object(Bucket='schperics.stock', Key=key_name)
    try:
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        print(e)
        return pd.DataFrame([], columns=['shcode', 'chetime', 'price', 'sign', 'cvolume'])
    
    df['dtime'] = df['chetime'].apply(lambda x: '{}{:06d}'.format(date, x))
    df['dtime'] = pd.to_datetime(df['dtime'], format='%Y-%m-%d%H%M%S')
    df = df.set_index('dtime')
    
    df = df[(90000<=df['chetime']) & (df['chetime']<=153000)]
    df = df[['shcode', 'chetime', 'price', 'sign', 'cvolume']]
    if type(codes) != type([]):
        return df[df['shcode']==codes]
    res_df = None
    for code in codes:
        if res_df is None:
            res_df = df[df['shcode']==code]
        else:
            res_df = res_df.append(df[df['shcode']==code])
    return res_df


def load_all_data(codes):
    date_list = get_datelist()
    res_df = None
    for cur_date in tqdm(date_list, ncols=80):
        df = load_data(codes, cur_date)
        if res_df is None:
            res_df = df
        else:
            res_df = res_df.append(df)
    return res_df.reset_index(drop=True)

