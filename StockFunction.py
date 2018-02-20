import numpy as np
import pandas as pd

#======================Time Series Tool 时间序列工具==================================
def ts_mean(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).mean().values[:,0])

def ts_std(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).std().values[:,0])

def ts_max(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).max().values[:,0])

def ts_min(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).min().values[:,0])

def ts_skew(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).skew().values[:,0])


def ts_kurt(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).kurt().values[:,0])

def ts_mid(data,n):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).median().values[:,0])

def ts_quantile(data,n,alpha):
    data=pd.DataFrame(data)
    return np.array(data.rolling(window=n).quantile(alpha).values[:,0])

def ts_retn(data,n):
    data=pd.DataFrame(data)
    return np.array((data/(data.rolling(window=n).sum()-data.rolling(window=n-1).sum())).values[:,0])

def ts_delay(data,n):
    data=pd.DataFrame(data)
    return np.array(data.shift(n).values[:,0])

def ts_corr(data1,data2,n):
    r1=pd.DataFrame(data1).rolling(window=n)
    r2=pd.DataFrame(data2).rolling(window=n)
    return np.array(r1.corr(r2).values[:,0])


#======================Cross Section 横截面工具==================================
def cs_rank(data):
    data = np.array(data)
    data_sort=np.argsort(data, axis=1)
    for i, j in enumerate(data_sort):
        for m, n in enumerate(data_sort[i]):
            data[i][n] = m
    return np.array(data+1)

def cs_percent(data):
    data = pd.DataFrame(data)
    return np.array(data.sub(np.min(data, axis=1), axis=0).div(np.max(data, axis=1) - np.min(data, axis=1), axis=0))


def cs_group(data,n):
    data_sort=cs_rank(data)
    m=len(data_sort[0])/n
    return np.array(np.floor(data_sort/m-0.0001)+1)

def cs_max(data):
    return np.array(list(np.max(data,axis=1))*len(data.columns)).reshape((len(data.columns),len(data.index))).T

def cs_min(data):
    return np.array(list(np.min(data,axis=1))*len(data.columns)).reshape((len(data.columns),len(data.index))).T

def cs_cuttail(data,a,b):
    data=cs_percent(data)
    data[data<=a]=0
    data[data>=b]=0
    return np.array(data)


#======================数据框==================================
def df_copy_as_ones(data):
    # return data.clip(1,1)
    return np.ones((len(np.array(data)),len(np.array(data)[0])))

def df_sign(data):
    data[data>0]=1
    data[data<0]=-1
    return np.array(data)

def df_replace_zero(data):
    data[data==0]=np.nan
    return np.array(data)

def df_replace_nan(data,number):
    data[np.isnan(data)]=number
    return np.array(data)

def df_power(data,n):
    data=np.array(data)
    return data**n

def df_abs(data):
    return np.array(np.abs(data))