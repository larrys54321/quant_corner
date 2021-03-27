import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from volatility.utils import get_percent_chg, Option, set_plot, get_ATR
from volatility.models_IV import get_IV
from pathlib import Path
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
ROOT_DIR = Path(__file__).parent.parent
cwd = os.getcwd()
df = pd.read_csv(str(ROOT_DIR)+'/data/SPY.zip')
df_option = pd.read_csv(str(ROOT_DIR)+'/data/SPY_option.zip')
df['Date'] = pd.to_datetime(df['Date'])
df['Date_str'] = df['Date'].dt.strftime('%Y%m%d')
df_option['Date'] = pd.to_datetime(df_option['Date'])
df_option['Date_str'] = df_option['Date'].dt.strftime('%Y%m%d')
df_option['Date'] = pd.to_datetime(df_option['Date'])
df_option['Expiration'] = pd.to_datetime(df_option['Expiration']).dt.strftime('%Y%m%d')

df['vol_5'] = 50 * np.log(df['Close'] / df['Close'].shift(1)).rolling(window=5).std() * np.sqrt(5)
df['vol_15'] = 50 * np.log(df['Close'] / df['Close'].shift(1)).rolling(window=15).std() * np.sqrt(21)
df['vol_5'] = df['vol_5'].fillna(0)
df['vol_15'] = df['vol_15'].fillna(0)
get_ATR(df, 5, f=50)
get_ATR(df, 15, f=50)
get_percent_chg(df, 5)
get_percent_chg(df, 15)

k, test_size, ir_free = 0, 250, 0.01
keyList, keyList_vol, keyList_ATR = ['ret_5', 'ret_15'], ['vol_5', 'vol_15'], ['ATR_5', 'ATR_15']
keyList, keyList_vol, keyList_ATR = ['ret_5'], ['vol_5'], ['ATR_5']
df_ret = get_IV(df, df_option, test_size, ir_free, keyList=keyList, keyList_vol=keyList_vol, keyList_ATR=keyList_ATR)
fig, ax = plt.subplots(figsize=(10, 5), nrows=2*len(keyList)+2, ncols=1)
# ret_5 vol_5  IV_c_0_ret_5  IV_p_0_ret_5  IV_c_u_ret_5  IV_p_u_ret_5  IV_c_d_ret_5  IV_p_d_ret_5
for k in range(0, len(keyList), 2):
    key, key_vol, key_ATR = keyList[k], keyList_vol[k], keyList_ATR[k]
    returns = 100 * df_ret[key].dropna()
    vols = df_ret[key_vol]*3
    ax[2*len(keyList)].plot(returns[-test_size:], label=key, color='purple')
    ax[2*len(keyList)].plot(vols, label=key + ' vols', color='b')
    ax[k].plot(df_ret['IV_c_0_'+key].rolling(window=5).mean(), label=key + ' call IV at money', color='black')
    ax[k].plot(df_ret['IV_c_u_'+key].rolling(window=5).mean(), label=key + ' call IV 10% up', color='g')
    ax[k].plot(df_ret['IV_c_d_'+key].rolling(window=5).mean(), label=key + ' call IV 10% down', color='b')
    ax[k+1].plot(df_ret['IV_p_0_'+key].rolling(window=5).mean(), label=key + ' put IV at money', color='black')
    ax[k + 1].plot(df_ret['IV_p_u_' + key].rolling(window=5).mean(), label=key + ' put IV 10% up', color='g')
    ax[k + 1].plot(df_ret['IV_p_d_' + key].rolling(window=5).mean(), label=key + ' put IV 10% down', color='b')
    ax[k].xaxis.set_ticklabels([])
    ax[k+1].xaxis.set_ticklabels([])
    set_plot(ax[k])
    set_plot(ax[k+1])
    ax[k].legend(['Call IV at money', 'Call IV 10% up', 'Call IV 10% down'], loc=2, fontsize=8)
    ax[k+1].legend(['Put IV at money', 'Put IV 10% up', 'Put IV 10% down'], loc=2, fontsize=8)

ax[2*len(keyList)].legend(keyList+keyList_vol, loc=2, fontsize=6)
ax[2*len(keyList)].xaxis.set_ticklabels([])
set_plot(ax[k+2])
ax[k+3].plot(df['Date'][-test_size:], df['Close'][-test_size:], label=key, color='black')
ax[k+3].legend(['Close'], loc=2, fontsize=6)
set_plot(ax[k+3])
plt.show()
