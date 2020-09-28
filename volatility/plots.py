import pandas as pd
import matplotlib.pyplot as plt
from volatility.models_IV import get_IV_predict, get_percent_chg
from pathlib import Path
import os


ROOT_DIR = Path(__file__).parent.parent
cwd = os.getcwd()
df = pd.read_csv(str(ROOT_DIR)+'/data/SPY.zip')
df_option = pd.read_csv(str(ROOT_DIR)+'/data/SPY_option.zip')
df['Date'] = pd.to_datetime(df['Date'])
df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
df_option['Date'] = pd.to_datetime(df_option['Date'])
df_option['Date_str'] = df_option['Date'].dt.strftime('%Y%m%d')
df_option['Date'] = pd.to_datetime(df_option['Date'])
df_option['Expiration'] = pd.to_datetime(df_option['Expiration']).dt.strftime('%Y%m%d')
get_percent_chg(df, 1)
get_percent_chg(df, 5)
get_percent_chg(df, 10)
get_percent_chg(df, 15)
get_percent_chg(df, 21)

k = 0
test_size = 365
ir_free = 0.01
keyList = ['ret_1', 'ret_5', 'ret_10', 'ret_15', 'ret_21']
df_ret = get_IV_predict(df, df_option, test_size, keyList, ir_free)
fig, ax = plt.subplots(figsize=(10, 5), nrows=5, ncols=1)
for key in keyList:
    returns = 100 * df_ret[key].dropna()
    predictions = df_ret['predict_'+key]
    predictions_c_IV = df_ret['IV_predict_c_'+key]
    predictions_p_IV = df_ret['IV_predict_p_'+key]
    ax[k].plot(returns[-test_size:], label=key, color='purple')
    ax[k].plot(predictions, label=key + ' volpred', color='b')
    ax[k].plot(predictions_c_IV, label=key + ' call IV at money', color='g')
    ax[k].plot(predictions_p_IV, label=key + ' put IV at money', color='r')
    ax[k].set_ylabel(key)
    k += 1
    ax[k-1].set_xlabel('Date')

plt.legend(['True Returns', 'Predicted Vol', 'c_IV', 'p_IV'], loc=2, fontsize=6)
plt.show()
