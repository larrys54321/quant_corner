import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from volatility.utils import get_percent_chg, Option, set_plot, get_ATR


start = datetime(2000, 1, 1)
end = datetime(2021, 3, 17)
symbol = 'QQQ'
tickerData = yf.Ticker(symbol)
df = tickerData.history(period='1d', start=start, end=end)
df['Date'] = df.index
df['vol_5'] = 50 * np.log(df['Close'] / df['Close'].shift(1)).rolling(window=5).std() * np.sqrt(5)
df['vol_15'] = 50 * np.log(df['Close'] / df['Close'].shift(1)).rolling(window=15).std() * np.sqrt(21)
df['vol_5'] = df['vol_5'].fillna(0)
df['vol_15'] = df['vol_15'].fillna(0)
get_ATR(df, 5, f=50)
get_ATR(df, 15, f=50)
get_percent_chg(df, 5)
get_percent_chg(df, 15)
closes = df.Close
returns = df.Close.pct_change().fillna(0)
df['ret_1a'] = returns
test_size = 365*5
test_size = 300

keyList, keyList_vol, keyList_ATR = ['ret_5', 'ret_15'], ['vol_5', 'vol_15'], ['ATR_5', 'ATR_15']
fig, ax = plt.subplots(figsize=(10, 5), nrows=3, ncols=1)
k = 0
for k in range(len(keyList)):
    key, key_vol, key_ATR = keyList[k], keyList_vol[k], keyList_ATR[k]
    returns = 100 * df[key].dropna()
    predictions = []
    print('key', key, 'key_vol', key_vol)
    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=2, q=2)
        model_fit = model.fit(disp='off')
        pred_val = model_fit.forecast(horizon=1)
        predictions.append(np.sqrt(pred_val.variance.values[-1,:][0]))
    predictions = pd.Series(predictions, index=returns.index[-test_size:])
    ax[k].plot(df['Date'][-test_size:], df[key_ATR][-test_size:], linewidth=0.5, color='g')
    ax[k].plot(df['Date'][-test_size:], df['vol_5'][-test_size:], linewidth=0.5, color='b')
    ax[k].plot(df['Date'][-test_size:], predictions, linewidth=0.5, color='r')
    ax[k].xaxis.set_ticklabels([])
    set_plot(ax[k])
    ax[k].legend([key_ATR, 'vol_5', 'Garch Vol '+key], loc=2, fontsize=8)
    k += 1
ax[k].set_xlabel('Date')
ax[k].plot(df['Date'][-test_size:], np.array(closes[len(closes)-test_size:])/5-50, label='Close', color='b')
ax[k].plot(df['Date'][-test_size:], 100 * df['ret_5'][-test_size:], label='ret_5', linewidth=0.5, color='r')
ax[k].plot(df['Date'][-test_size:], 100 * df['ret_15'][-test_size:], label='ret_15', linewidth=0.5, color='g')
set_plot(ax[k])
ax[k].legend(['Close', 'ret_5', 'ret_15'], loc=2, fontsize=8)
plt.show()