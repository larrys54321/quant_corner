import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from volatility.utils import get_percent_chg

start = datetime(2000, 1, 1)
end = datetime(2020, 9, 11)
symbol = 'SPY'
tickerData = yf.Ticker(symbol)
df = tickerData.history(period='1d', start=start, end=end)
get_percent_chg(df, 1)
get_percent_chg(df, 5)
get_percent_chg(df, 10)
get_percent_chg(df, 15)
get_percent_chg(df, 21)
returns = df.Close.pct_change().dropna()
df['ret_1a'] = returns
test_size = 365*5
test_size = 365

keyList = ['ret_1', 'ret_5', 'ret_10', 'ret_15', 'ret_21']
fig, ax = plt.subplots(figsize=(10, 5), nrows=5, ncols=1)
k = 0
for key in keyList:
    returns = 100 * df[key].dropna()
    predictions = []
    print('key', key)
    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=2, q=2)
        model_fit = model.fit(disp='off')
        pred_val = model_fit.forecast(horizon=1)
        predictions.append(np.sqrt(pred_val.variance.values[-1,:][0]))
    predictions = pd.Series(predictions, index=returns.index[-test_size:])
    ax[k].plot(returns[-test_size:], label=key, color='r')
    ax[k].plot(predictions, label=key+' volpred', color='b')
    ax[k].set_ylabel(key)
    k += 1
ax[k-1].set_xlabel('Date')
plt.legend(['True Returns', 'Predicted Volatility'], loc=2, fontsize=8)
plt.show()