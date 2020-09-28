import pandas as pd
import numpy as np
from datetime import datetime
from arch import arch_model
from volatility.utils import get_percent_chg, Option
import statsmodels.api as sm
from sklearn import linear_model

def get_IV_predict(df, df_option, test_size, keyList, ir_free):
    df_ret = pd.DataFrame()
    df_ret['Date'] = df['Date'][len(df)-test_size:]
    df_ret['Date_str'] = df['Date_str'][len(df)-test_size:]
    lm = linear_model.LinearRegression()
    dates = list(df_ret['Date_str'])
    for key in keyList:
        df_ret[key] = df[key]
        returns = 100 * df[key].dropna()
        predictions = []
        predictions_c_iv = []
        predictions_p_iv = []
        print('key', key)
        for i in range(test_size):
            date_str = dates[i].replace('-', '')
            df_option_ = df_option[df_option['Date_str']==date_str]
            train = returns[:-(test_size-i)]
            model = arch_model(train, p=2, q=2)
            model_fit = model.fit(disp='off')
            pred_val = model_fit.forecast(horizon=1)
            p_val = np.sqrt(pred_val.variance.values[-1,:][0])

            rows_iv = []
            s = 0
            for ii, row in df_option_.iterrows():
                k = row['Strike']
                exp_date = row['Expiration']
                price = row['Close']
                type_ = row['Type'][0]
                s = row['RootClose']
                d1 = datetime.strptime(date_str, "%Y%m%d")
                d2 = datetime.strptime(exp_date, "%Y%m%d")
                days_exp = (d2 - d1).days
                if days_exp > 50 or days_exp < 10: continue
                if float(abs(s-k))/k > 0.2: continue
                opt = Option(s=s, k=k, eval_date=date_str, exp_date=exp_date, price=price, rf=ir_free, vol=0.01*p_val, right=type_)
                iv = opt.get_implied_vol()*100
                rows_iv.append({'Strike':k, 'Days_exp':days_exp, 'Type':type_, 'IV':iv})
            df_iv = pd.DataFrame(rows_iv)
            df_iv_c, df_iv_p = df_iv[df_iv['Type'] == 'C'], df_iv[df_iv['Type'] == 'P']
            X = np.array(df_iv_c[['Strike', 'Days_exp']])
            y = np.array(df_iv_c['IV'])
            model = lm.fit(X, y)
            x_ = np.array(pd.DataFrame([{'Strike':s, 'Days_exp':30}]))
            iv_am_c = model.predict(x_)[0]
            X = np.array(df_iv_p[['Strike', 'Days_exp']])
            y = np.array(df_iv_p['IV'])
            model = lm.fit(X, y)
            x_ = np.array(pd.DataFrame([{'Strike':s, 'Days_exp':30}]))
            iv_am_p = model.predict(x_)[0]
            predictions.append(p_val)
            predictions_c_iv.append(iv_am_c)
            predictions_p_iv.append(iv_am_p)
        df_ret['predict_'+key] = predictions
        df_ret['IV_predict_c_'+key] = predictions_c_iv
        df_ret['IV_predict_p_'+key] = predictions_p_iv
    df_ret.set_index('Date', inplace=True)
    return df_ret