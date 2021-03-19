import math
from scipy.stats import norm
import pandas as pd, numpy as np
from datetime import date
from matplotlib import ticker
import matplotlib.dates as mdates

def set_plot(ax):
    ax.set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #myFmt = mdates.DateFormatter('%y-%m-%d')
    #myFmt = mdates.DateFormatter('%D')
    #ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(labelrotation=30, labelsize=6)
    M = 30
    xticks = ticker.MaxNLocator(M)
    ax.xaxis.set_major_locator(xticks)

def get_ATR(df, w, f=1):
    out = w * [0]
    highs = list(df.High)
    lows = list(df.Low)
    closes = list(df.Close)
    for i in range(w, len(closes)):
        ATR = f * max(highs[i]-lows[i], abs(highs[i]-closes[i-w]), abs(lows[i]-closes[i-w]))
        out.append(ATR/closes[i-w])
    df['ATR_'+str(w)] = out

def get_percent_chg(df, w):
    out = w * [0]
    closes = list(df.Close)
    for i in range(w, len(df)):
        out.append((closes[i]-closes[i-w])/closes[i-w])
    df['ret_'+str(w)] = out

class Option:
    """
    This class will calculate an black-shcoles opion
    """

    def __init__(self, right, s, k, eval_date, exp_date, price=0, rf=0.01, vol=0.3, div=0):
        self.k = float(k)
        self.s = float(s)
        self.rf = float(rf)
        self.vol = float(vol)
        self.eval_date = eval_date
        self.exp_date = exp_date
        self.t = self.calculate_t()
        if self.t == 0: self.t = 0.000001
        self.price = price
        self.right = right  # 'Call' or 'Put'
        self.div = div

    def calculate_t(self):
        if isinstance(self.eval_date, str):
            if '/' in self.eval_date:
                (day, month, year) = self.eval_date.split('/')
            else:
                (day, month, year) = self.eval_date[6:8], self.eval_date[4:6], self.eval_date[0:4]
            d0 = date(int(year), int(month), int(day))
        #elif type(self.eval_date) == float or type(self.eval_date) == long or type(self.eval_date) == np.float64:
        elif type(self.eval_date) == float or type(self.eval_date) == np.float64:
            (day, month, year) = (str(self.eval_date)[6:8], str(self.eval_date)[4:6], str(self.eval_date)[0:4])
            d0 = date(int(year), int(month), int(day))
        else:
            d0 = self.eval_date

        if isinstance(self.exp_date, str):
            if '/' in self.exp_date:
                (day, month, year) = self.exp_date.split('/')
            else:
                (day, month, year) = self.exp_date[6:8], self.exp_date[4:6], self.exp_date[0:4]
            d1 = date(int(year), int(month), int(day))
        elif type(self.exp_date) == float or type(self.exp_date) == int or type(self.exp_date) == np.float64:
            (day, month, year) = (str(self.exp_date)[6:8], str(self.exp_date)[4:6], str(self.exp_date)[0:4])
            d1 = date(int(year), int(month), int(day))
        else:
            d1 = self.exp_date

        return (d1 - d0).days / 365.0

    def get_price_delta(self):
        d1 = (math.log(self.s / self.k) + (self.rf + self.div + math.pow(self.vol, 2) / 2) * self.t) / (self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        if self.right == 'C':
            self.calc_price = (norm.cdf(d1) * self.s * math.exp(-self.div * self.t) - norm.cdf(d2) * self.k * math.exp(-self.rf * self.t))
            self.delta = norm.cdf(d1)
        elif self.right == 'P':
            self.calc_price = (-norm.cdf(-d1) * self.s * math.exp(-self.div * self.t) + norm.cdf(-d2) * self.k * math.exp(-self.rf * self.t))
            self.delta = -norm.cdf(-d1)

    def get_implied_vol(self):
        """
        Finding the implied volatility
        """
        ITERATIONS = 100
        ACCURACY = 0.05
        low_vol = 0
        high_vol = 1
        self.vol = 0.5  ## It will try mid point and then choose new interval
        self.get_price_delta()
        for i in range(ITERATIONS):
            if self.calc_price > self.price + ACCURACY:
                high_vol = self.vol
            elif self.calc_price < self.price - ACCURACY:
                low_vol = self.vol
            else:
                break
            self.vol = low_vol + (high_vol - low_vol) / 2.0
            self.get_price_delta()

        return self.vol

if __name__ == '__main__':
    # ===========================================================================
    # TO CHECK OPTION CALCULATIONS
    # ===========================================================================
    s = 110
    k = 115
    exp_date = '20150116'
    eval_date = '20140424'
    rf = 0.01
    price = 3.18
    vol = 0.15
    right = 'C'
    opt = Option(s=s, k=k, eval_date=eval_date, exp_date=exp_date, rf=rf, price=price, vol=0.3, right=right)
    ivol = opt.get_implied_vol()
    print("======== Implied Volatility:: " + str(ivol))