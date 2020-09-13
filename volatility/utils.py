
def get_percent_chg(df, w):
    out = w * [0]
    closes = list(df.Close)
    for i in range(w, len(df)):
        out.append((closes[i]-closes[i-w])/closes[i-w])
    df['ret_'+str(w)] = out