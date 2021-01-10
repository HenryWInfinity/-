import tensorflow
from keras.models import load_model
import tushare as ts
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import mplfinance as mpf
import talib as ta

sequence_length = 50
feature_num = 18
ts_code = '000001.sh'

pro = ts.pro_api('666d30165c823e44cf89a793c48158e1c9c5fa7ea0231793550c570a')
df = pro.index_daily(ts_code=ts_code, start_date='20000101')
df['volume'] = df.vol
df['date'] = pd.to_datetime(df['trade_date'])
df.set_index(['date'], inplace=True)
df = df.sort_index()

feature = pd.DataFrame()
feature['close'] = df.close
feature['open'] = df.open
feature['high'] = df.high
feature['low'] = df.low
feature['vol'] = df.vol
feature['pct_chg'] = df.pct_chg
feature['slowk'], feature['slowd'] = ta.STOCH(df.high, df.low, df.close, fastk_period=9,
                                              slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
feature['upper'], feature['middle'], feature['lower'] = ta.BBANDS(df.close, timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                  matype=0)
feature['macd'], feature['macdsignal'], feature['macdhist'] = ta.MACD(df.close, fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)
feature['ATR'] = ta.ATR(df.high, df.low, df.close, timeperiod=14)
feature['RSI'] = ta.RSI(df.open, timeperiod=14)
feature['ROC'] = ta.ROC(df.close, timeperiod=10)
feature['OBV'] = ta.OBV(df.close, df.vol)

feature = feature[-sequence_length:]

df = df[-sequence_length:]

# 设置k线图颜色
my_color = mpf.make_marketcolors(up='red',  # 上涨时为红色
                                 down='green',  # 下跌时为绿色
                                 edge='i',  # 隐藏k线边缘
                                 volume='in',  # 成交量用同样的颜色
                                 inherit=True)

my_style = mpf.make_mpf_style(gridaxis='both',  # 设置网格
                              gridstyle='-.',
                              y_on_right=True,
                              marketcolors=my_color)

mpf.plot(df, type='candle',
         style=my_style,
         volume=True,  # 展示成交量副图
         figratio=(2, 1),  # 设置图片大小
         savefig='近' + str(sequence_length) + '天k线图.png',
         figscale=5)

scaler = MinMaxScaler()
feature_scaler = scaler.fit_transform(feature)
x = []
x.append(feature_scaler[0:sequence_length])
x = np.array(x)
x = x.reshape(1, sequence_length, feature_num)
print(feature)
model = load_model('m.h5')
y_pred = model.predict(x)
scaler.fit_transform(pd.DataFrame(feature['close'].values))
y_pred = scaler.inverse_transform(y_pred)
print('预测明日收盘价:', y_pred[0][0])
if y_pred[0][0] > list(feature['close'])[sequence_length-1]:
    print('上涨')
else:
    print('下跌')
