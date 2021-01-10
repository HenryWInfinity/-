import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, scale
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.losses import mape
import talib as ta
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import mplfinance as mpf


sequence_length = 30
day_num = 800
feature_num = 18
ts_code = '000001.sh'

# 数据获取和预处理
pro = ts.pro_api('666d30165c823e44cf89a793c48158e1c9c5fa7ea0231793550c570a')
df = pro.index_daily(ts_code=ts_code, start_date='20000101')
df['volume'] = df.vol
df['date'] = pd.to_datetime(df['trade_date'])
df.set_index(['date'], inplace=True)
df = df.sort_index()


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
         savefig='k线图.png',
         figscale=5)

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

feature = feature[-day_num:]
scaler = MinMaxScaler()
feature_scaler = scaler.fit_transform(feature)
x = []
y = []
temp = []
for i in range(day_num - sequence_length):
    x.append(feature_scaler[i: i + sequence_length])
temp = list(df.close)[-day_num + sequence_length-1:]
for i in range(len(temp)-1):
    if temp[i+1]-temp[i] > 0:
        y.append(1)
    else:
        y.append(0)
x = np.array(x)
y = np.array(y)

x = x.reshape(day_num - sequence_length, sequence_length, feature_num)
y = y.flatten()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=0, shuffle=False)
y_train = tf.keras.utils.to_categorical(y_train, 2)  # 将标签向量转化为one-hot形式的向量
y_test = tf.keras.utils.to_categorical(y_test, 2)

# 建立模型
model = Sequential()
model.add(LSTM(input_shape=(sequence_length, feature_num), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# 拟合
model.fit(x_train, y_train, batch_size=16, epochs=30, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
pred = []
act = []
pred = np.argmax(y_pred, axis=1)
act = np.argmax(y_test, axis=1)
print(pred)
print(act)
print('准确率:', sum(pred == act)/pred.size)



