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
import mplfinance as mpf

sequence_length = 50
day_num = 800
feature_num = 18
ts_code = '000001.sh'

# 数据获取
pro = ts.pro_api('666d30165c823e44cf89a793c48158e1c9c5fa7ea0231793550c570a')
df = pro.index_daily(ts_code=ts_code, start_date='20000101', end_date='20210101')
# df.to_csv('df.csv')
df['volume'] = df.vol
df['date'] = pd.to_datetime(df['trade_date'])
df.set_index(['date'], inplace=True)
df = df.sort_index()

# ta-lib特征提取
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
# feature.to_csv('feature.csv')
# 特征预处理
feature = feature[-day_num:]
scaler = MinMaxScaler()
feature_scaler = scaler.fit_transform(feature)
x = []
y = []
for i in range(day_num - sequence_length):
    x.append(feature_scaler[i: i + sequence_length])
    y.append(feature_scaler[i + sequence_length][0])
x = np.array(x)
y = np.array(y)
x = x.reshape(day_num - sequence_length, sequence_length, feature_num)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.05, random_state=0,
                                                                            shuffle=False)
# 建立序列化模型
model = Sequential()
model.add(LSTM(input_shape=(sequence_length, feature_num), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()

# 训练-拟合
h = model.fit(x_train, y_train, batch_size=16, epochs=30, validation_data=(x_test, y_test))
model.save('m.h5')

# 预测
y_pred = model.predict(x_test)
scaler.fit_transform(pd.DataFrame(feature['close'].values))
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
total = y_pred.shape[0]
success = 0
for i in range(1, total):
    if (y_pred[i][0] - y_test[i - 1][0]) * (y_test[i][0] - y_test[[i - 1]][0]) > 0:
        success = success + 1
print('涨跌准确率:', success / (total - 1))

# 评估模型
print('均方误差:', metrics.mean_squared_error(y_test, y_pred))
print('均方根误差:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('平均绝对误差:', metrics.mean_absolute_error(y_test, y_pred))

# 画出预测图 损失值图 k线图
fig2 = plt.figure(1)
plt.plot(y_pred, 'g-')
plt.plot(y_test, 'r-')
plt.legend(['predict', 'true'])
plt.savefig('预测.png')

loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png', bbox_inches='tight', dpi=300)

my_color = mpf.make_marketcolors(up='red', down='green', edge='i', volume='in', inherit=True)
my_style = mpf.make_mpf_style(gridaxis='both', gridstyle='-.', y_on_right=True, marketcolors=my_color)
mpf.plot(df, type='candle', style=my_style, volume=True, figratio=(2, 1), savefig='k线图.png', figscale=5)
