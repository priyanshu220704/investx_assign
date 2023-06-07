#!/usr/bin/env python
# coding: utf-8

# In[817]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime


# In[845]:


df = pd.read_csv('C:\Train.csv')
print(df.head())
df=df.head(150)


# In[819]:


short_period = 12  
long_period = 26  
signal_period = 9 
ema_short = df['price'].rolling(window=short_period).mean()
ema_long = df['price'].rolling(window=long_period).mean()
macd_line = ema_short - ema_long
signal_line = macd_line.rolling(window=signal_period).mean()
histogram = macd_line - signal_line
df['liabilities'] = macd_line
df.fillna(df.mean(), inplace=True)
print(df)


# In[820]:





# In[821]:


cols = list(df_1)[2:12]
print(cols)


# In[822]:


df_for_training = df[cols].astype(float)


# In[823]:


scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


# In[824]:


trainX = []
trainY = []


# In[825]:


n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14 # Number of past days we want to use to predict the future.


# In[826]:


for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 9])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


# In[827]:


df_for_training_scaled


# In[828]:


trainY


# In[829]:


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)) 
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# In[830]:


history = model.fit(trainX, trainY, epochs=5, batch_size=3, validation_split=0.1, verbose=1)


# In[831]:


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


# In[832]:


from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# In[833]:


df2= df.head(100)
train_dates = pd.to_datetime(df2['Date'])
print(train_dates.tail(15)) 


# In[834]:


n_past = 1
n_days_for_prediction=51  #let us predict past 15 days

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='M').tolist()
print(predict_period_dates)


# In[835]:


prediction = model.predict(trainX[-n_days_for_prediction:]) 


# In[836]:


prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,9]


# In[837]:


forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'price':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


# In[838]:


original = df[['Date', 'price']]
original['Date']=pd.to_datetime(original['Date'])
original


# In[844]:


df_forecast=df_forecast['price']
df_forecast.to_numpy()


# In[840]:


original.plot(x='Date',y='price')
original= pd.concat([original,df_forecast],axis=0)
original.plot(x='Date',y='price')
#f_forecast.plot(x='Date',y='price')


# In[841]:


df = df[['price']]
og = df.loc[99:149]


# In[842]:


og.to_numpy()


# In[843]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse = mean_squared_error(og , df_forecast)
#print(100-mse)
rmse = np.sqrt(mse)
print(100-rmse)
mae = mean_absolute_error(og, df_forecast)
#print(100-mae)
r2 = r2_score(og, df_forecast)
#print(r2)


# In[ ]:





# In[ ]:




