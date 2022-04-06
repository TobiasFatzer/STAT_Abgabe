import numpy as np
import pandas as pd
import yfinance as yf  # Import from Yahoofinance API: https://python-yahoofinance.readthedocs.io/en/latest/api.htm
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame(yf.download(tickers=['^SSMI'], start='2000-01-01', end='2020-03-01'))

# Prepare Data and Add to Scale so that the Values go from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))

#Fit the Received Value to the before mentioned Scaler
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Get the last 60 Days to base the future prediction on
prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the Neural Model with 50 Units, Feeds back data twice
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')

#Create Model with 25 epochs
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Save Model so it can be used in Jupyter Notebook
model.save('stock_prediction.smi')
