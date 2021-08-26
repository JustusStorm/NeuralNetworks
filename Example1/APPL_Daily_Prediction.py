# This example will use LSTM(Long Short Term Memory)
# Predicting AAPL using the past 60 day stock price

#Import Libraries 
import math 
import pandas_datareader as web 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# Get Data
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17' )

# Get number of rows and columns in data set
print(df.shape)

# Visualize closing price history 
plt.figure(figsize=(16,8))
plt.title('AAPL price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close USD Price', fontsize=18)


# Create a new dataframe with only the 'Close' cloumn
data = df.filter(['Close'])
# Convert data to numpy array 
dataset = data.values


# Get number of rows to train the model on 
training_data_len = math.ceil(len(dataset)* .8)



# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# Create the training data set 
# Create the scaled  training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 61: 
        print(x_train)
        print(y_train)
        print()


# Convert the x_train and y_train data into numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)


# Reshape the data
x_train = np.reshape(x_train, (1543, 60, 1))


# Build the LSTM model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# Compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')



# Train the model 
model.fit(x_train, y_train, batch_size=1, epochs=2)


# Create the testing dataset 
# Create a new array containing scaled values from index 1543 to 2003, or training_data_len to last index in data
test_data = scaled_data[training_data_len-60: , :]



# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# Convert the data to numpy array
x_test = np.array(x_test)


# Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# Visualize the data 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')


# Show the valid and predicted prices 
print(valid)