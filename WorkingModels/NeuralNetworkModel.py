import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")

df = pd.read_csv("@MES00_Micro-e-mini-S&P_Day_20years.txt")

print(df)
df.shape
df.describe()


# Visualize the closing price history 
plt.figure(figsize=(16,8))
plt.title("Closing Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df["Close"].values.reshape(-1,1))
scaled_df



training_data_len = math.ceil(len(scaled_df) * .8)
training_data_len 



# create training data set
# create the scaled training dataset
train_data = scaled_df[0:training_data_len, :]
# split data into x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60: i, 0])
    y_train.append(train_data[i,0])
    #print(x_train)
    #print(y_train)
    print()


# convert x_train and y_train to numpy arrays
x_train= np.array(x_train)
y_train = np.array(y_train)


# reshape the data so tensorflow will accept it (has to be 3 demensional)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)



# build the LSTM Model  
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))



# compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model 
model.fit(x_train, y_train, batch_size=1, epochs=1)







# create the testing data set
# create a new array containing scaled values from index 1543 to 2003
test_data = scaled_df[training_data_len - 60:, :]
x_test = []
y_test = test_data[training_data_len: ]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# convert data to numpy array 
x_test = np.array(x_test)

# reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get models predicted price values
predictions = model.predict(x_test)
inversedPredictions = scaler.inverse_transform(predictions)





# plot test predictions
plt.figure(figsize=(16,8))
plt.plot(y_test, color='black')
plt.plot(inversedPredictions, color = 'green')
plt.title('S%P 500 Daily Timeseries')
plt.xlabel('Time')
plt.ylabel('Price (Points)')
plt.legend()
plt.show()
