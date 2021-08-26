#!/usr/bin/env python
# coding: utf-8

# ## Import requirements and Gather Data

# https://www.youtube.com/watch?v=QIUxPv5PJOY&t=2213s

# In[52]:


# This example will use LSTM(Long Short Term Memory)
# Predicting AAPL using the past 60 day stock price


# In[53]:


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


# In[57]:


# Get Data
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17' )
df


# In[58]:


# Get number of rows and columns in data set
df.shape


# In[59]:


# Visualize closing price history 
plt.figure(figsize=(16,8))
plt.title('AAPL price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close USD Price', fontsize=18)


# ## Organize Data/Clean Data

# In[14]:


# Create a new dataframe with only the 'Close' cloumn
data = df.filter(['Close'])
# Convert data to numpy array 
dataset = data.values


dataset


# In[15]:


# Get number of rows to train the model on 
training_data_len = math.ceil(len(dataset)* .8)


training_data_len


# ## Prepare the data

# In[16]:


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


scaled_data


# ## Split the data

# In[13]:


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


# In[23]:


# Convert the x_train and y_train data into numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)


x_train


# In[24]:


x_train.shape


# In[22]:


# Reshape the data
x_train = np.reshape(x_train, (1543, 60, 1))
x_train


# In[29]:


x_train.shape[1]


# ## Build the model

# In[30]:


# Build the LSTM model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[31]:


# Compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')


# ## Train the model

# In[32]:


# Train the model 
model.fit(x_train, y_train, batch_size=1, epochs=2)


# ## Create Test Data set

# In[37]:


# Create the testing dataset 
# Create a new array containing scaled values from index 1543 to 2003, or training_data_len to last index in data
test_data = scaled_data[training_data_len-60: , :]


# In[38]:


# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# In[40]:


# Convert the data to numpy array
x_test = np.array(x_test)


# In[42]:


# Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# ## Get Predictions 

# In[43]:


# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[44]:


# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# ## Plot Data

# In[45]:


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# In[48]:


# Visualize the data 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')


# In[49]:


# Show the valid and predicted prices 
valid


# ## Predicting future price

# In[51]:


# Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')


# In[60]:


# Create new dataframe
new_df = apple_quote.filter(['Close'])


# In[61]:


# Get the last 60 day closing price calues and convert the dataframe to an array 
last_60_days = new_df[-60:].values 


# In[62]:


# Scale the data to be values between 0 and 1 
last_60_days_scaled = scaler.transform(last_60_days)


# In[63]:


# Create an empty list 
X_test = []


# In[64]:


# Append the last 60 days 
X_test.append(last_60_days_scaled)


# In[75]:


# Convert X_test dataset to numpy array 
X_test = np.array(X_test)
print(X_test)


# In[71]:


# Reshape data 
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))


# In[72]:


# Get the predicted scale price 
pred_price = model.predict(X_test)


# In[73]:


# Undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[74]:


apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])


# In[ ]:




