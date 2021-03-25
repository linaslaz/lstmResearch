import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout

import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')

# conda activate python38
#conda install -c conda-forge keras

def make_date(datestr):
    return pd.datetime.strptime(datestr, '%m/%d/%Y %H:%M')

def make_date2(datestr):
    return pd.datetime.strptime(datestr, '%m/%d/%Y')

def loadOHLC(file):

    dataset = pd.read_csv(file,
        header = 0,
        delimiter = ",",
        date_parser = make_date2,
        names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume" ],
        #"Date", "Time", "Open", "High", "Low", "Close", "Vol", "OI", "Avg", "Avg"

        #parse_dates = [["Date", "Time"]],
        parse_dates=["Date"],

        usecols = [0, 1, 2, 3, 4, 5, 6],
        index_col = 0)

    return dataset

import os
def printToFile(dff, fname):
    filename = os.path.join('c:\!TradeStation', fname + ".csv2")

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "w") as csv_file:

        for index, value in dff.items():

            s = str(index) + "," + str(value)
            s = s + '\n'
            csv_file.write(s)

file = 'c:/!TradeStation/ES_d1_10k_.txt'
file1 = 'c:/!TradeStation/GC_d1_10k_.txt'
file2 = 'c:/!TradeStation/CL_d1_10k_.txt'
file3 = 'c:/!TradeStation/FDAX_d1_10k_.txt'

fut = {}

fut['ES'] = loadOHLC(file)
fut['GC'] = loadOHLC(file1)
fut['CL'] = loadOHLC(file2)
fut['FDAX'] = loadOHLC(file3)

fuName = 'FDAX'
#stock = 'AMZN' #'AAPL'
#df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2020-12-31')
df = fut[fuName]

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

data = df.filter(['Close'])#Converting the dataframe to a numpy array
dataset = data #.values#Get /Compute the number of rows to train the model on

# d2 = dataset.pct_change() #* 50.0 # diff * bigpoint
# d2 = d2.dropna()

dataset = dataset.values

training_data_len = math.ceil(len(dataset) *.8)

step = 12

train_data = dataset[0:training_data_len , : ]

#y = (x - min) / (max - min)

#y = (x - mean) / standard_deviation


x_train=[]
y_train = []
for i in range(step, len(train_data)):
    p = train_data[i-step:i,0]
    pn = train_data[i,0]

    a = (p - np.min(p)) / (np.max(p) - np.min(p))
    an = (pn - np.min(p)) / (np.max(p) - np.min(p))

    # a = (p - np.mean(p)) / np.std(p)
    # an = (pn - np.mean(p)) / np.std(p)

    # a = np.log(p)
    # an = np.log(pn)

    # a = (p / p[0]) - 1
    # an = (pn / p[0]) - 1

    x_train.append(a)
    y_train.append(an)

    # if (train_data[i,0] > 0):
    #     #y_train.append(1.)  # longas
    #     y_train.append([0., 1.])  # longas
    # else:
    #     #y_train.append(-1.)
    #     y_train.append([1., 0.])

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(x_train.shape)

#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# new
model.add(Activation("linear"))
#model.add(Activation("relu"))


#Compile the model

#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(loss="mse", optimizer="rmsprop")

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)

#Test data set
test_data = dataset[training_data_len - step: , : ]#Create the x_test and y_test data sets

x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(step,len(test_data)):
    p = test_data[i - step:i, 0]
    a = (p - np.min(p)) / (np.max(p) - np.min(p))
    #a = (p - np.mean(p)) / np.std(p)
    #a = np.log(p)

    #a = (p / p[0]) - 1
    x_test.append(a)

#Convert x_test to a numpy array
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions = model.predict(x_test)

#Get the root mean squared error (RMSE), which is a good measure of how accurate the model is. A value of 0 would indicate that the models predicted values match the actual values from the test data set perfectly.
#The lower the value the better the model performed. But usually it is best to use other metrics as well to truly get an idea of how well the model performed.

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
print('rmse', rmse)

train = data[:training_data_len]
valid = data[training_data_len:]

#Visualize the data

valid['predictions'] = predictions
valid['predictions_diff'] = valid['predictions'].diff()
valid['predicted_signal'] = np.where(valid['predictions_diff'] > 0, 1, -1)

cc = train['Close'].tail(1).values[0]

valid['returns'] = valid.Close.diff()
valid['strategy_returns'] = valid.returns * valid.predicted_signal.shift(1)

valid['strategy_res'] = valid['strategy_returns'].cumsum() + cc

valid['strategy_res'].plot()
plt.show()


plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'strategy_res']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

printToFile(valid['predicted_signal'], fuName)

