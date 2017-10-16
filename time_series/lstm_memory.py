# LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset

#filedata= 'C:/Users/C00310965/Documents/livros/source/train_test_lago_sul_20160907_20160930.csv'
#filedata= 'C:/Users/C00310965/Documents/livros/source/load_bafsa.csv'
filedata= 'C:/Users/C00310965/Documents/livros/source/hs_users_prsc.csv'
#skipfooter=3
dataframe = pandas.read_csv(filedata, usecols=[1], engine='python', sep=';',skipfooter=1)
dataset = dataframe.values
dataset = dataset.astype('float32')

start_time = time.time()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
#train_size = int(len(dataset) * 0.75)
train_size = int(len(dataset) * 0.674682)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
for i in range(100):
	model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
#testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
#print('Test Score before inverting: %.2f RMSE' % (testScore))

testScore = model.evaluate(testX, testY, verbose=0,batch_size=1)
print(testScore)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#print(trainY[0])
#print("hahaha")
# calculate root mean squared error
##trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
##print('Train Score: %.2f RMSE' % (trainScore))
##testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
##testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
##print('Test Score: %.2f RMSE' % (testScore))

###
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

####time taken
timestamp = (time.time() - start_time)
print("time: "+str(timestamp))

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset),label='Original')
plt.plot(trainPredictPlot,label='Train Prediction')
plt.plot(testPredictPlot,label = 'Test Prediction')
plt.legend(loc='best')
plt.title('LSTM Memory')
plt.show()
