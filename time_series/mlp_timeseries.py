# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import numpy
import matplotlib.pyplot as plt
import pandas
import time
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
#filedata= 'C:/Users/C00310965/Documents/livros/source/train_lago_sul_20160907_20160930.csv'

filedata= 'C:/Users/C00310965/Documents/livros/source/train_test_lago_sul_20160907_20160930.csv'
#filedata= 'C:/Users/C00310965/Documents/livros/source/load_bafsa.csv'
#filedata= 'C:/Users/C00310965/Documents/livros/source/hs_users.csv'
#filedata= 'C:/Users/C00310965/Documents/livros/source/hs_users_prsc.csv'
##2,4
dataframe = pandas.read_csv(filedata, usecols=[4], engine='python', sep=',')
dataset = dataframe.values
dataset = dataset.astype('float32')

start_time = time.time()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.674682)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# reshape dataset
look_back = 36
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(15, input_dim=look_back, activation='relu'))
#model.add(Dense(3, init='uniform', activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
# Estimate model performance

#trainScore = model.evaluate(trainX, trainY, verbose=0)
#print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
#print(testX)
#print("HAHAHA")
#print(testY)
testScore = model.evaluate(testX, testY, verbose=0)
print(testScore)
#print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
##trainPredict = scaler.inverse_transform(trainPredict)
##trainY = scaler.inverse_transform([trainY])
##testPredict = scaler.inverse_transform(testPredict)
##testY = scaler.inverse_transform([testY])
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
plt.plot(dataset,label='Original')
plt.plot(trainPredictPlot,label='Train Prediction')
plt.plot(testPredictPlot,label = 'Test Prediction')
plt.legend(loc='best')
plt.title('Throughput HSDPA: Multilayer Perceptron')
plt.show()
