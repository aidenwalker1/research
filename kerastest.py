import test as DataMiner
import numpy as np
import pandas as pd
import random
import time
random.seed(time.time())
# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, SimpleRNN, LSTM, TimeDistributed# for creating regular densely-connected NN layers and RNN layers

# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
import math # to help with data reshaping of the data

# Sklearn
import sklearn # for model evaluation
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import mean_squared_error # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from matplotlib import pyplot

def plot_data(data, y) :
    #heartrate = 2
    

    #x = [i for i in range(len(data))]
   
    x = [data[i][8] for i in range(0,len(data))]
    #x1 = [data[i][15] for i in range(0,len(data))]
    #x2 =[data[i][18] for i in range(0,len(data))]
    pyplot.scatter(y,x)
    #pyplot.scatter(y,x1)
    #pyplot.scatter(y,x2)
    pyplot.show()
    return
    ax.plot(x, y, "r--")
    #ax.plot(x, y2, "b")
    pyplot.xlabel("Heartrate")
    pyplot.ylabel("Feeling")
    pyplot.title("Graph")
    pyplot.ylim([0, 6])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start, end, 30))
    pyplot.show()

dyad = "dyads/dyadH05A1w"
path = dyad + ".prompt_groups.json"
responses, audio_responses = DataMiner.read_prompts(path)

time_step = 1
data_length = 60000

data = DataMiner.get_data(data_length)
scaler = MinMaxScaler()

def lstm(data, responses) :
    features = len(data[0][0]) - 1

     # design network
    model = Sequential()
    model.add(LSTM(100, return_sequences=True,input_shape=(1, features)))
    model.add(LSTM(25))
    #model.add(TimeDistributed(Dense(20, activation='sigmoid')))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    train_data = data[:len(data)//2]
    test_data = data[len(data)//2:]
    for cur_data in train_data :
        npdata = np.array(cur_data)
        X = npdata[:,0:len(cur_data[0]) - 1]
        y = DataMiner.get_y(responses, cur_data, 2)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # X_train = X[0:data_length//2,:]
        # X_test = X[data_length//2:data_length:,:]
        # y_train = np.array(y[0:data_length//2])
        # y_test = np.array(y[data_length//2:data_length:])

        # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        model.fit(x=X,y=np.array(y), epochs=100,verbose=True)

    correct = 0
    top_2 = 0
    total = 0    

    for test in test_data :
        npdata = np.array(test)
        X = npdata[:,0:len(cur_data[0]) - 1]
        x_t = X.reshape(X.shape[0], 1, X.shape[1])
        y_t = DataMiner.get_y(responses, test, 2)

        ys = model.predict(x_t)

        if int(round(ys[-1][0])) == y_t[-1] :
            correct += 1
        elif abs(int(round(ys[-1][0])) - y_t[-1]) <= 1 :
            top_2 += 1
        else :
            print("correct:" + str(y_t[-1]) + "got:" + str(ys[-1][0]))
        total += 1
            

    print (correct / total)
    return model

def nn(data, responses) :
    #X = scaler.fit_transform(X)
    npdata = [data[i][j] for i in range(len(data)) for j in range(len(data[0]))]
    npdata = np.array(npdata)
    test_size = len(data)
    X_train = X[0:data_length//2,:]
    X_test = X[data_length//2:data_length:,:]
    acc = 0
    count = 7
    for i in range(1,8) :
        print('i:'+str(i))
        y = np.array(DataMiner.get_y(responses, data, i))
        indices = np.where(y != -1)[0]
        y = y[indices]
        X = npdata[indices,0:len(data[0]) - 1]
        test_size = X.shape[0]
        X_train = X[0:data_length//2,:]
        X_test = X[data_length//2:data_length:,:]
        y_train = np.array(y[0:data_length//2])
        y_test = np.array(y[data_length//2:data_length:])
        #plot_data(data, y)
        clf, score = DataMiner.train(X_train,X_test,y_train,y_test)
        acc += score
    print(acc / count)

lstm(data, responses)

# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

def get_test_batches(X_test,y_test, batch_size, num_batches) :
    outX = []
    outY = []
    for i in range(0,num_batches) :
        index = random.randint(0,X_test.shape[0] - 1 - batch_size)

        batch_x = X_test[index:index+batch_size]
        batch_y = y_test[index:index+batch_size]
        outX.append(batch_x)
        outY.append(batch_y)
    return outX, outY



