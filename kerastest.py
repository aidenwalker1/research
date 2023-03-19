import test as DataMiner
import numpy as np
import pandas as pd
import random
import time
from sklearn.feature_selection import SequentialFeatureSelector
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
    y1 = [data[i][92] for i in range(0,len(data))]
    mv = max(y1)
    x = [i for i in range(len(data))]
    y = [y[i]*(mv/5) for i in range(len(y))]
    
    #x1 = [data[i][15] for i in range(0,len(data))]
    #x2 =[data[i][18] for i in range(0,len(data))]
    #pyplot.scatter(y,x)
    #pyplot.scatter(y,x1)
    #pyplot.scatter(y,x2)
    #pyplot.show()
    #return
    fig, ax = pyplot.subplots()
    ax.plot(x, y, "r--")
    ax.plot(x, y1, "b")
    pyplot.xlabel("Heartrate")
    pyplot.ylabel("Feeling")
    pyplot.title("Graph")
    #pyplot.ylim([0, ])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 1))
    pyplot.show()

dyad = "dyads/dyadH05A1w"
path = dyad + ".prompt_groups.json"
responses, audio_responses = DataMiner.read_prompts(path)

time_step = 1
data_length = 120000

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
            print("correct:" + str(y_t[-1]) + "got:" + str(ys[-1][0]))
            top_2 += 1
        else :
            print("correct:" + str(y_t[-1]) + "got:" + str(ys[-1][0]))
        total += 1
            

    print (correct / total)
    print((top_2 + correct) / total)
    return model

def nn(all_data, responses) :
    data = np.array(all_data)
    ys = [[data[i,-1]] for i in range(data.shape[0])]
    alls = []
    cur = np.zeros(shape=(8, data.shape[0], data.shape[1] - 1))

    for i in range(data.shape[1] - 1) :
        
        n = np.array(data[:,i])
        if n.size == 0 :
            print('continue' + str(i))
            continue
        npdata = np.zeros(shape=(data.shape[0], len(data[i][0])))
        for j in range(data.shape[0]) :
            npdata[j] = n[j]

        for j in range(1,8) :
            if j == 5 or j == 6 :
                continue

            y = np.array(DataMiner.get_y(responses, ys, j))
            indices = np.where(y != -1)[0]
            y = y[indices]
            if y.size == 0 :
                continue
            X = npdata[indices,0:]
            
            data_length = X.shape[0]
            X_train = X[0:data_length//2,:]
            X_test = X[data_length//2:data_length:,:]
            y_train = np.array(y[0:data_length//2])
            y_test = np.array(y[data_length//2:data_length:])
            
            clf, score = DataMiner.train(X,X,y,y)
            preds = clf.predict(X)
            cur[j,:,i] = preds
    acc = 0
    print()
    for i in range(1,8) :
        if i == 5 or i == 6 :
            continue
        y = np.array(DataMiner.get_y(responses, ys, i))
        data_length = data.shape[0]
        indices = np.where(y != -1)[0]
        y = y[indices]
        if y.size == 0 :
            return
        X = cur[i,indices,:]
        X_train = X[0:data_length//2,:]
        X_test = X[data_length//2:data_length:,:]
        y_train = np.array(y[0:data_length//2])
        y_test = np.array(y[data_length//2:data_length:])
        print('i:'+str(i))
        clf, score = DataMiner.train(X_train,X_test,y_train,y_test)
        acc += score
    print('acc:' + str(acc/5))

        

def easy_nn(data, responses) :
    npdata = np.array(data)
    data_length = npdata.shape[0]
    X = npdata[:,:len(data[0]) - 1]
    X_train = X[0:data_length//2,:]
    X_test = X[data_length//2:data_length:,:]
    
    acc = 0
    count = 0

    for j in range(1,8) :
        if j == 5 or j == 6 :
            continue
        print('j:'+str(j))
        y = np.array(DataMiner.get_y(responses, data, j))
        indices = np.where(y != -1)[0]
        y = y[indices]
        if y.size == 0 :
            print('inner continue ' + str(j))
            continue
        X = npdata[indices,:len(data[0]) - 1]
        data_length = X.shape[0]
        X_train = X[0:data_length//2,:]
        X_test = X[data_length//2:data_length:,:]
        y_train = np.array(y[0:data_length//2])
        y_test = np.array(y[data_length//2:data_length:])
        
        clf, score = DataMiner.train(X_train,X_test,y_train,y_test)
        acc += score
    print('average acc: ' + str(acc / 5))
        
nn(data, responses)

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



