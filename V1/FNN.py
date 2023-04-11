import csv
import time
import keyboard
import matplotlib
import pandas
import pandas as pd
import sklearn
import sklearn.model_selection as ms
import tensorflow as tf
import torch
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import warnings

from pasta.augment import inline

warnings.simplefilter(action='ignore', category=FutureWarning)


from numpy.random import seed
seed(1)


def wait():
    # Pauses program until enter is pressed

    print(
        '-------------------------------------------------------------------------------------------------------------')
    print('Press Enter to continue . . .')
    keyboard.wait('enter')
    print('')


def gpuCheck():

    use_cuda = torch.cuda.is_available()

    print(tf.config.list_physical_devices())

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)

    else:

        print("CUDA compatible GPU not available")
        print("Terminating program")
        exit(0)

    wait()

def initialize():



    trainingFile = pandas.read_csv('csv Files/trainingFile.csv')

    print(trainingFile.head().to_string())
    print('\nIs there any NaN: ' + str(trainingFile.isnull().values.any()))
    print('-----------------------------------------------------------------------------------------------------------')
    print(trainingFile.describe().to_string())

    wait()

    return trainingFile


def setupTrainingTest(data):
    # Splits data into training and test sets where x is all features but result, and y is just the result. Returns
    # xTrain, xTest, yTrain, yTest

    x = data.iloc[1:, :-9]
    x = pandas.concat([x, data.iloc[1:, -7:-5]], axis=1)
    y = data.iloc[1:, -3:-2]
    print(y)

    xTrain, xTest, yTrain, yTest = ms.train_test_split(x, y, random_state=0, test_size = 0.2)


    print('Training set and Test set has been created.')

    wait()

    return [xTrain, xTest, yTrain, yTest]


# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    # Defining the list of hyper parameters to try
    batch_size_list = list(range(0, 100, 5))
    epoch_list = list(range(0,100,5))

    import pandas as pd

    SearchResultsData = []

    start = time.time()

    # initializing the trials
    TrialNumber = 0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=(X_train.shape[1] + 1), input_dim=X_train.shape[1], kernel_initializer='normal', activation='swish'))

            # Defining the Second layer of the model

            # The output neuron is a single fully connected node
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))

            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Fitting the ANN to the Training set
            model.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial, verbose=0)

            mse = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test))
            R2 =  sklearn.metrics.r2_score(y_test,model.predict(X_test))

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'MSE:', mse, 'R2:', R2)

            SearchResultsData.append([TrialNumber, str(batch_size_trial) + '--' + str(epochs_trial), mse, R2, model])


    target_value = 5
    index = None
    for i, row in enumerate(SearchResultsData):
        if target_value >= float(row[2]) * 100:
            print(SearchResultsData[i])
            row[-1].save('models/ANN Ticket ' + str(row[0]))


    end = time.time()
    print("Total time elapsed: " + str((end - start)/60) + " minutes")
    return SearchResultsData

def create_HyperparameterAccFile(data):

    f = open('csv Files/hyperparameterAccTickets.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['TrialNumber', 'Parameters', 'Sensor Tickets MSE', 'R2'])
    for row in data:
        writer.writerow(row[:-1])

    f.close()

#gpuCheck()
trainingFileData = initialize()
splitsData = setupTrainingTest(trainingFileData)
searchResult = FunctionFindBestParams(splitsData[0], splitsData[2], splitsData[1], splitsData[3])
create_HyperparameterAccFile(searchResult)

exit(0)




