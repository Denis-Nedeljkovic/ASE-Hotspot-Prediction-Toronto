#/usr/bin/env python3

import cplex
import joblib
from docplex.mp.model import Model
import docplex
import keyboard
import pandas
from sklearn import linear_model
import sklearn as sk
import matplotlib.pyplot as plt

def wait():
    # Pauses program until enter is pressed

    print(
        '---------------------------------------------------------------------------------------------------------------')
    print('Press Enter to continue . . .')
    keyboard.wait('enter')
    print('')


def initialize():

    dataFile = pandas.read_csv('csv Files/trainingFile.csv')


    print(dataFile.head().to_string())
    print('\nIs there any NaN: ' + str(dataFile.isnull().values.any()))
    print('---------------------------------------------------------------------------------------------------------------')
    print(dataFile.describe().to_string())

    wait()

    return dataFile

def setupTrainingTest(data, output):
    # Splits data into training and test sets where x is all features but result, and y is just the result. Returns
    # xTrain, xTest, yTrain, yTest

    x = data.iloc[1:, :-9]
    x = pandas.concat([x, data.iloc[1:, -7:-5]], axis=1)
    y = data.iloc[1:, output]

    xTrain, xTest, yTrain, yTest = sk.model_selection.train_test_split(x, y, random_state = 0, test_size = 0.2)

    print('Training set and Test set has been created.')

    wait()

    return [xTest, yTest, xTrain, yTrain]


def createLinearRegression(data,title):

    regr = linear_model.LinearRegression()
    regr.fit(data[2], data[3])
    yPred = regr.predict(data[0])


    name = title + ' LR.save'

    joblib.dump(regr, 'models/' + name)

    return yPred


def modelmetrics(yPred, data):
    # calculates assorsted range of meitrcs using yPred and trainingDataList[1] (yTest) and outputs their results

    mae = sk.metrics.mean_absolute_error(data[1], yPred)
    mse = sk.metrics.mean_squared_error(data[1], yPred)
    r2 = sk.metrics.r2_score(data[1], yPred)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))

    print('')
    print('All metrics calculated')


    wait()

def graphIt(yPred, data):
    fig, ax = plt.subplots()
    ax.scatter(yPred, data[1], edgecolors=(0, 0, 1))
    ax.plot([data[1].min(), data[1].max()], [data[1].min(), data[1].max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

data = initialize()

#trainingDataListLong = setupTrainingTest(data, -1)
#trainingDataListLat = setupTrainingTest(data, -2)
trainingDataListTicket = setupTrainingTest(data, -3)

#yPredLat = createSVM(trainingDataListLat, 'Lat')
#yPredLong = createSVM(trainingDataListLong, 'Long')
yPredTicket = createLinearRegression(trainingDataListTicket, 'Ticket')

#graphIt(yPredLat, trainingDataListLat)
#graphIt(yPredLong, trainingDataListLong)
graphIt(yPredTicket, trainingDataListTicket)

#modelmetrics(yPredLat, trainingDataListLat)
#modelmetrics(yPredLong, trainingDataListLong)
modelmetrics(yPredTicket, trainingDataListTicket)









