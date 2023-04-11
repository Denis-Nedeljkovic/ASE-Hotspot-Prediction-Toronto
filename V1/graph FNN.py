import keyboard
import matplotlib.pyplot as plt
import pandas
from tensorflow import keras
import sklearn as sk
import sklearn.model_selection


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

def setupTrainingTest(data):
    # Splits data into training and test sets where x is all features but result, and y is just the result. Returns
    # xTrain, xTest, yTrain, yTest

    x = data.iloc[1:, :-9]
    x = pandas.concat([x, data.iloc[1:, -7:-5]], axis=1)
    y = data.iloc[1:, -3:-2]

    import sklearn.model_selection
    xTrain, xTest, yTrain, yTest = sk.model_selection.train_test_split(x, y, random_state = 0, test_size = 0.2)

    print('Training set and Test set has been created.')

    wait()

    return [xTest, yTest, xTrain, yTrain]

def loadModel(data):
    model = keras.models.load_model('models/ANN Ticket 195')
    yPred = model.predict(data[0])

    return yPred

def graphIt(yPred, data):
    fig, ax = plt.subplots()
    ax.scatter(yPred, data[1], edgecolors=(0, 0, 1))
    ax.plot([data[1].min(), data[1].max()], [data[1].min(), data[1].max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

data = initialize()

trainingDataList = setupTrainingTest(data)

yPred = loadModel(trainingDataList)

graphIt(yPred, trainingDataList)