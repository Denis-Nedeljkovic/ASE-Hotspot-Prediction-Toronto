import numpy


import pandas
import keyboard
import sklearn
from sklearn import svm


def wait():
    # Pauses program until enter is pressed

    print(
        '---------------------------------------------------------------------------------------------------------------')
    print('Press Enter to continue . . .')
    keyboard.wait('enter')
    print('')


def initialize():
    # Initalizes data file as diabetesFile, outputs the first five rows, checks if there are any missing values, outputs
    # basic statical analysis measures, count, mean, standerd diviation, minimun, first quartile, median, third quartile,
    # maximun, and creates variable dataClean. Returns diabetesFile and dataClean

    diabetesFile = pandas.read_csv('csv Files/diabetes.csv')

    print(diabetesFile.head().to_string())
    print('\nIs there any NaN: ' + str(diabetesFile.isnull().values.any()))
    print('---------------------------------------------------------------------------------------------------------------')
    print(diabetesFile.describe().to_string())

    wait()

    dataClean = ['Glucose', 'BloodPressure', 'SkinThickness']

    return [diabetesFile, dataClean]


def cleanData(diabetesFile, dataClean):
    # Cleans diabetesFile with columns Glucose, Blood Pressure, SkinThickness found in dataClean. Cleaning is done by
    # replacing any 0s found in the three coloumns and replacing it with NaN, calculating its mean, then replacing the
    # NaN with the mean. Outputs outputs the first five rows, and same basic statitcs from prevoious function. Returns
    # the updated diabetesFile

    for clean in dataClean:

        diabetesFile[clean] = diabetesFile[clean].replace(0, numpy.NAN)
        temp_mean = int(diabetesFile[clean].mean())
        diabetesFile[clean] = diabetesFile[clean].replace(numpy.NAN,temp_mean)

        print('Data column: ' + clean + ' . . . cleaned')

        wait()

    print(diabetesFile.head().to_string())
    print('')
    print('Data has been updated')
    print(
        '---------------------------------------------------------------------------------------------------------------')
    print(diabetesFile.describe().to_string())

    wait()

    return diabetesFile


def setupTrainingTest(data):
    # Splits data into training and test sets where x is all features but result, and y is just the result. Returns
    # xTrain, xTest, yTrain, yTest

    x = data.iloc[:, :-2]
    y = data.iloc[:, -1]

    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, random_state = 0, test_size = 0.2)

    print('Training set and Test set has been created.')

    wait()

    return [xTest, yTest, xTrain, yTrain]


def createSVM(trainingDataList):
    # creates SVM Radial Basis Function Kernel for classifiction. Rturns prediction value yPred

    model = svm.SVC(kernel='rbf')
    model.fit(trainingDataList[2], trainingDataList[3])
    yPred = model.predict(trainingDataList[0])

    print('SVM model created')

    wait()

    return yPred


def modelmetrics(yPred, trainingDataList):
    # calculates assorsted range of meitrcs using yPred and trainingDataList[1] (yTest) and outputs their results

    print("Accuracy: " + str(sklearn.metrics.accuracy_score(trainingDataList[1], yPred)))
    print("Confusion Matrix: " + str(sklearn.metrics.confusion_matrix(trainingDataList[1], yPred)))
    print("Balanced Accuracy: " + str(sklearn.metrics.balanced_accuracy_score(trainingDataList[1], yPred)))
    print("Top k Accuracy: " + str(sklearn.metrics.top_k_accuracy_score(trainingDataList[1], yPred, k=1)))
    print("Average Precision: " + str(sklearn.metrics.average_precision_score(trainingDataList[1], yPred))) #macro
    print("Brier Score Loss: " + str(sklearn.metrics.brier_score_loss(trainingDataList[1], yPred)))
    print("F1 Binary: " + str(sklearn.metrics.f1_score(trainingDataList[1], yPred)))
    print("Log Loss: " + str(sklearn.metrics.log_loss(trainingDataList[1], yPred)))
    print("Precision : " + str(sklearn.metrics.precision_score(trainingDataList[1], yPred)))
    print("Recall : " + str(sklearn.metrics.recall_score(trainingDataList[1], yPred)))
    print("Jaccard similarity coefficient : " + str(sklearn.metrics.jaccard_score(trainingDataList[1], yPred)))
    print("Receiver Operating Characteristic Curve: " + str(sklearn.metrics.roc_auc_score(trainingDataList[1], yPred)))

    print('')
    print('All metrics calculated')

    wait()


dataList = initialize()
data = cleanData(dataList[0],dataList[1])
trainingDataList = setupTrainingTest(data)
yPred = createSVM(trainingDataList)
modelmetrics(yPred, trainingDataList)

exit(0)
