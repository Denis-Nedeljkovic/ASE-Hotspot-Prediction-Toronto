import webbrowser
import numpy as np
import joblib
import keras
import keyboard
import pandas
import folium
import branca.colormap as cm
from geopy.geocoders import Nominatim
from IPython.display import HTML, display

scaler = joblib.load('MinMaxScalar.save')

LR_Tickets = joblib.load('models/Ticket LR.save')
LR_Lat = joblib.load('models/Lat LR.save')
LR_Long = joblib.load('models/long LR.save')

SVM_Tickets = joblib.load('models/Ticket SVR.save')
SVM_Lat = joblib.load('models/Lat SVR.save')
SVM_Long = joblib.load('models/Long SVR.save')

ANN_Tickets = keras.models.load_model('models/ANN Ticket 195')
ANN_Lat = keras.models.load_model('models/ANN Lat 33')
ANN_Long = keras.models.load_model('models/ANN Long 52')

output_file = "final.html"
m = folium.Map(location=[43.6532, -79.3832], zoom_start=15)


def wait():
    # Pauses program until enter is pressed
    print(
        '---------------------------------------------------------------------------------------------------------------')
    print('Press Enter to continue . . .')
    keyboard.wait('enter')
    print('')


def initialize():

    data = pandas.read_csv('csv Files/testingFile.csv')
    dataB = pandas.read_csv('csv Files/trainingFile.csv')
    x = data.iloc[1:, :-6]
    x = pandas.concat([x, data.iloc[1:, -4:-2]], axis=1)
    y = dataB.iloc[1:, -3:]

    print('All Data has been loaded')
    wait()

    print(y.head().to_string())
    return [x,y['Monthly_Median_Tickets'].values, y['Sensor_Latitude'].values, y['Sensor_Longitude'].values]


def pairData(Tickets, Lat, Long, scaler):

    data = pandas.read_csv('csv Files/testingFile.csv')
    data = data.iloc[1:,]
    data = data.to_numpy()

    setList = []

    i = 0
    while i < len(Tickets):
        temp = []

        for item in data[i]:
            temp.append(item)
        temp.append(Tickets[i])
        temp.append(Lat[i])
        temp.append(Long[i])

        setList.append(temp)
        i = i + 1

    setList = scaler.inverse_transform(setList)

    print('Prediction Done')
    wait()

    return setList


def graphSet(data, name, color):

    top_instances = data[np.argsort(data[:, -3])[::-1][:50]]

    print(top_instances[:,-3])
    for row in top_instances:
        TempLat = row[-2]
        TempLong = row[-1]
        folium.CircleMarker(location=[TempLat,TempLong],radius=5, color=color, fill=True).add_to(m)


    wait()


Data = initialize()
testData = Data[0]

True_Ticket = Data[1]
True_Lat = Data[2]
True_Long = Data[3]

LR_Tickets = LR_Tickets.predict(testData)
LR_Lat = LR_Lat.predict(testData)
LR_Long = LR_Long.predict(testData)

LR_Results = np.unique(np.array(pairData(LR_Tickets, LR_Lat, LR_Long, scaler)), axis=0)
graphSet(LR_Results,'LR', 'Black')

SVM_Tickets = SVM_Tickets.predict(testData)
SVM_Lat = SVM_Lat.predict(testData)
SVM_Long = SVM_Long.predict(testData)

SVM_Results = np.unique(np.array(pairData(SVM_Tickets, SVM_Lat, SVM_Long, scaler)), axis=0)
graphSet(SVM_Results, 'SVM', 'Blue')

ANN_Tickets = ANN_Tickets.predict(testData)
ANN_Lat = ANN_Lat.predict(testData)
ANN_Long = ANN_Long.predict(testData)

ANN_Results = np.unique(np.array(pairData(ANN_Tickets, ANN_Lat, ANN_Long, scaler)), axis=0)
graphSet(ANN_Results, 'ANN', 'Red')

True_Results = np.unique(np.array(pairData(True_Ticket, True_Lat, True_Long, scaler)), axis=0)
graphSet(True_Results,'True', 'Green')

m.save(output_file)
webbrowser.open(output_file, new=2)

exit(0)
