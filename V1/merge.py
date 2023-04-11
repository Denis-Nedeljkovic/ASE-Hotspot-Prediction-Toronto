import csv
import datetime
import statistics
import math
import random
import sys
import codecs
import re
import time

import geopy.distance
import joblib
import numpy
import numpy as np
import pandas
import requests
import pandas as pd
import openpyxl
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from geopy.geocoders import Photon
from geopy.geocoders import ArcGIS
import geopandas
from numpy import NaN

maxInt = sys.maxsize

# geolocator = Nominatim(user_agent="SpeedCameraSensorsToronto")
# geolocator2 = Photon(user_agent="SpeedCameraSensorsToronto")
geolocator3 = ArcGIS(user_agent="SpeedCameraSensorsToronto")

sensor_path = "csv Files/Automated Speed Enforcement - Monthly Charges"
# Features: Site Code, Location*, Enforcement Start Date, Enforcement End Date, Tickets per month

crime_path = "csv Files/neighbourhood-crime-rates - 4326"
# Features: _id, OBJECTID, HoodName, HoodID, . . . , AutoTheft_Rate2021[37], . . . , TheftfromMotorVehicle_Rate2021[-2],
# geometry[-1]

sign_path = "csv Files/Stationary Sign locations - 4326"

signMonthlyReadings_path = "csv Files/WYS Stationary monthly summary"

read_file = pd.read_excel(sensor_path + ".xlsx")
read_file.to_csv(sensor_path + ".csv", index=None, header=True)

sensor_list = []
crime_list = []
sign_list = []
signMonthlyReadings_list = []


def set_maxFileLimit(maxInt):

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)


def get_latLongV2(location):
    # Gets requested latitude and longitude coordinates for the nearst intersection with the provided location

    if ' Near ' in location:
        add = location.split(' Near ')
    elif ' North of ' in location:
        add = location.split(' North of ')
    elif ' East of ' in location:
        add = location.split(' East of ')
    elif ' South of ' in location:
        add = location.split(' South of ')
    elif ' West of ' in location:
        add = location.split(' West of ')
    else:
        raise ValueError(' Location value does not fit expected format')

    time.sleep(1)
    latlong = geolocator3.geocode(add[0] + ' & ' + add[1] + ', Toronto, ON')

    return [latlong.latitude, latlong.longitude]


def set_monthlyMedianTickets(sensor):

    ticketNumberList = []
    med = ''
    for i in sensor:
        if i.isdigit():
            ticketNumberList = ticketNumberList + [int(i)]

    if ticketNumberList:
        med = statistics.median(ticketNumberList)
    else:
        return ['','','']

    return [sensor[0], sensor[1], med]


def cleanSensorData():
    # New Features:Site Code, Location, Monthly Median Tickets

    temp1 = []

    for sensor in sensor_list: #~~~~~~~~~~~~~~~~~~~~~~~~
        if sensor[0] == '':
            return temp1
        else:
            temp = set_monthlyMedianTickets(sensor)
            temp[1] = get_latLongV2(temp[1])
            temp1.append(temp)
            print(temp)

    return temp1


def set_centerPoint(polygon):

    polygon = re.sub("[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz{}()':]", "", polygon)
    polygon = polygon.split(',')
    polygon.remove(' ')
    polygon.remove('')

    _x_list = [float(x) for x in polygon if float(x) > 0]
    _y_list = [float(y) for y in polygon if float(y) < 0]
    polygon_len = len(polygon) / 2
    _x = sum(_x_list) / polygon_len
    _y = sum(_y_list) / polygon_len

    return [_x, _y]


def cleanCrimeRate():
    # New Features: HoodName, AutoTheft_Rate, TheftfromMotorVehcile_Rate, location

    temp1 = []
    for crime in crime_list:
        temp = crime
        temp[-1] = set_centerPoint(crime[-1])
        temp = [temp[2], temp[36], temp[-2], temp[-1]]
        temp1.append(temp)

    return temp1


def create_translatorFile():

    f = open('csv Files/Hood_GPS_Pair.csv', 'w', newline='')
    writer = csv.writer(f)

    writer.writerow(['HoodName', 'Polygon Center point Latitude', 'Polygon Center point Longitude'])

    for crime in crime_list:
        temp = [crime[2], crime[-1][0], crime[-1][1]]
        writer.writerow(temp)

    f.close()


def matchCrimeToSensor(sensor):

    aux = []
    for crime in crime_list:
        aux.append(geopy.distance.geodesic(sensor[-2], crime[-1]))

    return aux.index(min(aux))


def matchCrimeToSign(sign):

    aux = []
    for crime in crime_list:
        aux.append(geopy.distance.geodesic([sign[-2], sign[-1]], crime[-1]))

    return aux.index(min(aux))


def matchSignToSensor(sensor):

    aux = []
    for sign in sign_list:
        aux.append(geopy.distance.geodesic(sensor[-2], [sign[-2],sign[-1]]))
    return aux.index(min(aux))

def create_trainingFile(sensor_list):

    f = open('csv Files/trainingFile.csv', 'w', newline='')
    writer = csv.writer(f)


    writer.writerow([ 'pct_05', 'pct_10', 'pct_15', 'pct_20', 'pct_25',  'pct_30', 'pct_35', 'pct_40', 'pct_45', 'pct_50',
                     'pct_55', 'pct_60', 'pct_65', 'pct_70', 'pct_75', 'pct_80', 'pct_85', 'pct_90', 'pct_95',
                     'spd_05', 'spd_10', 'spd_15', 'spd_20', 'spd_25', 'spd_30', 'spd_35', 'spd_40', 'spd_45', 'spd_50',
                     'spd_55', 'spd_60', 'spd_65', 'spd_70', 'spd_75', 'spd_80', 'spd_85', 'spd_90', 'spd_95', 'spd_100',
                     'sign_volume', 'sign_lat', 'sign_long', 'AutoTheft_Rate2021_monthly', 'TheftfromMotorVehcile_Rate2021_monthly',
                     'Polygon_Center_Point_Latitude', 'Polygon_Center_Point_Longitude', 'Monthly_Median_Tickets', 'Sensor_Latitude',
                     'Sensor_Longitude'])

    scalar = MinMaxScaler()

    temp_matrix = []

    for sensor in sensor_list:

        crimeIndex = matchCrimeToSensor(sensor)
        signIndex = matchSignToSensor(sensor)

        temp = [str(sign_list[signIndex][3]),str(sign_list[signIndex][4]),str(sign_list[signIndex][5]),str(sign_list[signIndex][6]),
                str(sign_list[signIndex][7]),str(sign_list[signIndex][8]),str(sign_list[signIndex][9]),str(sign_list[signIndex][10]),
                str(sign_list[signIndex][11]),str(sign_list[signIndex][12]),str(sign_list[signIndex][13]),str(sign_list[signIndex][14]),
                str(sign_list[signIndex][15]),str(sign_list[signIndex][16]),str(sign_list[signIndex][17]),str(sign_list[signIndex][18]),
                str(sign_list[signIndex][19]),str(sign_list[signIndex][20]),str(sign_list[signIndex][21]),
                str(sign_list[signIndex][23]),str(sign_list[signIndex][24]),str(sign_list[signIndex][25]),str(sign_list[signIndex][26]),
                str(sign_list[signIndex][27]),str(sign_list[signIndex][28]),str(sign_list[signIndex][29]),str(sign_list[signIndex][30]),
                str(sign_list[signIndex][31]),str(sign_list[signIndex][32]),str(sign_list[signIndex][33]),str(sign_list[signIndex][34]),
                str(sign_list[signIndex][35]),str(sign_list[signIndex][36]),str(sign_list[signIndex][37]),str(sign_list[signIndex][38]),
                str(sign_list[signIndex][39]),str(sign_list[signIndex][40]), str(sign_list[signIndex][41]),str(sign_list[signIndex][42]),
                str(sign_list[signIndex][43]), str(sign_list[signIndex][44]), str(sign_list[signIndex][45]),
                str(float(crime_list[crimeIndex][1]) / 12), str(float(crime_list[crimeIndex][2]) / 12), crime_list[crimeIndex][-1][0], crime_list[crimeIndex][-1][1], sensor[-1], sensor[1][0], sensor[1][1]]

        temp_matrix.append(temp)

    scaled = scalar.fit_transform(temp_matrix)

    for row in scaled:
        writer.writerow(row)

    joblib.dump(scalar, 'MinMaxScalar.save')
    f.close()


def create_testingFile(sign_list):

    f = open('csv Files/testingFile.csv', 'w', newline='')
    writer = csv.writer(f)

    writer.writerow(['pct_05','pct_10', 'pct_15', 'pct_20', 'pct_25',  'pct_30', 'pct_35', 'pct_40', 'pct_45', 'pct_50',
                     'pct_55', 'pct_60', 'pct_65', 'pct_70', 'pct_75', 'pct_80', 'pct_85', 'pct_90', 'pct_95',
                     'spd_05', 'spd_10', 'spd_15', 'spd_20', 'spd_25', 'spd_30', 'spd_35', 'spd_40', 'spd_45', 'spd_50',
                     'spd_55', 'spd_60', 'spd_65', 'spd_70', 'spd_75', 'spd_80', 'spd_85', 'spd_90', 'spd_95', 'spd_100',
                     'sign_volume', 'sign_lat', 'sign_long', 'AutoTheft_Rate2021_monthly', 'TheftfromMotorVehcile_Rate2021_monthly',
                     'Polygon_Center_Point_Latitude', 'Polygon_Center_Point_Longitude'])

    scalar = MinMaxScaler()

    temp_matrix = []

    for sign in sign_list:

        crimeIndex = matchCrimeToSign(sign)

        temp = [str(sign[3]),str(sign[4]),str(sign[5]),str(sign[6]),
                str(sign[7]),str(sign[8]),str(sign[9]),str(sign[10]),
                str(sign[11]),str(sign[12]),str(sign[13]),str(sign[14]),
                str(sign[15]),str(sign[16]),str(sign[17]),str(sign[18]),
                str(sign[19]),str(sign[20]),str(sign[21]),
                str(sign[23]),str(sign[24]),str(sign[25]),str(sign[26]),
                str(sign[27]),str(sign[28]),str(sign[29]),str(sign[30]),
                str(sign[31]),str(sign[32]),str(sign[33]),str(sign[34]),
                str(sign[35]),str(sign[36]),str(sign[37]),str(sign[38]),
                str(sign[39]),str(sign[40]), str(sign[41]),str(sign[42]),
                str(sign[43]), str(sign[44]), str(sign[45]),
                str(float(crime_list[crimeIndex][1]) / 12), str(float(crime_list[crimeIndex][2]) / 12), crime_list[crimeIndex][-1][0],
                crime_list[crimeIndex][-1][1]]

        temp_matrix.append(temp)

    scaled = scalar.fit_transform(temp_matrix)

    for row in scaled:
        writer.writerow(row)

    f.close()


def get_signLocation(sign):

    for location in sign_list:
        if sign[1] == location[8]:
            temp = re.sub("[typePoin':()crdas{} ]",'',location[5]).split(',')
            return [temp[1], temp[2]]


def set_monthlyCountMedian():


    arr = numpy.array(signMonthlyReadings_list)

    temp_list= []
    ids = []

    used = {}
    dates = ['2020', '2021', '2022']
    d = numpy.array(dates)

    for sign in sign_list:
        id = sign[8]
        for d in dates:
            ids.append((id, d))

    for i, x in enumerate(arr):
        arr[i][2] = datetime.datetime.strptime(x[2], '%Y-%m-%d').year


    for id, d in ids:
        temp_row = ['null', id, d]
        selected_rows = arr[np.where(arr[:,1] == id)[0]]

        df = pandas.DataFrame(selected_rows)
        i = 3
        while i < 44:

            column_median = df[i].median()
            temp_row.append(column_median)
            i= i + 1

        if not numpy.nan in temp_row:
            temp_list.append(temp_row)

    return temp_list



def cleanSigns():

    temp1 = []

    tempL = set_monthlyCountMedian()
    for sign in tempL:
        if datetime.datetime(day=1, month=1, year=2020) < datetime.datetime.strptime(sign[2], '%Y') < datetime.datetime(day=1, month=1, year=2023):

            temp = get_signLocation(sign)
            temp2 = sign
            temp2.append(float(temp[1]))
            temp2.append(float(temp[0]))
            temp1.append(temp2)

    return temp1



def get_list(lst, path):
    # loads requested file into a list for future use

    with open(path + ".csv") as csv_file:

        first = True
        rows = csv.reader(csv_file)
        for row in rows:
            if not first:
                lst = lst + [row]
            else:
                first = False
        return lst


set_maxFileLimit(maxInt)

sensor_list = get_list(sensor_list, sensor_path)
crime_list = get_list(crime_list, crime_path)
sign_list = get_list(sign_list, sign_path)
signMonthlyReadings_list = get_list(signMonthlyReadings_list, signMonthlyReadings_path)

sensor_list = cleanSensorData()
crime_list = cleanCrimeRate()
sign_list = cleanSigns()


create_translatorFile()
create_testingFile(sign_list)
create_trainingFile(sensor_list)



