import requests
from keras.models import load_model
import pickle
import pandas as pd
from io import BytesIO
import numpy as np

#load solar prediction model
solarModel = load_model('solarForecast.h5')
#load wind prediction model
windModel = load_model('windForecast.h5')

#load solar scaler
with open('SolarScaler.pkl', 'rb') as file:
    solarScaler = pickle.load(file)

#load wind scaler
with open('WindScaler.pkl', 'rb') as file:
    wind_scaler = pickle.load(file)

#solar dataset url
solar1url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Solar1.xlsx'
solar2url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Solar2.xlsx'
solar3url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Solar3.xlsx'
solar4url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Solar4.xlsx'
solar5url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Solar5.xlsx'

#wind dataste url
wind1url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind1.xlsx'
wind2url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind2.xlsx'
wind3url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind3.xlsx'
wind4url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind4.xlsx'
wind5url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind5.xlsx'
wind6url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind6.xlsx'
wind7url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind7.xlsx'
wind8url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind8.xlsx'
wind9url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind9.xlsx'
wind10url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind10.xlsx'
wind11url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind11.xlsx'
wind12url = 'https://github.com/ShalikaNuwan/prediction/raw/main/Wind12.xlsx'

def create_solar_input(df,scaler):
    df['Generation'] = scaler.transform(df[['Generation']])
    solarIrradiance = df['SolarIrradiance'].values
    temperature = df['Temperature'].values
    generation = df['Generation'].values
    x = [generation,temperature,solarIrradiance]
    return np.array(x)

def create_wind_input(df,scaler):
    df['Generation'] = scaler.transform(df[['Generation']])
    windspeed = df['Windspeed'].values
    generation = df['Generation'].values
    x = [generation,windspeed]
    return np.array(x)

solarUrls = [solar1url,solar2url,solar3url,solar4url,solar5url]
windUrls = [wind1url,wind2url,wind3url,wind4url,wind5url,wind6url,wind7url,wind8url,wind9url,wind10url,wind11url,wind12url]


solarArr = []
for index,url in enumerate(solarUrls):
    response = requests.get(url)
    file_content = BytesIO(response.content)
    dataset = pd.read_excel(file_content)

    x = create_solar_input(dataset,solarScaler)
    x = x.reshape(1,3,10)
    prediction = solarModel.predict(x).flatten()
    prediction_non_scaled = solarScaler.inverse_transform([prediction]).flatten()
    # temp_dict = {
    #     'plant_name' : f'solar{index+1}',
    #     'Generaton(Mw)' : float(round(prediction_non_scaled[0]/1000,3))
    # }
    solarArr.append(float(round(prediction_non_scaled[0]/1000,3)))

windArr = []

for index,url in enumerate(windUrls):
    response = requests.get(url)
    file_content = BytesIO(response.content)
    dataset = pd.read_excel(file_content)

    x = create_wind_input(dataset,wind_scaler)
    x = x.reshape(1,2,10)
    prediction = windModel.predict(x).flatten()
    prediction_non_scaled = wind_scaler.inverse_transform([prediction]).flatten()
    # temp_dict = {
    #     'plant_name' : f'wind{index+1}',
    #     'Generaton(Mw)' : float(round(prediction_non_scaled[0]/1000,3))
    # }
    windArr.append(float(round(prediction_non_scaled[0]/1000,3)))


def getSolarGeneration():
    return solarArr

def getWindGeneration():
    return windArr










