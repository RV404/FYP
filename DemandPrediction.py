import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
import requests
from io import BytesIO
from keras.models import Model
from keras import layers
from keras.layers import Input, LSTM, Dense, Bidirectional, Layer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K


#load the demand forecasting model
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        u_it = K.tanh(K.dot(inputs, self.W) + self.b)
        ait = K.dot(u_it, K.expand_dims(self.u))
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        # Compute the context vector
        weighted_input = inputs * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
ConsumptionModel = load_model('DemandForecastV2.h5',custom_objects={'Attention': Attention})

#load the scaler
with open('DemandScaler.pkl', 'rb') as file:
    demandScaler = pickle.load(file)

#get all demand data
demandUrl1 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand1.xlsx'
demandUrl2 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand2.xlsx'
demandUrl3 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand3.xlsx'
demandUrl4 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand4.xlsx'
demandUrl5 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand5.xlsx'
demandUrl6 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand6.xlsx'
demandUrl7 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand7.xlsx'
demandUrl8 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand8.xlsx'
demandUrl9 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand9.xlsx'
demandUrl10 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand10.xlsx'
demandUrl11 = 'https://github.com/ShalikaNuwan/prediction/raw/main/demand11.xlsx'

demandUrls = [demandUrl1,demandUrl2,demandUrl3,demandUrl4,demandUrl5,demandUrl6,demandUrl7,demandUrl8,demandUrl9,demandUrl10,demandUrl11]

def create_input(df,scaler):
    df['Consumption'] = scaler.transform(df[['Consumption']])
    consumption = df['Consumption'].values
    x = [consumption]
    return np.array(x)

demandArr = []
for index,url in enumerate(demandUrls):
    response = requests.get(url)
    file_content = BytesIO(response.content)
    dataset = pd.read_excel(file_content)

    x = create_input(dataset,demandScaler)
    x = x.reshape(1,10,1)
    prediction = ConsumptionModel.predict(x).flatten()
    prediction_non_scaled = demandScaler.inverse_transform([prediction]).flatten()
    # temp_dict = {
    #     'locatoion_name' : f'location{index+1}',
    #     'consummption(Mwh)' : float(round(prediction_non_scaled[0]/1000,3))
    # }
    demandArr.append(float(round(prediction_non_scaled[0]/1000,3)))


def getDemandForecast():
    return demandArr





