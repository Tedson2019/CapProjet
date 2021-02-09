# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 02:21:46 2021

@author: samba
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:31:13 2020

@author: samba
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
import seaborn as sns
import pandas as pd
from pandas import to_numeric
import re 


df = pd.read_csv("temperature.csv",header = None)
df = pd.read_csv("temperature.csv" )
df['temperature'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[0:5]))
df['temperature_target'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[0:5]))
df['date'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[6:16]))
df['time'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[17:19]))
df['conectiondeviced'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[30:38]))
df = df.drop('temperature;EnqueuedTime;ConnectionDeviceId', axis = 1)
df = df.drop('date',axis = 1)
times = np.array(df['time'])
temp = np.array(df['temperature'])

###On s'assure que toutes les variables  numériques sont des nombres et celles catégorielles sont des chaines de caractère 
df[["temperature","time","temperature_target"]] = df[["temperature","time","temperature_target"]].apply(pd.to_numeric)
df[["conectiondeviced"]] = df[["conectiondeviced"]].astype(object)

## étiquettage des températures en fonctions des heures d'entrées  
def target(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe["target"] = np.array(dataframe.index , dtype = int)
    means = []
    for i in range(24):
        indexes = dataframe["time"] == i
        mean = dataframe["temperature_target"][indexes].mean()
        std = np.std(dataframe["temperature_target"][indexes])
        #(On affecte la moyenne des mesures de temperature  par heure )
        dataframe["temperature_target"][indexes] = mean 
        for index in np.array(df.index)[indexes]:
            dataframe["target"][index] = "Normale" if abs(dataframe["temperature"][index] - mean) <= 2.20 else "Anormale" 
    dataframe = dataframe.drop("temperature_target" , axis = 1)
    return dataframe.to_csv (r'C:\Users\samba\OneDrive\Bureau\PFE_Réseau_Neuronne\new_dataset.csv', index = False, header = True)
        

target(df)