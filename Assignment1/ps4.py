import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import math

# Load the dataset
miss_data = pd.read_csv('landslide_data_miss.csv')
orig_data = pd.read_csv('landslide_data_original.csv')

df = pd.read_csv("after_removing_outliers.csv")

parameter = ["temperature","humidity","pressure","rain","lightavg","lightmax","moisture"]


def STD(values):
    s = 0
    m = statistics.mean(values)
    for i in values:
        s+= ((i-m)**2)
    return round((s/len(values))**0.5,2)


temp_data =  []
humidity_data = []
pressure_data = []
rain_data = []
lightavg_data = []
lightmax_data = []
moisture_data = []
station_ids =  []


arr = [temp_data,humidity_data,pressure_data,rain_data,lightavg_data,lightmax_data,moisture_data]

max_before_normalisation = []
min_before_normalisation = []

max_after_normalisation = []
min_after_normalisation = []


def normalisation(values,i):
    minimum = min(values)
    maximum = max(values)
    ##diff = maximum - minimum
    max_before_normalisation
    for j in values:
        r = (j-minimum)/(maximum-minimum)
        arr[i].append((r*7)+5)

for i in range(7):
    max_before_normalisation.append(round(max(df[parameter[i]]),2))
    min_before_normalisation.append(round(min(df[parameter[i]]),2))
    
    normalisation(df[parameter[i]],i)
    max_after_normalisation.append(round(max(arr[i]),2))
    min_after_normalisation.append(round(min(arr[i]),2))



temp_data =  []
humidity_data = []
pressure_data = []
rain_data = []
lightavg_data = []
lightmax_data = []
moisture_data = []
station_ids =  []


arr = [temp_data,humidity_data,pressure_data,rain_data,lightavg_data,lightmax_data,moisture_data]

def standardization(values,i):
    means = values.mean()
    for j in values:
        arr[i].append((j-means)/STD(values))
        
means_before_normalisation = []
stds_before_normalisation = []
means_after_normalisation = []
stds_after_normalisation = []

for i in range(7):
    means_before_normalisation.append(round(df[parameter[i]].mean(),2))
    stds_before_normalisation.append(round(STD(df[parameter[i]]),2))
    
    normalisation(df[parameter[i]],i)
    means_after_normalisation.append(round(statistics.mean(arr[i]),2))
    stds_after_normalisation.append(round(STD(arr[i]),2))


print(max_before_normalisation)
print(min_before_normalisation)

print(max_after_normalisation)
print(min_after_normalisation)

print(means_before_normalisation)
print(stds_before_normalisation)
print(means_after_normalisation)
print(stds_after_normalisation)
