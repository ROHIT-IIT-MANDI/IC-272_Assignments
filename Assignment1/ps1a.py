import pandas as pd
import numpy as np


def mean(values):
	total = 0
	for i in values:
		total+=i
	return round(total/len(values))

def maximum(values):
    largest = 0
    for i in values:
        if largest<i:
            largest = i
    return round(largest,2)

def minimum(values):
    smallest = 0
    for i in values:
        if smallest>i:
            smallest = i
    return round(smallest,2)

def STD(values):
    s = 0
    m = mean(values)
    for i in values:
        s+= ((i-m)**2)
    return round(s/len(values),2)


def pearson(values1,values2):
    m1 = mean(values1)
    m2 = mean(values2)
    std1 = STD(values1)
    std2 = STD(values2)
    
    numerator = np.sum((values1 - m1) * (values2 - m2))
    denominator = np.sqrt(np.sum((values1 - m1) ** 2)) * np.sqrt(np.sum((values2 - m2) ** 2))
    
    return round(numerator/denominator,2)



df = pd.read_csv('landslide_data_original.csv') ## df is a data frame object type
temp_data = df['temperature']
humidity_data = df['humidity']
pressure_data = df['pressure']
rain_data = df['rain']
lightavg_data = df['lightavg']
lightmax_data = df['lightmax']
moisture_data = df['moisture']
station_ids = df['stationid']

arr = [temp_data,humidity_data,pressure_data,rain_data,lightavg_data,lightmax_data,moisture_data]

print("Mean of the temperature is : ",mean(temp_data))
print("Max of the temperature is : ",maximum(temp_data))
print("Min of the temperature is : ",minimum(temp_data))
print("STD of the temperature is : ",STD(temp_data))

for i in arr:
    for j in arr:
        print(pearson(i,j),end=" | ")
    print()
    print("_")
    print()
        