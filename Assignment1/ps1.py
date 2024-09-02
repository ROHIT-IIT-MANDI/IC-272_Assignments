import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


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
    return round((s/len(values))**0.5,2)


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
parameter = ["Temperature","Humidity","Pressure","Rain","Lightavg","Lightmax","Moisture"]

print("Mean of the temperature is : ",mean(temp_data))
print("Max of the temperature is : ",maximum(temp_data))
print("Min of the temperature is : ",minimum(temp_data))
print("STD of the temperature is : ",STD(temp_data))

print()

k = 0
l = 0
for i in arr:
    print(parameter[k],end=" ")
    for j in arr:
        
        print(pearson(i,j),end=" | ")
    k+=1
    print()
    print("_")
    print()

x = []
k = 0 
for i in station_ids:
    if i=="t12":
        x.append(humidity_data[k])
    k+=1

df=df[df['stationid']=='t12']['humidity'].to_numpy()
bin_size = 5
min_value = math.floor(minimum(df))
max_value = math.ceil(maximum(df))
bins = list(range(min_value, max_value + bin_size, bin_size))
histogram = {bin_value: 0 for bin_value in bins}

for i in df:
    for j in bins:
        if j<=i < j+bin_size:
            histogram[j]+=1
            break

bin_edges = list(histogram.keys())
counts = list(histogram.values())

plt.figure(figsize=(10,6))
plt.bar(bin_edges, counts, width=bin_size, edgecolor="black", align = "edge", color="red")
plt.title("Jumidity for stationid t12")
plt.xlabel("Humidity")
plt.ylabel("Frequency")
plt.xticks(bin_edges)
plt.show()


        