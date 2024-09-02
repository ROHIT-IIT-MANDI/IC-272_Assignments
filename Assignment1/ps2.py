import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def isK(k,para):
    for i in df[df[para].isna()].index:
        if k==i:
            return True
    return False

def less(k,para):
    while isK(k,para):
        k-=1
    return k

def more(k,para):
    while isK(k,para):
        k+=1
    return k



def rmse(arr1, arr2):
    # Check if both arrays have the same length
    if len(arr1) != len(arr2):
        raise ValueError("Bhai kya kar rha h tu ?? Length hi same nhi h arrays ki")
    
    # Calculate the squared differences
    squared_diff = [(abs(x) - abs(y)) ** 2 for x, y in zip(arr1, arr2)]
    
    # Calculate the mean of the squared differences
    mean_squared_diff = sum(squared_diff) / len(squared_diff)
    
    # Calculate the square root of the mean squared difference
    rmse_value = mean_squared_diff ** 0.5
    
    return rmse_value


def mean2(values):
	total = 0
	for i in values:
		total+=i
	return round(total/len(values))

def mean(v1,v2):
	return round(((v1+v2)/2),2)


def median(arr):
    
    arr = sorted(arr)
    
    
    n = len(arr)
    if n % 2 == 1:
        # Return the middle element
        return arr[n // 2]
    else:
        # Return the average of the two middle elements
        mid1 = arr[n // 2 - 1]
        mid2 = arr[n // 2]
        return (mid1 + mid2) / 2



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
    m = mean2(values)
    for i in values:
        s+= ((i-m)**2)
    return round((s/len(values))**0.5,2)





df = pd.read_csv("landslide_data_miss.csv")

missing_indices = df[df["temperature"].isna()].index

print(missing_indices)

df.cleaned = df.dropna(subset=["stationid"])

df_cleaned = df.dropna(thresh=6)

##df_cleaned.to_csv("after_deleting3_values.csv",index = False)

df = pd.read_csv("landslide_data_miss.csv")

temp_data = df[df['temperature'].isna()].index
humidity_data = df[df['humidity'].isna()].index
pressure_data = df[df['pressure'].isna()].index
rain_data = df[df['rain'].isna()].index
lightavg_data = df[df['lightavg'].isna()].index
lightmax_data = df[df['lightmax'].isna()].index
moisture_data = df[df['moisture'].isna()].index
station_ids = df[df['stationid'].isna()].index

arr = [temp_data,humidity_data,pressure_data,rain_data,lightavg_data,lightmax_data,moisture_data]
parameter = ["temperature","humidity","pressure","rain","lightavg","lightmax","moisture"]

for i in range(len(arr)):
    for j in arr[i]:
        if j!=0 and j!=len(arr[i])-1:
            df.loc[j, parameter[i]] = mean(df[parameter[i]].iloc[less(j,parameter[i])],df[parameter[i]].iloc[more(j,parameter[i])])
        elif j==1:
            df.loc[j, parameter[i]] = mean(df[parameter[i]].iloc[more(j,parameter[i])],df[parameter[i]].iloc[more(more(j,parameter[i]),parameter[i])])
        else:
            df.loc[j, parameter[i]] = mean(df[parameter[i]].iloc[less(j,parameter)],df[parameter[i]].iloc[less(less(j,parameter),parameter)])

df.to_csv("after_linear_interopolation.csv",index = False)


means = [0]*7
medians = [0]*7
stds = [0]*7

for i in range(7):
    means[i] = mean2(df[parameter[i]])
    medians[i] = median(df[parameter[i]])
    stds[i] = STD(df[parameter[i]])

print("Means are: ",means)
print("Medians are: ",medians)
print("Standard Deviations are: ",stds)

original = pd.read_csv("landslide_data_original.csv")

error = [0]*7

for i in range(7):
    error[i] = rmse(original[parameter[i]],df[parameter[i]])

print("ERROR IS : ", error)

plt.bar(parameter,error)
plt.xlabel("Attributes")
plt.ylabel("RMSE")
plt.show()


