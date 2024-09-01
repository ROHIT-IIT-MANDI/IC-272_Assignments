import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_miss = pd.read_csv('landslide_data_miss.csv')
data_orig = pd.read_csv('landslide_data_original.csv')

# Dropping Rows with Missing 'stationid'
data1 = data_miss.dropna(subset=['stationid'])

# Dropping Rows with More than One-Third of Attributes Missing
threshold = data1.shape[1] * (2/3)
data2 = data1.dropna(thresh=threshold)
data_with_outliers = data2

columns = np.array(data2.columns)

# Function for Linear Interpolation
def linear_interpolation(data):
    index = []
    for i in range(len(data)):
        if np.isnan(data[i]):
            prev_index = i - 1
            while prev_index >= 0 and np.isnan(data[prev_index]):
                prev_index -= 1

            next_index = i + 1
            while next_index < len(data) and np.isnan(data[next_index]):
                next_index += 1

            if prev_index >= 0 and next_index < len(data):
                data[i] = (data[prev_index] + data[next_index]) / 2
                index.append(i)
    
    return data, index

# Apply Linear Interpolation
listed = []
for i in columns[2:]:
    data = data2[i].values
    filled_data, index_list = linear_interpolation(data)
    listed.append((i, index_list))
    
columns_for_outliers = columns[2:]

# Outlier Bounds Calculation
bounds = []
for i in range(len(columns_for_outliers)):
    data = data_with_outliers[columns_for_outliers[i]].values
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    bounds.append((columns_for_outliers[i], lower_bound, upper_bound))

# Removing Outliers by Replacing with Bounds
data_without_outliers = data_with_outliers.copy()
for i in columns_for_outliers:
    data = data_with_outliers[i].values
    lower_bound = bounds[columns_for_outliers.tolist().index(i)][1]
    upper_bound = bounds[columns_for_outliers.tolist().index(i)][2]
    for j in range(len(data)):
        if data[j] < lower_bound:
            data[j] = lower_bound
        elif data[j] > upper_bound:
            data[j] = upper_bound

# Min-Max Normalization
list_of_min_max = []
for i in columns_for_outliers:
    min_val = min(data_without_outliers[i].values)
    max_val = max(data_without_outliers[i].values)
    list_of_min_max.append((i, min_val, max_val))

data_normalized = data_without_outliers.copy()
for i in columns_for_outliers:
    data = data_normalized[i].values
    min_val = list_of_min_max[columns_for_outliers.tolist().index(i)][1]
    max_val = list_of_min_max[columns_for_outliers.tolist().index(i)][2]
    for j in range(len(data)):
        data[j] = ((data[j] - min_val) / (max_val - min_val)) * 7 + 5

# Calculate and Print Min and Max After Normalization
list_of_min_max_after_normalization = []
for i in columns_for_outliers:
    min_val = min(data_normalized[i].values)
    max_val = max(data_normalized[i].values)
    print(f'After Min-Max Normalization for {i}: min = {min_val}, max = {max_val}')
    list_of_min_max_after_normalization.append((i, min_val, max_val))

# Mean and Standard Deviation Before Standardization
list_of_means_before = []
list_of_std_devs_before = []
for i in columns_for_outliers:
    data = data_without_outliers[i].values
    mean_val = np.mean(data)
    std_dev_val = np.std(data)
    list_of_means_before.append(mean_val)
    list_of_std_devs_before.append(std_dev_val)
    print(f'Before Standardization for {i}: mean = {mean_val}, std dev = {std_dev_val}')

# Standardization
data_standardized = data_without_outliers.copy()
for i in columns_for_outliers:
    data = data_standardized[i].values
    mean_val = list_of_means_before[columns_for_outliers.tolist().index(i)]
    std_dev_val = list_of_std_devs_before[columns_for_outliers.tolist().index(i)]
    for j in range(len(data)):
        data[j] = (data[j] - mean_val) / std_dev_val

# Mean and Standard Deviation After Standardization
list_of_means_after = []
list_of_std_devs_after = []
for i in columns_for_outliers:
    data = data_standardized[i].values
    mean_val = np.mean(data)
    std_dev_val = np.std(data)
    list_of_means_after.append(mean_val)
    list_of_std_devs_after.append(std_dev_val)
    print(f'After Standardization for {i}: mean = {mean_val}, std dev = {std_dev_val}')

# Comparing Mean and Standard Deviation Before and After Standardization
for i in range(len(columns_for_outliers)):
    print(f'Comparison for {columns_for_outliers[i]}:')
    print(f'   Mean Before: {list_of_means_before[i]}, After: {list_of_means_after[i]}')
    print(f'   Std Dev Before: {list_of_std_devs_before[i]}, After: {list_of_std_devs_after[i]}')