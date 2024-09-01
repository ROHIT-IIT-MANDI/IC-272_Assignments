import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
miss_data = pd.read_csv('landslide_data_miss.csv')
orig_data = pd.read_csv('landslide_data_original.csv')

# Dropping rows where 'stationid' is missing
filtered_data1 = miss_data.dropna(subset=['stationid'])

# Dropping rows with more than a third of the attributes missing
missing_threshold = filtered_data1.shape[1] * (2/3)
filtered_data2 = filtered_data1.dropna(thresh=missing_threshold)
data_with_extremes = filtered_data2

cols_array = np.array(filtered_data2.columns)

# Function for performing linear interpolation
def interp_linear(vals):
    altered_indices = []
    for idx in range(len(vals)):
        if np.isnan(vals[idx]):
            prior_idx = idx - 1
            while prior_idx >= 0 and np.isnan(vals[prior_idx]):
                prior_idx -= 1

            next_idx = idx + 1
            while next_idx < len(vals) and np.isnan(vals[next_idx]):
                next_idx += 1

            if prior_idx >= 0 and next_idx < len(vals):
                vals[idx] = (vals[prior_idx] + vals[next_idx]) / 2
                altered_indices.append(idx)
    
    return vals, altered_indices

# Applying the linear interpolation
updated_list = []
for col in cols_array[2:]:
    col_data = filtered_data2[col].values
    updated_vals, affected_indices = interp_linear(col_data)
    updated_list.append((col, affected_indices))
    
cols_to_check = cols_array[2:]

# Calculating bounds for detecting outliers
outlier_limits = []
for idx in range(len(cols_to_check)):
    col_data = data_with_extremes[cols_to_check[idx]].values
    first_quartile = np.percentile(col_data, 25)
    third_quartile = np.percentile(col_data, 75)
    inter_quartile_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * inter_quartile_range
    upper_limit = third_quartile + 1.5 * inter_quartile_range
    outlier_limits.append((cols_to_check[idx], lower_limit, upper_limit))

# Replacing outliers with calculated bounds
data_without_extremes = data_with_extremes.copy()
for col in cols_to_check:
    col_data = data_with_extremes[col].values
    low_limit = outlier_limits[cols_to_check.tolist().index(col)][1]
    high_limit = outlier_limits[cols_to_check.tolist().index(col)][2]
    for j in range(len(col_data)):
        if col_data[j] < low_limit:
            col_data[j] = low_limit
        elif col_data[j] > high_limit:
            col_data[j] = high_limit

# Min-Max Normalization process
min_max_vals = []
for col in cols_to_check:
    min_value = min(data_without_extremes[col].values)
    max_value = max(data_without_extremes[col].values)
    min_max_vals.append((col, min_value, max_value))

normalized_data = data_without_extremes.copy()
for col in cols_to_check:
    col_data = normalized_data[col].values
    min_value = min_max_vals[cols_to_check.tolist().index(col)][1]
    max_value = min_max_vals[cols_to_check.tolist().index(col)][2]
    for j in range(len(col_data)):
        col_data[j] = ((col_data[j] - min_value) / (max_value - min_value)) * 7 + 5

# Printing Min and Max After Normalization
normalized_min_max = []
for col in cols_to_check:
    min_value = min(normalized_data[col].values)
    max_value = max(normalized_data[col].values)
    print(f'Post-Normalization for {col}: minimum = {min_value}, maximum = {max_value}')
    normalized_min_max.append((col, min_value, max_value))

# Mean and Standard Deviation before standardization
means_before_std = []
std_devs_before_std = []
for col in cols_to_check:
    col_data = data_without_extremes[col].values
    mean_value = np.mean(col_data)
    std_dev_value = np.std(col_data)
    means_before_std.append(mean_value)
    std_devs_before_std.append(std_dev_value)
    print(f'Pre-Standardization for {col}: mean = {mean_value}, std dev = {std_dev_value}')

# Standardization process
standardized_data = data_without_extremes.copy()
for col in cols_to_check:
    col_data = standardized_data[col].values
    mean_value = means_before_std[cols_to_check.tolist().index(col)]
    std_dev_value = std_devs_before_std[cols_to_check.tolist().index(col)]
    for j in range(len(col_data)):
        col_data[j] = (col_data[j] - mean_value) / std_dev_value

# Mean and Standard Deviation after standardization
means_after_std = []
std_devs_after_std = []
for col in cols_to_check:
    col_data = standardized_data[col].values
    mean_value = np.mean(col_data)
    std_dev_value = np.std(col_data)
    means_after_std.append(mean_value)
    std_devs_after_std.append(std_dev_value)
    print(f'Post-Standardization for {col}: mean = {mean_value}, std dev = {std_dev_value}')

# Comparing mean and standard deviation before and after standardization
for idx in range(len(cols_to_check)):
    print(f'Comparison for {cols_to_check[idx]}:')
    print(f'   Mean Before: {means_before_std[idx]}, Mean After: {means_after_std[idx]}')
    print(f'   Std Dev Before: {std_devs_before_std[idx]}, Std Dev After: {std_devs_after_std[idx]}')
