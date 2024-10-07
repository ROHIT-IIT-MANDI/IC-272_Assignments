import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

original_data = pd.read_csv("Iris.csv")

attributes = original_data.columns[:-1]  # Dropped species name

# Removing outliers
for i in range(len(attributes)):
    col_data = original_data[attributes[i]]
    
    per_25 = np.percentile(col_data, 25)
    per_75 = np.percentile(col_data, 75)
    iqr = per_75 - per_25
    lower = per_25 - 1.5 * iqr
    upper = per_75 + 1.5 * iqr
    
    median = col_data.median()
    
    for j in range(len(original_data[attributes[i]])):
        value = original_data.loc[j, attributes[i]]
        if value < lower or value > upper:
            original_data.loc[j, attributes[i]] = median

data_1 = original_data.copy()
corrected_data = original_data.copy()

# Mean subtracted data
for i in range(len(attributes)):
    mean = np.mean(corrected_data[attributes[i]])
    x_xi = corrected_data[attributes[i]] - mean
    corrected_data[attributes[i]] = x_xi

mean_sub_data = corrected_data.copy()
mean_sub_data_arr = mean_sub_data[attributes].values

# Covariance matrix
cov = np.dot(mean_sub_data_arr.T,mean_sub_data_arr)/150

# Eigen analysis
eigenvalues, eigenvectors = np.linalg.eig(cov)
values = eigenvalues.argsort()[::-1] #will return indices to be used in eigenvectors
eigenvalues = eigenvalues[values]
eigenvectors = eigenvectors[:, values]
eigenvector = eigenvectors[:, :2]

# Projecting the data on directions of first two eigen vectors
corrected_data = corrected_data.drop(columns=original_data.columns[-1])
data = corrected_data.values
dim_reduced = np.dot(data, eigenvector)
dim_reduced = pd.DataFrame(dim_reduced, columns=[0,1])
print(dim_reduced)

# Reconstructing the data
data = pd.DataFrame(np.dot(np.array(dim_reduced), eigenvector.T), columns=attributes)
data1=data.copy()

print(data)

rmse_values = []
for i in range(len(attributes)):
    sum_sq_errors = sum((data1[attributes[i]] - corrected_data[attributes[i]]) ** 2)
    rmse = math.sqrt(sum_sq_errors / len(data[attributes[i]]))
    rmse_values.append(rmse)
print("Rmse values are: ",rmse_values)

dim_reduced[2] = original_data['Species']
dim_reduce = dim_reduced.rename(columns={0: 'PC1', 1: 'PC2', 2: 'Species'})
dim_reduce.to_csv("Reduced_dim.csv", index=False)

print(dim_reduced)
# Plotting
plt.figure(figsize=(8, 6))
origin = np.zeros(2)

print(eigenvalues)
print("Eigen vecotr is: ",eigenvector)

eigenvectors = eigenvector.T
scale_factor = 3
for eigenvector in eigenvectors:
    plt.quiver(*origin, eigenvector[0], eigenvector[1], 
               angles='xy', scale_units='xy', scale=1, color='k')
plt.gca().set_aspect('equal', adjustable='box')
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=dim_reduce, palette='Set1')
plt.title('2D Reduced Iris Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
