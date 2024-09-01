import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_boxplots(df):
    
    # Create subplots with 1 row and `num_columns` columns
    fig, axs = plt.subplots(4,2, figsize=(10, 15))
    
    
    
    
    # Iterate over each column and create a box plot
    
    for i in range(7):
        axs[i//2,i%2].boxplot(df[parameter[i]])  
        axs[i//2,i%2].set_title(parameter[i])
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    ##plt.savefig("ps3_outliers.jpg")
    
    


def find_outliers(data):
 
  # Calculating the first quartile (Q1) and third quartile (Q3)
  Q1 = np.quantile(data, 0.25)
  Q3 = np.quantile(data, 0.75)

  # Calculating the interquartile range (IQR)
  IQR = Q3 - Q1 


  # Calculating the lower and upper bounds for outliers
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Finding outliers
  outliers = []
  for i,value in enumerate(data):
    if value < lower_bound or value > upper_bound:
      outliers.append(i)

  return outliers

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

def remove_outliers():
    new_df = df.copy()
    for j in range(7):
        for i in find_outliers(df[parameter[j]]):
            new_df.at[i,parameter[j]] = medians[j]
    new_df.to_csv("after_removing_outliers.csv",index=False)
    

df = pd.read_csv("after_linear_interopolation.csv")

parameter = ["temperature","humidity","pressure","rain","lightavg","lightmax","moisture"]
    

plot_boxplots(df)

medians = [0]*7
for i in range(7):
    medians[i] = median(df[parameter[i]])


##new = remove_outliers()
##new.to_csv("after_removing_outliers.csv")
remove_outliers()
after = pd.read_csv("after_removing_outliers.csv")

#print(find_outliers(df["rain"]))

plot_boxplots(after)



