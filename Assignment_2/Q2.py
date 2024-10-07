import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("Reduced_dim.csv")

columns = data.columns[:-1]
X = data[columns]
y = data['Species']

#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=104, shuffle=True)

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()

def euclid_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(test_point, k=5):
    distances = []
    for i in range(len(X_train_np)):
        dist = euclid_distance(test_point, X_train_np[i])
        distances.append((dist, y_train_np[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]#taking initial 5 sorted values
    k_labels = list(map(lambda x: x[1], k_nearest))#looking which class label is nearest

    #counting the no. o max occurances
    counts = {}
    for label in k_labels:
        counts[label] = counts.get(label, 0) + 1
    majority_vote = max(counts, key=counts.get)
    return majority_vote

#applying to Xtest
predictions = [knn_predict(test) for test in X_test_np]
actual_predicted = pd.DataFrame({'Actual': y_test.values,'Predicted': predictions})
print(actual_predicted)
print()
conf_matrix = metrics.confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(conf_matrix)

labels = np.unique(y)
conf_matrix_plot = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
conf_matrix_plot.plot(cmap=plt.cm.Blues)
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

results = pd.DataFrame({'Actual': y_test.values,'Predicted': predictions})
results.to_csv('actual_vs_predicted.csv', index=False)
