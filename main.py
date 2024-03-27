''' a telecommunications provider has segmented his customer base by
service usage patterns, categorizing the customers into four groups.
If demographic data can be used to predict group membership, the company
can customize offers for individual perspective customers.
here is a model to be used to predict the class of a new or unknown case
using K-Nearest Neighbours '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Downloading the file
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'

import requests
def download(url, filename):
    response = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(response.content)

download(path, 'teleCust1000t.csv')

df = pd.read_csv('teleCust1000t.csv')

df['custcat'].value_counts()

df.hist(column='income', bins=50)

x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values

# Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

# Normalize Data

x_train_norm = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))

# Classification
# K nearest neighbor

from sklearn.neighbors import KNeighborsClassifier

# Training

k=4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors= k).fit(x_train_norm, y_train)
x_test_norm = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))

# Predicting
yhat = neigh.predict(x_test_norm)

# Accuracy Evaluation

from sklearn import metrics

print('Train set Accuracy: ', metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, yhat))

# What About Other K?

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train_norm, y_train)
    yhat = neigh.predict(x_test_norm)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plot the model accuracy for a different number of neighbors

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)