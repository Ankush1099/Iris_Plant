# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:34:39 2020

@author: Ankush
"""

#Step 0 : Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Import dataset
iris_dataset = pd.read_csv('Iris.csv')
iris_dataset.head(5)
iris_dataset.tail(5)

#Step 2: Visualize the dataset
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data = iris_dataset)
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data = iris_dataset)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='PetalLengthCm', hue='Species', data = iris_dataset)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='PetalWidthCm', hue='Species', data = iris_dataset)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='SepalLengthCm', hue='Species', data = iris_dataset)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='SepalWidthCm', hue='Species', data = iris_dataset)

sns.pairplot(iris_dataset, hue = 'Species')

sns.heatmap(iris_dataset.corr(), annot = True)


#Step 3: Data Cleaning
X = iris_dataset.drop(['Species'], axis=1)
X
y = iris_dataset['Species']
y
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)

#Step 4: Training the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(X_train,y_train)

#Step 5: Evaluation
y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict))
    



