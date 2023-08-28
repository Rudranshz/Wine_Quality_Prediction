import os
os.system("cls")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from IPython.display import clear_output

wine_dataset = pd.read_csv(r'C:\winequality-red.csv')  
print(wine_dataset.shape)
print(wine_dataset.head())

print(wine_dataset.isnull().sum())
print(wine_dataset.describe())

sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.show()

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y= 'volatile acidity', data= wine_dataset)
plt.show()

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y= 'citric acid', data= wine_dataset)
plt.show()

correlation = wine_dataset.corr()
plot = plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt=' .1f', annot= True, annot_kws={'size':8}, cmap='Blues')
plt.show()

x = wine_dataset.drop('quality', axis=1)
print(x)

y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)
print(y.shape, y_train.shape, y_test.shape)

model = RandomForestClassifier()
model.fit(x_train, y_train)

x_train_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_train_prediction)
print('Acuracy:', test_data_accuracy)

input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')