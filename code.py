# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('3MINDIA.NS.csv')
dataset['Date'] = dataset.index


#X = dataset.iloc[:, :-2].values
#y = dataset.iloc[:, 6].values

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#first 20% test set , rest 80% training set
valid = dataset[:247]
train = dataset[247:]

#first 20% train , 20% test set , rest 60% training set
#valid = dataset[247:494]
#train1 = dataset[:247]
#train2 = dataset[494:]
#frames = [train1, train2]
#train = pd.concat(frames)

#first 40% train , 20% test set , rest 40% training set
#valid = dataset[494:741]
#train1 = dataset[:494]
#train2 = dataset[741:]
#frames = [train1, train2]
#train = pd.concat(frames)

#first 60% train , 20% test set , rest 20% training set
#valid = dataset[741:988]
#train1 = dataset[:741]
#train2 = dataset[998:]
#frames = [train1, train2]
#train = pd.concat(frames)

#first 80% train set , rest 20% test set
#valid = dataset[990:]
#train = dataset[:990]

X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
#X_train = X_train.drop('100MA', axis=1)
#X_valid = X_valid.drop('100MA', axis=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_valid)

df1 = pd.DataFrame({'Actual': y_valid, 'Predicted': y_pred} )  
df1['Percentage'] =(df1['Predicted'])/( df1['Actual'] )*100
df1['Percentage'] = np.where(df1['Percentage']>100, 200 - df1['Percentage'], df1['Percentage'])
df1


plt.plot(dataset['Close'])
plt.plot(df1[['Actual', 'Predicted']])

df1.to_csv('train80_test20.csv', sep=',')


________--
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#dataset = pd.read_csv('AMZN.csv')
dataset = pd.read_csv('3MINDIA.NS.csv')

dataset['Date'] = dataset.index


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


train = dataset[:987]
valid = dataset[987:]

X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_valid_scaled = scaler.fit_transform(X_valid)
X_valid = pd.DataFrame(X_valid_scaled)

params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

model.fit(X_train,y_train)
preds = model.predict(X_valid)

valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])

df1 = pd.DataFrame({'Actual': y_valid, 'Predicted':preds} )  
df1['Percentage'] =(df1['Predicted'])/( df1['Actual'] )*100
df1['Percentage'] = np.where(df1['Percentage']>100, 200 - df1['Percentage'], df1['Percentage'])
df1

df1.to_csv('train80_test20.csv', sep=',')

_________________
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg 
import time


#dataset = pd.read_csv('AMZN.csv')
dataset = pd.read_csv('3MINDIA.NS.csv')

dataset['Date'] = dataset.index


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


train = dataset[:987]
valid = dataset[987:]

X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
type(X_valid)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
#y_train = sc.fit_transform(y_train)
model = GaussianNB()

#Predict Output 
#predicted= model.predict(y_valid,X_train)
#print predicted

from sklearn.naive_bayes import GaussianNB
##classifier.fit(X_train, y_train)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
#y_pred = classifier.predict(X_valid)
X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
type(X_valid)
from sklearn.metrics import confusion_matrix
valid['Predictions'] = 0
valid['Predictions'] = 0


df1 = pd.DataFrame({'Actual': y_valid, 'Predicted':0} )  
df1['Percentage'] =(df1['Predicted'])/( df1['Actual'] )*100
df1['Percentage'] = np.where(df1['Percentage']>100, 200 - df1['Percentage'], df1['Percentage'])
df1

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)





from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)


GNB = GaussianNB()
GNB.fit(X_train,y_train)
#Predict the y test
y_pred=GNB.predict(X_test)
#Print the accuracy score of our predicted y using metrics from sklearn
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_pred))

img = mpimg.imread('n.jpg') 
plt.imshow(img) 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_valid)
_________

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from err import error_count, calc_diff
from visual import plot


dataset = pd.read_csv('3MINDIA.NS.csv')

dataset['Date'] = dataset.index


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


train = dataset[:987]
valid = dataset[987:]

X_train = train.drop('Close', axis=1)
y_train = train['Close']
X_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]


regressor = Sequential()

regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
regressor.add(Dropout(.2))

regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))


regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)



regressor.save('1.h5')

del regressor

regressor = load_model('1.h5')

real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)

dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 


diff_rate = calc_diff(predicted_stock_price[:, 0], predicted_stock_price[:, 1])

inputs = np.array(X)

all_real_price = np.array(y)
all_predicted_price = regressor.predict(inputs)

dataset_pred_real = pd.DataFrame()
dataset_pred_real['real'] = all_real_price
dataset_pred_real['predicted'] = all_predicted_price

all_prices = scaler.inverse_transform(dataset_pred_real)  

plot(predicted=all_prices[:, 0])
plot(real=all_prices[:, 1])
plot(predicted=all_prices[:, 0], real=all_prices[:, 1])