import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split, cross_val_score

dataSets = pd.read_csv('D:/00AllData/21112018N1.csv')
# print(dataSets)

X = dataSets.iloc[:, 0:6].values
y = dataSets.iloc[:, 6].values
# print(X)


# data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# data_scaled = data_scaler.fit_transform(X)
for i in range(1):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

	# Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	# print(X_test)

	clf_SVR_RBF = SVR(kernel='rbf', C=10, epsilon=0.1)
	clf_SVR_RBF.fit(X_train, y_train)
	scores = clf_SVR_RBF.score(X_test, y_test)
	print(scores)

	y_pred = clf_SVR_RBF.predict(X_test)
	mse1 = mean_squared_error(y_test, y_pred)
	rmse = math.sqrt(mse1)
	print(rmse)
	# y_aktual = y_test
	# cofMatrix = confusion_matrix(X_test, y_pred)
	# print(cofMatrix)


# clf_randomForest = ExtraTreesClassifier(n_estimators=500, random_state=5)
# clf_randomForest.fit(X_train, y_train)
# scores = cross_val_score(clf_randomForest, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
# print(scores)
# print("Random_Values -> cv accuracy: mean = %0.3f std = %0.3f" % (np.mean(scores), np.std(scores))