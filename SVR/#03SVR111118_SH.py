import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(SVR/14112018D2.csv')
X = dataset.iloc[:, :6]
y = dataset.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for k in ['linear','poly','rbf','sigmoid']:
	classifier = svm.SVR(kernel=k, C=1, epsilon=0.1)
	classifier.fit(X_train, y_train)
	confidence = classifier.score(X_test, y_test)

	X_set, y_set = X_train, y_train
	# print(X_set)

	# X1, X2, X3, X4, X5, X6 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
										# np.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01),
										# np.arange(start=X_set[:,2].min() - 1, stop=X_set[:,2].max() + 1, step=0.01),
										# np.arange(start=X_set[:,3].min() - 1, stop=X_set[:,3].max() + 1, step=0.01),
										# np.arange(start=X_set[:,4].min() - 1, stop=X_set[:,4].max() + 1, step=0.01),
										# np.arange(start=X_set[:,5].min() - 1, stop=X_set[:,5].max() + 1, step=0.01))

	print(X.shape)