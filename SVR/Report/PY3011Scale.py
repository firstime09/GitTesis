## Scaling Data with StandardScaler
## Sumber -> https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
import pandas as pd
import numpy as np
import fungsi
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split, cross_val_score
# hitSVR = fungsi.allFunction

dFrame = pd.read_excel('C:/Users/user/Dropbox/FORESTS2020/00AllData/Data From Mas Sahid/FRCI 870_2611N.xlsx')
# print(dFrame, dFrame.shape)
features = dFrame.columns.drop(['FID', 'frci', 'Class', 'Band_9', 'Band_1'])
target = 'frci'
dFrameX = np.asarray(dFrame[features])
dFrameY = np.asarray(dFrame[target])
# print(dFrameY)

# clfSVR = hitSVR.F2020_SVR(dFrameX, dFrameY, 0.3, 0)
# print(clfSVR)
X_train, X_test, y_train, y_test = train_test_split(dFrameX, dFrameY, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfReg = RandomForestRegressor(n_estimators=30, random_state=0)
rfReg.fit(X_train, y_train)
y_pred = rfReg.predict(X_test)
print('Mean Absolute Error :', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error :', mean_squared_error(y_test, y_pred))
print('Root MSE :', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('Akurasi Skor :', rfReg.score(y_test, y_pred))
# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=0.3, random_state=0)
# Scores = clf.score(dFrameX, dFrameY)
# print(Scores)

# Normalisasi data X_train dan X_test
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
