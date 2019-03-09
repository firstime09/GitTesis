from __future__ import print_function, division
from sklearn.svm import SVR
from osgeo import gdal, gdal_array
from Modul_ML.F17122018ML import F2020ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gdal.UseExceptions()
gdal.AllRegister()
path1 = 'D:/00PyCode/00AllData/Test_Data_08032019/'
img_ds = gdal.Open(path1 + 'layerstack.TIF', gdal.GA_ReadOnly)
# roi_ds = gdal.Open('D:/00PyCode/00AllData/Test_Data_08032019/training/training.TIF', gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
# print(img)
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
# roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# plt.subplot(121)
plt.imshow(img[:, :, 4], cmap = plt.cm.Greys_r)
plt.title('DATA LandSat')

path2 = 'C:/Users/filkom2016/Dropbox/FORESTS2020/00AllData/Dataframe Cidanau/'
loadFile = pd.read_excel(path2 + 'Cidanau_350_00.xlsx')
select_col = ['Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
select_row = 'frci'

dfx = pd.DataFrame(loadFile, columns=select_col)
dfy = np.asarray(loadFile[select_row])

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=5)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

best_score = 0
for gamma in [0.01, 0.1, 1]:
    for C in [1, 2, 5, 8, 10]:
        for epsilon in [0.001, 0.01, 0.1, 1]:
            clfSVR = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            clfSVR.fit(X_train, y_train)
            score = clfSVR.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_C = C
                best_gamma = gamma
                best_epsilon = epsilon
                best_parm = {'C':best_C, 'Gamma':best_gamma, 'Epsilon':best_epsilon}

clfSVR1 = SVR(kernel='rbf', C=best_C, epsilon=best_epsilon, gamma=best_gamma)
clfSVR1.fit(X_train, y_train)
clfSVR.score(X_test, y_test)
y_pred = clfSVR1.predict(X_test)
a = F2020ML.F2020_RMSE(y_test, y_pred)
print(best_parm)
print('Values RMSE:', a, '.......', 'Values R2:', best_score)

plt.show()