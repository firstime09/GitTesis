from __future__ import print_function, division
from sklearn.svm import SVR
from osgeo import gdal, gdal_array
from Modul_ML.F17122018ML import F2020ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Min_Max_Norm(data):
    Norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return Norm

gdal.UseExceptions()
gdal.AllRegister()

path1 = 'D:/GitFolder1611/GitTesis/TIF RAW/'
img_ds = gdal.Open(path1 + 'Cidanau_Stack_150319.tiff', gdal.GA_ReadOnly)
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

path2 = 'D:/00RCode/Result/Data Sumatera/Data Sumatera No_Normalize/'
loadFile = pd.read_excel(path2 + 'Cidanau_Join_LINE6_61.18.xlsx')
select_col = ['Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
select_row = 'frci'

dfx = pd.DataFrame(loadFile, columns=select_col)
dfy = np.asarray(loadFile[select_row])

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=5)
# sc = MinMaxScaler()
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

best_score = 0
for gamma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
    for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for epsilon in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
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
clfSVR1.score(X_test, y_test)
# tes= clfSVR1.score(X_test, y_test)
# y_pred = clfSVR1.predict(X_test)
y_pred = clfSVR1.predict(X_test)
a = F2020ML.F2020_RMSE(y_test, y_pred)

new_shape = (img.shape[0] * img.shape[1], img.shape[2])

img_as_array = img[:, :, :6].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

# Now predict for each pixel
class_prediction = clfSVR1.predict(img_as_array)
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
# class_prediction = Min_Max_Norm(class_prediction)

# Make data prediction to TIF file
output_path = path1 + "frci6.TIF"
raster = path1 + 'Cidanau_Stack_150319.tiff'
in_path = gdal.Open(raster)
in_array = class_prediction
# global proj, geotrans, row, col
proj        = in_path.GetProjection()
geotrans    = in_path.GetGeoTransform()
row         = in_path.RasterYSize
col         = in_path.RasterXSize
driver      = gdal.GetDriverByName("GTiff")
outdata     = driver.Create(output_path, col, row, 1, gdal.GDT_CFloat32)
outband     = outdata.GetRasterBand(1)
outband.SetNoDataValue(-9999)
outband.WriteArray(in_array)
outdata.SetGeoTransform(geotrans) # Georeference the image
outdata.SetProjection(proj) # Write projection information
outdata.FlushCache()
outdata = None

# # y_pred = clfSVR1.predict(img_as_array)
# a1 = F2020ML.F2020_RMSE(y_test, class_prediction)
print(best_parm)
print('Max Pred:', class_prediction.max(), '.....', 'Min Pred:', class_prediction.min())
print('Min img[1]:',img[1].min(),'.....','Max img[1]:',img[1].max(),'.....','Mean img[1]:',img[1].mean())
print('Min img[2]:',img[2].min(),'.....','Max img[2]:',img[2].max(),'.....','Mean img[2]:',img[2].mean())
# print(class_prediction)
print('Values RMSE:', a, '.......', 'Values R2:', best_score)
# print(new_shape, img_as_array)

# plt.imshow(class_prediction, interpolation='none')
# plt.show()