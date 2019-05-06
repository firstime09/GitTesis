from __future__ import print_function, division
import sys, glob
sys.path.append('D:\\FORESTS2020\\GITHUB\\Plugin\\GitTesis\\21122018')
sys.path.append('C:\\Program Files\\GDAL')
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

#### stack layer data
path_layer = r"D:\FORESTS2020\GITHUB\Plugin\GitTesis\TIF RAW\stack"
file_layer = glob.glob(path_layer + "/*.tif")
# system('gdal_merge -o cidanau_stack.tif {fileraster}'.format(fileraster=file_layer))
# gm.main(['', '-o', 'cidanau_stack.tif', '{fileraster}'.format(fileraster=file_layer)])
file_vrt = path_layer + "/stacked.vrt"
file_tif = path_layer + "/cidanau_stack.tif"
vrt = gdal.BuildVRT(file_vrt,file_layer, separate=True)
stack_layer = gdal.Translate(file_tif, vrt)


#####
AOI_1 = gdal.Open(file_tif)
AOI_2 = AOI_1.GetRasterBand(1).ReadAsArray()
AOI= AOI_2 > 0

####
img_ds = gdal.Open(file_tif, gdal.GA_ReadOnly)
img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
# print(img)
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
# roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# plt.subplot(121)
plt.imshow(img[:, :, 4], cmap = plt.cm.Greys_r)
plt.title('DATA LandSat')

#### load data sample
path2 = r"D:\FORESTS2020\GITHUB\Plugin\GitTesis"
loadFile = pd.read_excel(path2 + '/Cidanau_Join_LINE6_61.18.xlsx')
select_col = ['Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
select_row = 'frci'

dfx = pd.DataFrame(loadFile, columns=select_col)
dfy = np.asarray(loadFile[select_row])

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=5)

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


#### Prediction
y_pred = clfSVR1.predict(X_test)
a = F2020ML.F2020_RMSE(y_test, y_pred)

new_shape = (img.shape[0] * img.shape[1], img.shape[2])

img_as_array = img[:, :, :6].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

# Now predict for each pixel
class_prediction = clfSVR1.predict(img_as_array)
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
# ####class_prediction_adjusted
# class_prediction[class_prediction < 0] = 0
####class_prediction_adjusted
# class_prediction[class_prediction <= 0] = 0.01
final_prediction = class_prediction * AOI
# def Min_Max_Norm(data):
#     Norm = (data - np.min(data)) / (np.max(data) - np.min(data))
#     return Norm
#
# predi_norm = Min_Max_Norm(class_prediction)
# Make data prediction to TIF file
output_path = path_layer + "/Frci_predic_Cidanau-AOI-020519_CF64.TIF"
# output_path = path_layer + "/AOI.TIF"
raster = file_tif
in_path = gdal.Open(raster)
# in_array = class_prediction
in_array = final_prediction
# global proj, geotrans, row, col
proj        = in_path.GetProjection()
geotrans    = in_path.GetGeoTransform()
row         = in_path.RasterYSize
col         = in_path.RasterXSize
driver      = gdal.GetDriverByName("GTiff")
outdata     = driver.Create(output_path, col, row, 1, gdal.GDT_CFloat64)
outband     = outdata.GetRasterBand(1)
outband.SetNoDataValue(-9999)
outband.WriteArray(in_array)
# outband.WriteArray(AOI)
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