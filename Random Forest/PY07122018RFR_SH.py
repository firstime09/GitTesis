import math, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from osgeo import gdal

def rSquared(ActualY, PredictY):  # --- value of r^2 in statistic (04/12-2018)
    rScores = (1 - sum((ActualY - PredictY) ** 2) / sum((ActualY - ActualY.mean(axis=0)) ** 2))
    return rScores
def rMSE(ActualY, PredictY):  # --- Root mean squared error in statistical model (04/12-2018)
    rootMSE = (math.sqrt(sum((ActualY - PredictY)**2) / ActualY.shape[0]))
    return rootMSE

band = gdal.Open("C:/Users/Forests2020/Documents/GitHub/GitTesis/RAW DATA/Band 5.img")
def export_array(in_array, output_path):
    """This function is used to produce output of array as a map."""
    global proj, geotrans, row, col
    proj     = band.GetProjection()
    geotrans = band.GetGeoTransform()
    row      = band.RasterYSize
    col      = band.RasterXSize
    driver   = gdal.GetDriverByName("GTiff")
    outdata  = driver.Create(output_path, col, row, 1, gdal.GDT_CFloat32)
    outband  = outdata.GetRasterBand(1)
    outband.SetNoDataValue(-9999)
    outband.WriteArray(in_array)
    # Georeference the image
    outdata.SetGeoTransform(geotrans)
    # Write projection information
    outdata.SetProjection(proj)
    outdata.FlushCache()
    outdata = None

df = pd.read_excel('C:/Users/Forests2020/Documents/GitHub/GitTesis/Random Forest/FRCI 870_2611N.xlsx')
features = df.columns.drop(['FID', 'frci', 'Class', 'Band_9', 'Band_1'])
targets = 'frci'
df_X = np.asarray(df[features])
df_y = np.asarray(df[targets])

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=4)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clfRF = RandomForestRegressor(n_estimators=50, random_state=4)
clfRF.fit(X_train, y_train)
y_pred = clfRF.predict(X_test)
print('RSquare : ', rSquared(y_test, y_pred))
print('RMSe :', rMSE(y_test, y_pred))

plt.title('RMSE : ' + str(rMSE(y_test, y_pred)))
plt.xlabel('y_Predict')
plt.ylabel('y_Actual')
plt.scatter(y_pred, y_test, label='Data')
# plt.plot(X_test, y_pred)
plt.legend()
plt.show()

Ndf = pd.read_excel('C:/Users/Forests2020/Documents/GitHub/GitTesis/Random Forest/New_Data Train CIDANAU.xlsx')
Newfeatures = Ndf.columns.drop(['Band_1', 'Band_9'])
Ndf_X = np.asarray(Ndf[Newfeatures])

Nsc = StandardScaler()
Ndf_X_sc = sc.fit_transform(Ndf_X)
#
y_pred_new = clfRF.predict(Ndf_X)
y_pred_nSC = clfRF.predict(Ndf_X_sc)
print(y_pred_nSC - y_pred_new)
print(type(y_pred_new), y_pred_new.shape)

Cidanau_1 = y_pred_new.reshape(597,728)

export_array (Cidanau_1,"C:/Users/Forests2020/Documents/GitHub/GitTesis/Random Forest/Cidanau_RF1.TIF" )
