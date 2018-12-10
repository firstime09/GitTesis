## Data 870 Cidanau menggunakan SVR
import pandas as pd
import numpy as np
from osgeo import gdal
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from CodeFix import fungsi

## Load function RMSE and Rsquared from fungsi 
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

df = pd.read_excel('D:/00AllData/00 Data Load/data plot.xlsx') # Data 300 Cidanau
features = df.columns.drop(['Kategori', 'FRCI', 'Band_1', 'Band_9'])
targets = 'FRCI'
df_X = np.asarray(df[features])
df_y = np.asarray(df[targets])
# print(df.head())

## For Check Manually
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=4)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clfSVR = SVR(kernel='rbf', C=10, epsilon=0.1, gamma=0.1)
clfSVR.fit(X_train, y_train)
scores = clfSVR.score(X_test, y_test)
y_pred = clfSVR.predict(X_test)
print('Acc Score SVR: ', scores)
print('RMSE Score SVR: ', rMSE(y_test, y_pred))

## Load For new data predict
Newdf = pd.read_excel('D:/00AllData/00 Data Load/New_Data Train CIDANAU.xlsx') # Data predict
Newfeatures = Newdf.columns.drop(['Band_1', 'Band_9'])
Newdf_X = np.asarray(Newdf[Newfeatures])

# NewSC = StandardScaler()
# Newdf_X = sc.fit_transform(Newdf_X)
y_pred_new = clfSVR.predict(Newdf_X) # Predict new data set from model clfSVR
# print(y_pred_new)

Cidanau_1 = y_pred_new.reshape(597,728)

export_array (Cidanau_1,"C:/Users/Forests2020/Documents/GitHub/GitTesis/Random Forest/Cidanau_RF1.TIF" )