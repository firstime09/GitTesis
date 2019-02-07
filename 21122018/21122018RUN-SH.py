## Run Modul_ML and Modul_TOPO
import pandas as pd
import numpy as np
from os import system
import sys
sys.path.append('D:\\FORESTS2020\\GITHUB\\Plugin\\GitTesis\\21122018')

from Modul_Topo import FTEST01
from Modul_Topo.FORESTS2020 import allFunc
from Modul_ML.F17122018ML import F2020ML

# df = pd.read_csv(r'D:\FORESTS2020\GITHUB\Plugin\GitTesis\SVR\DBSCAN_5_I.csv')
# df = pd.read_csv(r'D:\FORESTS2020\GITHUB\Plugin\GitTesis\SVR\Dbscan1000Samp_0.05_5.csv')
script = sys.argv[0]
input = sys.argv[1]
df = pd.read_csv(input)
# print(df.head())
column = ['Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7']
# column = ['Band_4']
target = 'frci'
dfX = pd.DataFrame(df, columns=column)
dfY = np.asarray(df[target])
# Check SVR and RFR parameters
print('Values SVR: ', F2020ML.F2020_SVR(dfX, dfY, 0.3, 4))
print('Values RFR: ', F2020ML.F2020_RFR(dfX, dfY, 0.3, 4))
