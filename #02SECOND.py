import pickle, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ------- Import Data Level-1
with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Bercak(Energy).pkl', 'rb') as f:
    Energy_Run_B_lv1 = pickle.load(f)
with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Bercak(Entropy).pkl', 'rb') as f:
    Entropy_Run_B_lv1 = pickle.load(f)

with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Hawar(Energy).pkl', 'rb') as f:
    Energy_Run_H_lv1 = pickle.load(f)
with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Hawar(Entropy).pkl', 'rb') as f:
    Entropy_Run_H_lv1 = pickle.load(f)

## ------- Import Data Level-2
# with open('Nilai-Energy_Run_B-lv2.pkl', 'rb') as f:
#     Energy_Run_B_lv2 = pickle.load(f)
# with open('Nilai-Entropy_Run_B-lv2.pkl', 'rb') as f:
#     Entropy_Run_B_lv2 = pickle.load(f)
#
# with open('Nilai-Energy_Run_H-lv2.pkl', 'rb') as f:
#     Energy_Run_H_lv2 = pickle.load(f)
# with open('Nilai-Entropy_Run_H-lv2.pkl', 'rb') as f:
#     Entropy_Run_H_lv2 = pickle.load(f)
#
## ------- Import Data Level-3
# with open('Nilai-Energy_Run_B-lv3.pkl', 'rb') as f:
#     Energy_Run_B_lv3 = pickle.load(f)
# with open('Nilai-Entropy_Run_B-lv3.pkl', 'rb') as f:
#     Entropy_Run_B_lv3 = pickle.load(f)
#
# with open('Nilai-Energy_Run_H-lv3.pkl', 'rb') as f:
#     Energy_Run_H_lv3 = pickle.load(f)
# with open('Nilai-Entropy_Run_H-lv3.pkl', 'rb') as f:
#     Entropy_Run_H_lv3 = pickle.load(f)

# print(Entropy_Run_B_lv1)
# print(Energy_Run_B_lv1)
# print(Entropy_Run_H_lv1)
# print(Energy_Run_H_lv1)

#----- Test Convert pkl to csv
a = Entropy_Run_B_lv1 + Entropy_Run_H_lv1
b = Energy_Run_B_lv1 + Energy_Run_H_lv1
c = np.r_[np.zeros(len(Entropy_Run_B_lv1)), np.ones(len(Entropy_Run_H_lv1))]

df = pd.DataFrame({'Energy': b, 'Entropy': a, 'Class': c})
df.to_csv('091118Data_S2-DATA_JABON_Foto_Jabon_Run.csv', encoding = 'utf-8', index = False)

## ------ Print Plot Hasil
fig1, ax1 = plt.subplots()
ax1.plot(Energy_Run_B_lv1, Entropy_Run_B_lv1, 'ro', label = 'Data-091118-B')
ax1.plot(Energy_Run_H_lv1, Entropy_Run_H_lv1, 'go', label = 'Data-091118-H')
#legend1 = ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
legend1 = ax1.legend(loc = 'lower right', shadow = True)
legend1.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Nilai Energy')
plt.ylabel('Nilai Entropy')
#
# fig2, ax2 = plt.subplots()
# ax2.plot(Energy_Run_B_lv2, Entropy_Run_B_lv2, 'ro', label='Data-Test-B')
# ax2.plot(Energy_Run_H_lv2, Entropy_Run_H_lv2, 'go', label='Data-Test-H')
# #legend1 = ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
# legend1 = ax2.legend(loc='lower right', shadow=True)
# legend1.get_frame().set_facecolor('#00FFCC')
# plt.xlabel('Nilai Energy')
# plt.ylabel('Nilai Entropy')
# #
# fig3, ax3 = plt.subplots()
# ax3.plot(Energy_Run_B_lv3, Entropy_Run_B_lv3, 'ro', label='Data-Test-B')
# ax3.plot(Energy_Run_H_lv3, Entropy_Run_H_lv3, 'go', label='Data-Test-H')
# #legend1 = ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
# legend1 = ax3.legend(loc='lower right', shadow=True)
# legend1.get_frame().set_facecolor('#00FFCC')
# plt.xlabel('Nilai Energy')
# plt.ylabel('Nilai Entropy')

plt.xlabel('Nilai Energy')
plt.ylabel('Nilai Entropy')
plt.show()