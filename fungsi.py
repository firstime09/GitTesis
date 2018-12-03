import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class allFunction:
    def F2020_SVR(dataX, dataY, tsize, rstate): #--- Model SVR kernel=rbf FORESTS2020
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        sc = StandardScaler()
        # sc.fit(X_train)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        best_score = 0
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            for C in [0.001, 0.01, 0.1, 1, 10, 100]:
                for epsilon in [0.001, 0.01, 0.1, 1, 10, 100]:
                    # Train Model SVR
                    clfSVR = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                    clfSVR.fit(X_train, y_train)
                    score = clfSVR.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_parameters = {'C': C, 'gamma': gamma, 'epsilon': epsilon}
        return(best_score, best_parameters)

    def F2020_SVM(krnel, dataX, dataY, tsize, rstate): #--- Model SVM (30/11-2018)
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        best_score = 0
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
                # Make the model SVM
                clfSVM = SVC(kernel='krnel', C=C, gamma=gamma)
                clfSVM.fit(X_train, y_train)
                score = clfSVM.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_parameters = {'C':C, 'gamma':gamma}
        return(best_score, best_parameters)

    def normaliZe(data): #--- Normalisasi Data
        outp = np.array(np.ravel(data), copy=True)
        maxVal = np.max(np.abs(outp))
        if maxVal > 0:
            for i in range(0,len(outp)):
                outp[i] = outp[i]/maxVal
        return outp

#--- Perhitungan Segmentasi Citra Dengan pengurangan antar Channel Warna
    def imInput1(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        gr = g - r

        return gr # Informasi Area Penyakit

    def imInput2(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        rg = r - g

        return rg # Informasi Area Daun Sehat

    def imInput3(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        gb = g - b

        return gb # Informasi Keseluruhan Daun

    def imInput4(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        bg = b - g

        return bg

    def imInput5(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        rb = r - b

        return rb

    def imInput6(data):
        im = cv2.imread(data)
        b, g, r = cv2.split(im)
        br = b - r

        return br
#--- Hanya menggunakan 3 pengurangan Channel karena 3 tiganya yang sangat berpengaruh

    def spectrum_print(data, no): #---- Membuat Ploting Nilai Spectrum Wavelet
        rows = len(data)
        cols = len(data[0])
        ##### spectrum ###########
        plt.plot(np.ravel(data))
        plt.title('coeff')
        plt.xlim(0, rows * cols)
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.savefig(no + '_coeff.png')
        plt.close()
        plt.close('all')
        plt.gcf().clear()

    def total_intensity(data): #--- Menghitung Nilai Intensity data
        temp = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                temp += data[i, j]
        return temp

    def prob_entropy(data, total): #--- Menghitung nilai Entropy
        ent = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                temp1 = data[i, j] / total
                if temp1 > 0:
                    ent += -(temp1 * np.log2(temp1)) #---- Rumus Shannon Entropy
        return ent

    def prob_energy(data):  # --- Menghitung nilai Energy
        ent = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                #temp1 = data[i, j] / total
                #if temp1 > 0:
                ent += np.abs(data[i, j])  # ---- Rumus Nilai Energy
        return ent

    def hom_prob(data, total):
        hom = 0.0
        for i in range(len(data)):
            for j in range(len(data[i])):
                temp2 = data[i, j] / total
                # temp2 = total(int(data[i][j]))
                if temp2 > 0:
                    hom += temp2 / (1 + (abs(i - j)))
        return hom

    def Homegeneity(data, total):
        homogen = 0.0
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                homogen += total[int(data[i][j])] / (1 + abs(i - j))
        return homogen

    def sumPix(data):                           #--- Test [24-08-2018]
        pix = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                pix = np.sum(data[i, j])
        return pix
