import cv2
import numpy as np
import matplotlib.pyplot as plt

class allFunction:
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
