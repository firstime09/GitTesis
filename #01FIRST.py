import glob, pickle, cv2
import fungsi
import pywt

segment = fungsi.allFunction
spectrum = fungsi.allFunction
t_inten = fungsi.allFunction
t_entro = fungsi.allFunction
t_energ = fungsi.allFunction

entropy = list()
energy = list()

for image in glob.glob('D:/PyProject/#00Data/S2-DATA_JABON/Foto_Jabon_Run/Hawar/*.jpg'):
    im_info1 = segment.imInput1(image) #--- Area Penyakit
    # im_info2 = segment.imInput2(image) #--- Area Daun Sehat

    coeff1 = pywt.dwt2(im_info1, "db2")  # --- Pergitungan Wavelet (haar,db2,db3)
    LL, (LH, HL, HH) = coeff1

    # Hitung nilai Entropy berdasarkan Hasil wavelet
    hitInten = t_inten.total_intensity(LL)
    hitEntro = t_entro.prob_entropy(LL, hitInten)
    # print(image + 'Nilai Entropy:',hitEntro)
    entropy.append(hitEntro)

    # Hitung nilai Entropy berdasarkan Hasil wavelet
    hitEnerg = t_energ.prob_energy(LL)
    # print(image + 'Nilai Energy:',hitEnerg)
    energy.append(hitEnerg)

    # cv2.imwrite(image + 'Hasil-Daun-G-R.png', im_info1)
    # cv2.imwrite(image + 'Hasil-Daun-R-G.png', im_info2)

    #--- Print Hasil Ploting Wavelet
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv1-(A).png', LL)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv1-(V).png', LH)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv1-(H).png', HL)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv1-(D).png', HH)
    #
    # spectrum.spectrum_print(LL, image + 'Hasil-(a)_Level-1')
    # spectrum.spectrum_print(LH, image + 'Hasil-(v)_Level-1')
    # spectrum.spectrum_print(HL, image + 'Hasil-(h)_Level-1')
    # spectrum.spectrum_print(HH, image + 'Hasil-(d)_Level-1')

    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv2-(A).png', LL2)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv2-(V).png', LH2)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv2-(H).png', HL2)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv2-(D).png', HH2)
    #
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv3-(A).png', LL3)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv3-(V).png', LH3)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv3-(H).png', HL3)
    # cv2.imwrite(image + 'Hasil-Daun_Area-Penyakit_lv3-(D).png', HH3)

# Data pickle Bercak 091118
# with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Bercak(Entropy).pkl','wb') as f:
#     pickle.dump(entropy, f)
# with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Bercak(Energy).pkl','wb') as f:
#     pickle.dump(energy, f)

# Data pickle Hawar 091118
with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Hawar(Entropy).pkl','wb') as f:
    pickle.dump(entropy, f)
with open('091118Data_S2-DATA_JABON_Foto_Jabon_Run_Hawar(Energy).pkl','wb') as f:
    pickle.dump(energy, f)