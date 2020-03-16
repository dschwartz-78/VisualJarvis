""" Test de détecteur de mouvement temps-réel maison sur cas simulé """

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi  # convolution
import cv2

# Paramètres d'entrée
N = 100         # Largeur frame simulée
Nbox = 20       # Largeur carré central qui va être déplacé

# Génération des données pour les frames : sans bruit, avec bruit
# bkgnd = np.zeros((N,N))                                 # sans bruit
# box = np.ones((Nbox,Nbox))
bkgnd = np.random.standard_normal((N,N))               # avec bruit
box = np.random.standard_normal((Nbox,Nbox)) + 4


# Initialisation
frame_prvs = np.zeros(bkgnd.shape)
diff_prvs = np.zeros(bkgnd.shape)
N2 = np.round(N/2).astype(int)
Nbox2 = np.round(Nbox/2).astype(int)
ker = cv2.getGaussianKernel(21, 4)  # le rayon du noyau est fixé à 10px
ker = ker[10:]

for i in np.arange(7):
    frame = bkgnd.copy()
    # Simulation mouvement : ascendant, descendant, droite, gauche
    frame[N2-Nbox2-i:N2-Nbox2+Nbox-i, N2-Nbox2:N2-Nbox2+Nbox] = box   # ascendant
    # frame[N2-Nbox2+i:N2-Nbox2+Nbox+i, N2-Nbox2:N2-Nbox2+Nbox] = box   # descendant
    # frame[N2-Nbox2:N2-Nbox2+Nbox, N2-Nbox2+i:N2-Nbox2+Nbox+i] = box   # droite
    # frame[N2-Nbox2:N2-Nbox2+Nbox, N2-Nbox2-i:N2-Nbox2+Nbox-i] = box   # gauche
    # plt.imshow(frame) ; plt.pause(.5)

    diff = abs(frame_prvs.astype("float") - frame.astype("float"))
    diffXa = abs(diff[:-1,:] + diff_prvs[1:,:])
    diffXb = abs(diff[1:,:] + diff_prvs[:-1,:])
    diffYa = abs(diff[:,:-1] + diff_prvs[:,1:])
    diffYb = abs(diff[:,1:] + diff_prvs[:,:-1])

    # Classification
    classifTmp = np.dstack((diffXa[:,0:-1], diffXb[:,0:-1], diffYa[0:-1,:], diffYb[0:-1,:]))
    classif = np.argmax(classifTmp, 2)
    mask = np.nonzero(diff)
    val_classif = classif[mask[0]-1, mask[1]-1]
    nb_0 = np.sum(val_classif == 0)
    nb_1 = np.sum(val_classif == 1)
    nb_2 = np.sum(val_classif == 2)
    nb_3 = np.sum(val_classif == 3)
    if np.argmax((nb_0, nb_1, nb_2, nb_3)) == 0:
        print('Mouvement ascendant')
    elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 1:
        print('Mouvement descendant')
    elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 2:
        print('Mouvement gauche')
    elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 3:
        print('Mouvement droite')
    plt.imshow(classif) ; plt.pause(.5)

    frame_prvs = frame
    diff_prvs = diff

    # # Possible problème car le décalage n'a pas été fait il me semble
    # diff = abs(frame_prvs.astype("float") - frame.astype("float"))
    # diffConvX = ndi.convolve(diff, ker)
    # diffConvY = ndi.convolve(diff, np.transpose(ker))
    # diffConvX_prvs = ndi.convolve(diff_prvs, ker)
    # diffConvY_prvs = ndi.convolve(diff_prvs, np.transpose(ker))
    # diffX = diffConvX + diffConvX_prvs
    # diffY = diffConvY + diffConvY_prvs

    # if i == 4:
    #     plt.figure() ; plt.imshow(frame) ; plt.title('frame') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffConvX) ; plt.title('diffConvX') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffConvY) ; plt.title('diffConvY') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffConvX_prvs) ; plt.title('diffConvX_prvs') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffConvY_prvs) ; plt.title('diffConvY_prvs') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffX) ; plt.title('diffX') ; plt.pause(.1)
    #     plt.figure() ; plt.imshow(diffY) ; plt.title('diffY') ; plt.pause(.1)
    #     plt.pause(-1)
plt.pause(-1)
