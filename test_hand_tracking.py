"""
Tests pour savoir comment :
    - détecter la main en position ouverte et fermée
    - faire le suivi du mouvement (ex mvt ascendant)
    - prendre une décision quand au mouvement identifié
"""

# from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
# import imutils
import time
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy import ndimage as ndi  # convolution
import os       # Controle du son

def beep():
    """ Faire un beep système """
    print('\a')

def boundingRect(bw):
    """ Renvoie les 2 coins opposés du rectangle englobant les pixels non-nuls d'une image binaire """
    w_pts = np.nonzero(bw)
    z1 = complex(w_pts[0].min(), w_pts[1].min())
    z2 = complex(w_pts[0].max(), w_pts[1].max())
    return [z1, z2]

def linearDynamic(im, alpha=None, beta=None):
    """ Change la dynamique d'une image à alpha, beta """
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im)
    if alpha == None:
        alpha = minVal
    if beta == None:
        beta = maxVal
    out = (beta-alpha) / (maxVal-minVAl) * (im - minVal) + alpha
    return out

# Lecture des arguments d'entrée : webcam ou fichier video
ap = argparse.ArgumentParser(description="Control du volume par une source vidéo (webcam, fichier vidéo ou image)")
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
ap.add_argument("-i", "--image", help="Path to the (optional) image")
ap.add_argument("-w", "--webcam", action="store_true", help="Use webcam as source")
ap.add_argument("-V", "--verbose", action="store_true", help="Display processing steps")
args = vars(ap.parse_args())

IMAGE_MODE = False
VIDEO_MODE = False
WEBCAM_MODE = False
VERBOSE_MODE = False
if args.get("verbose"):
    VERBOSE_MODE = True
if args.get("webcam"):
    if VERBOSE_MODE:
        print("Source : webcam")
    vs = VideoStream(src=0).start()
    WEBCAM_MODE = True
    time.sleep(2.0)
elif args.get("video"):
    if VERBOSE_MODE:
        print("Source : vidéo")
    vs = cv2.VideoCapture(args["video"])
    VIDEO_MODE = True
    # Laisser le temps à la caméra de chauffer
    time.sleep(2.0)
elif args.get("image"):
    if VERBOSE_MODE:
        print("Source : image")
    IMAGE_MODE = True

# Lecture de la première image pour initialiser le flot optique
frame = vs.read()
frame = frame[1] if VIDEO_MODE else frame

# Initialisation de variables
ker = cv2.getGaussianKernel(21, 4)  # le rayon du noyau est fixé à 10px
ker = ker[10:]
dsize = (int(np.around(frame.shape[1]*600/frame.shape[0])), 600)
frame = cv2.resize(frame, dsize)
frame_prvs = frame
frame_prvs = cv2.cvtColor(frame_prvs, cv2.COLOR_BGR2GRAY)
diff_prvs = np.zeros(frame_prvs.shape)
hsv = np.zeros_like(frame_prvs)
hsv[..., 1] = 255
state_list = np.ndarray(0)
frame_state_list = np.ndarray(0)

# Boucle lecture du flux entrant, frame par frame
if VERBOSE_MODE:
    print("Lecture du flux entrant")
iFrame = 0
maxValmax = 0
liste_mag_max = list()
liste_RectPixels= list()
# print('iFrame\tMag - MagMax')
t0 = cv2.getTickCount()
while True:
    # Lecture d'une frame
    if IMAGE_MODE:
        frame = cv2.imread(args["image"])
    else:
        frame = vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if VIDEO_MODE else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
    if VERBOSE_MODE:
        print(iFrame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, dsize)
    print((cv2.getTickCount() - t0)/ cv2.getTickFrequency()) ; t0 = cv2.getTickCount()
    cv2.imshow('FRAME', frame) ; cv2.waitKey(30)

    diff = abs(frame_prvs.astype("float") - frame.astype("float"))
    retval, disp = cv2.threshold(diff, 4, 1, cv2.THRESH_TOZERO)
    closed = cv2.erode(disp, None, iterations=3)
    opened = cv2.dilate(closed, None, iterations=3)
    # cv2.imshow('FRAME', closed) ; cv2.waitKey(100)
    #cv2.imshow('FRAME', diff/255*10) ; cv2.waitKey(30)

    # Détection sauce David
    # Si aucun mouvement depuis 3 secondes, alors supprimer l'historique des mouvements
    if (cv2.getTickCount() - t0)/ cv2.getTickFrequency() > 3:
        state_list = np.ndarray(0)
    mask = np.nonzero(opened)
    if len(mask[0]) < 100:  # Aucun mouvement, interprétation des mouvements et gestes
        # Gestion des mouvements unitaires (issus des frames)
        asc =  np.sum(frame_state_list == 0) - 1 # car ajouté un élément lors définition variable
        desc = np.sum(frame_state_list == 1)
        gau =  np.sum(frame_state_list == 2)
        droi = np.sum(frame_state_list == 3)
        if frame_state_list.shape[0] > 2:
            if   asc > desc and asc > gau and asc > droi:
                if VERBOSE_MODE:
                    print('    Geste ascendant')
                    print('    ' + np.array2string(frame_state_list))
                state_list = np.append(state_list, 0)
                t0 = cv2.getTickCount()
            elif desc > asc and desc > gau and desc > droi:
                if VERBOSE_MODE:
                    print('    Geste descendant')
                    print('    ' + np.array2string(frame_state_list))
                state_list = np.append(state_list, 1)
                t0 = cv2.getTickCount()
            elif gau > asc and gau > desc and gau > droi:
                if VERBOSE_MODE:
                    print('    Geste gauche')
                    print('    ' + np.array2string(frame_state_list))
                state_list = np.append(state_list, 2)
                t0 = cv2.getTickCount()
            elif droi > asc and droi > gau and droi > desc:
                if VERBOSE_MODE:
                    print('    Geste droite')
                    print('    ' + np.array2string(frame_state_list))
                state_list = np.append(state_list, 3)
                t0 = cv2.getTickCount()
        frame_state_list = np.ndarray(0)
        # Gestion des gestes (issus des mouvements unitaires)
        print('    #### ' + np.array2string(state_list))
        if state_list.shape[0] >= 4:
            if   all( state_list[-4:] == np.array([0,1,0,0]) ):
                print('      Séquence : augmenter volume')
                os.system('osascript -e "set volume output volume (output volume of (get volume settings)+20)"')
                beep()
                state_list = np.ndarray(0)
                t0 = cv2.getTickCount()
            elif all( state_list[-4:] == np.array([0,1,1,0]) ):
                print('      Séquence : baisser volume')
                os.system('osascript -e "set volume output volume (output volume of (get volume settings)-20)"')
                beep()
                state_list = np.ndarray(0)
                t0 = cv2.getTickCount()
            elif all( state_list[-4:] == np.array([0,1,2,0]) ):
                print('      Séquence : morceau suivant')
                beep()
                state_list = np.ndarray(0)
                t0 = cv2.getTickCount()
            elif all( state_list[-4:] == np.array([0,1,3,0]) ):
                print('      Séquence : morceau précédent')
                beep()
                state_list = np.ndarray(0)
                t0 = cv2.getTickCount()
    else:       # Détection de mouvements
        # print('  %d' % len(mask[0]))
        diffXa = abs(diff[:-1,:] + diff_prvs[1:,:])
        diffXb = abs(diff[1:,:] + diff_prvs[:-1,:])
        diffYa = abs(diff[:,:-1] + diff_prvs[:,1:])
        diffYb = abs(diff[:,1:] + diff_prvs[:,:-1])
        # cv2.imshow('diffXa', diffXa.astype(float)/255*10) ; cv2.waitKey(30)
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(diff)
        # print((minVal, maxVal))
        # if iFrame == 42:
        #     plt.imshow(disp) ; plt.pause(0.)
        #     break
        #     #plt.pause(-1)

        # Classification
        # plt.imshow(mask) ; plt.pause(0.005)
        classifTmp = np.dstack((diffXa[:,0:-1], diffXb[:,0:-1], diffYa[0:-1,:], diffYb[0:-1,:]))
        classif = np.argmax(classifTmp, 2)
        val_classif = classif[mask[0]-1, mask[1]-1]
        nb_0 = np.sum(val_classif == 0)
        nb_1 = np.sum(val_classif == 1)
        nb_2 = np.sum(val_classif == 2)
        nb_3 = np.sum(val_classif == 3)
        # print('  %d\t%d\t%d\t%d' % (nb_0, nb_1, nb_2, nb_3))
        if np.argmax((nb_0, nb_1, nb_2, nb_3)) == 0:
            if VERBOSE_MODE:
                print('  Mouvement ascendant')
            frame_state_list = np.append(frame_state_list, 0)
        elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 1:
            if VERBOSE_MODE:
                print('  Mouvement descendant')
            frame_state_list = np.append(frame_state_list, 1)
        elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 2:
            if VERBOSE_MODE:
                print('  Mouvement gauche')
            frame_state_list = np.append(frame_state_list, 2)
        elif np.argmax((nb_0, nb_1, nb_2, nb_3)) == 3:
            if VERBOSE_MODE:
                print('  Mouvement droite')
            frame_state_list = np.append(frame_state_list, 3)
        # plt.imshow(classif) ; plt.pause(.005)
        # print((cv2.getTickCount() - t0)/ cv2.getTickFrequency()) ; t0 = cv2.getTickCount()

    # # Détection par différence d'images successives
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # #frame = cv2.GaussianBlur(frame, (9, 9), 4)
    # diff = abs(frame_prvs.astype("float") - frame.astype("float"))
    # diffConvX = ndi.convolve(diff, ker)
    # diffConvY = ndi.convolve(diff, np.transpose(ker))
    # diffConvX_prvs = ndi.convolve(diff_prvs, ker)
    # diffConvY_prvs = ndi.convolve(diff_prvs, np.transpose(ker))
    # # diffX = cv2.subtract(diffConvX, diffConvX_prvs)
    # # diffY = cv2.subtract(diffConvY, diffConvY_prvs)
    # diffX = diffConvX + diffConvX_prvs
    # diffY = diffConvY + diffConvY_prvs
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(diffX)
    # print('%2.2f\t%2.2f' % (minVal, maxVal))
    # # retval, diffDisp = cv2.threshold(diff / 10 * 255, 255, 255, cv2.THRESH_TRUNC)
    # # cv2.imshow('DIFF', diffX/255*2)
    # # cv2.imshow('DIFF SOBEL', diff_prvs /255*2)
    # # cv2.setWindowProperty('DIFF', cv2.WND_PROP_TOPMOST, 2)
    # if iFrame == 19:
    #     # plt.figure() ; plt.imshow(diff) ; plt.title('diff')
    #     # plt.figure() ; plt.imshow(diff_prvs) ; plt.title('diff_prvs')
    #     plt.figure() ; plt.imshow(diffConvX) ; plt.title('diffConvX')
    #     # plt.figure() ; plt.imshow(diffConvY) ; plt.title('diffConvY')
    #     plt.figure() ; plt.imshow(diffConvX_prvs) ; plt.title('diffConvX_prvs')
    #     # plt.figure() ; plt.imshow(diffConvY_prvs) ; plt.title('diffConvY_prvs')
    #     plt.figure() ; plt.imshow(diffX) ; plt.title('diffX')
    #     # plt.figure() ; plt.imshow(diffY) ; plt.title('diffY')
    #     plt.pause(-1)
    #     break

    # # Détection de la main par flot optique
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # frame = cv2.GaussianBlur(frame, (5, 5), 3)
    # flow = cv2.calcOpticalFlowFarneback(frame_prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # magDisp = cv2.threshold(mag / 22 * 255, 255, cv2.THRESH_TRUNC)
    # # cv2.imshow('MAG', magDisp)
    # frame2 = frame.copy()
    # print(type(frame))
    # print(frame.shape)
    # frame2[mag > 5] = 255
    # cv2.imshow('MAG_THRES', frame2)
    #
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
    # if maxVal > maxValmax:
    #     maxValmax = maxVal
    # print('%d\t%3.1f - %3.1f' % (iFrame, maxVal, maxValmax))
    # liste_mag_max.append(maxVal)
    # #mag[0,0] = 22  # pour l'amplitude d'affichage
    #
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #
    # # plt.plot(liste_mag_max) ; plt.pause(0.05) # Laisser le temps de s'afficher
    #
    # # Moyenne des angles
    # # if iFrame > 30:
    # #retval, mag2 = cv2.threshold(mag, 8, 1, cv2.THRESH_BINARY)
    # bw_mag = mag > 8
    # liste_ang = ang[bw_mag]
    # if len(liste_ang) > 100:
    #     mean_real_part = np.sum(np.cos(liste_ang))
    #     mean_imag_part = np.sum(np.sin(liste_ang))
    #     mean_cpx = complex(mean_real_part, mean_imag_part)
    #     dir_ppale = np.angle(mean_cpx)
    #     force_ppale = abs(mean_cpx) / len(liste_ang)
    #     if cv2.inRange( (dir_ppale, ), -np.pi/2-np.pi/4, -np.pi/2+np.pi/4) > 0:
    #         print('  Test direction: validé')
    #         test_asc = True
    #         # cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         # print(type(cnts))
    #         # print(cnts)
    #         # print('-------------------------------')
    #         # cnts2 = imutils.grab_contours(cnts)
    #         # print(type(cnts2))
    #         # print(cnts2)
    #         # bw_mag2 = cv2.morphologyEx(bw_mag, cv2.MORPH_OPEN, cv2.getStructuringElement(cv.MORPH_RECT, 4, cv2.Point(-1,-1)), cv2.Point(-1,-1), 1, cv2.BORDER_CONSTANT, 0)
    #         # cv2.imshow('Affichage', bw_mag2)
    #
    #         # # Code alla David, remplacé par code OpenCV
    #         # z1, z2 = boundingRect(1.0 * bw_mag)
    #         # ScreenPixels = mag.shape[0] * mag.shape[1]
    #         # RectPixels = abs(z2.real-z1.real) * abs(z2.imag-z1.imag)
    #         # liste_RectPixels.append(RectPixels)
    #         # plt.plot(np.sqrt(liste_RectPixels)) ; plt.pause(0.05)
    #         # if cv2.inRange((RectPixels, ), ScreenPixels / 80, ScreenPixels / 16):
    #         #     print('    Test taille: validé')
    #         #     box = [[z1.real, z1.imag],[z1.real, z2.imag],[z2.real, z2.imag],[z2.real, z2.imag]]
    #         #     cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    #         # else:
    #         #     print('    Test taille: non validé')
    #     else:
    #         print('  Test direction: non validé')
    #         test_asc = False
    #     #break

    # k = cv2.waitKey(30) & 0xff   # 30
    # if k == 27:
    #     break
    # elif k == ord('i'):
    #     print(iFrame)
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png', frame)
    #     cv2.imwrite('opticalhsv.png', bgr)

    # Conditions de fin de boucle
    iFrame = iFrame + 1
    if IMAGE_MODE:
        break
    frame_prvs = frame
    diff_prvs = diff
    cv2.waitKey(60)

print('Nombre de frames : %d' % iFrame)
print('Amplitude maximale du flot optique : %2.1f' % maxValmax)


# if we are not using a video file, stop the camera video stream
if VERBOSE_MODE:
    print("Fermeture du flux entrant")
if WEBCAM_MODE:
    vs.stop()
# otherwise, release the camera
elif VIDEO_MODE:
    vs.release()

# close all windows
cv2.destroyAllWindows()
