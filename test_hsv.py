"""
Test de la détection de gestes de controle par détection de la couleur verte.
"""

import cv2
image = cv2.imread("hand_control_green.jpg")
# image = imutils.resize(image, width=600)
blurred = cv2.GaussianBlur(image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow("IMAGE", image)
cv2.imshow("BLURRED", blurred)
cv2.imshow("HSV", hsv)
cv2.imshow("HUE", hsv[:,:,0])
cv2.waitKey(-1)

# Détection par HSV impossible
""" # Bout de code supprimé de la grande boucle du scirpt test_hand_tracking.py
# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize(frame, width=600)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow("Point de vue ordinateur", frame)
cv2.imshow("Premiere fenetre", hsv[:,:,0])
cv2.imshow("Seconde fenetre", hsv[:,:,1])
cv2.imshow("Troisieme fenetre", hsv[:,:,2])
Vcolored = cv2.applyColorMap(hsv[:,:,1], cv2.COLORMAP_RAINBOW)
cv2.imshow("Nouvelle fenetre", Vcolored)
key = cv2.waitKey(-1) & 0xFF
"""

"""
Impossible de détecter la couleur verte dans les lumières LED
"""
