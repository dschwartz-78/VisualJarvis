# TODO
#   Docstring

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

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
if args.get("webcam"):
    if args.get("verbose"):
        print("Source : webcam")
    vs = VideoStream(src=0).start()
    WEBCAM_MODE = True
    time.sleep(2.0)
elif args.get("video"):
    if args.get("verbose"):
        print("Source : vidéo")
    vs = cv2.VideoCapture(args["video"])
    VIDEO_MODE = True
    time.sleep(2.0)
elif args.get("image"):
    if args.get("verbose"):
        print("Source : image")
    IMAGE_MODE = True

# Boucle lecture du flux entrant, frame par frame
while True:
    # Implémentation du diagramme d'état
    # Si état == 0
    #   calculer métriques : flux optique, dir_ppale, bounding_rect
    #   test de changement d'état : rester 0 ou devenir 1
    # Si état == 1
    #   calculer 


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


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
