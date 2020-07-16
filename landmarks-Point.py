import cv2
import numpy as np
import dlib
from math import cos, sqrt

cap = cv2.VideoCapture('http://100.97.218.207/camara/TWD/TWD-1elenco.jpg')
#cap = cv2.VideoCapture(1)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        print (face)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            if (n == 27):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                facecenter = (x, y)
            if (n == 36):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                Ojo1 = (x, y)
                Ojo1x = int(x)
                Ojo1y = int(y)
            if (n == 45):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                Ojo2 = (x, y)
                Ojo2x = int(x)
                Ojo2y = int(y)


    cv2.imshow("Frame", frame)

    print ("-- facecenter: ", facecenter)
    print ("-- Ojo1: ", Ojo1)
    print ("-- Ojo2: ", Ojo2)

    IDP = sqrt((Ojo1x - Ojo2x)**2 + (Ojo1y - Ojo2y)**2)  
    print ("IDP: ", IDP)

    key = cv2.waitKey(0)
    if key == 27:
        break