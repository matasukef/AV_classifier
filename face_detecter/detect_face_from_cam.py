import dlib
import cv2
import sys
import os
import argparse


def detectFaces():

    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)

    color = (255, 0, 0)

    while True:
        ret, frame = cap.read()
        
        coors = detector(frame, 1)

        if len(coors) > 0:
            for j, coor in enumerate(coors):
                top = coor.top()
                bottom = coor.bottom()
                left = coor.left()
                right = coor.right()
                if left - right < 80 and bottom - top < 80:
                    pass
                else:
                    cv2.rectangle(frame, (left, bottom), (right, top), color, thickness=3)
                    
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectFaces()
