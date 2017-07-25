import dlib
import cv2
import sys
import os
import argparse


def detectFaces(pic):

    detector = dlib.get_frontal_face_detector()

    img = cv2.imread(pic)
    
    color = (255, 0, 0)

    coors = detector(img, 1)
    cut_img = []

    if len(coors) > 0:
        for j, coor in enumerate(coors):
            top = coor.top()
            bottom = coor.bottom()
            left = coor.left()
            right = coor.right()
            if left - right < 80 and bottom - top < 80:
                pass
            else:
                print('test')
                cut_img.append(img[top:bottom, left:right])
                cv2.rectangle(img, (left, bottom), (right, top), color, thickness=3)
                
                return cut_img, img

    else:
        return (0, 0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='test face detection.')
    parser.add_argument(
        'file_path', type=str,
        help = 'choose file path'
    )

    args = parser.parse_args()

    cut_img, img = detectFaces(args.file_path)

    if len(cut_img):
        #cv2.imshow('cut_image', cut_img[0])
        cv2.imwrite('cut_img.jpg', cut_img[0])
        cv2.imwrite('img.jpg', img)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    else:
        print('NO FACES ARE DETECTED!')
