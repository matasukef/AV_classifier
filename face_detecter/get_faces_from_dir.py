import os
import sys
import glob
import cv2
import dlib
from tqdm import tqdm
from setup.settings import *

detector = dlib.get_frontal_face_detector()

# get the list of av actoresses
dir_list = os.listdir(INPUT_DIR)

# get the list of images for each dir
for dir_name in tqdm(dir_list):
    print(dir_name)
    if not os.path.exists(os.path.join(OUTPUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUTPUT_DIR, dir_name))
    image_files = glob.glob(os.path.join(INPUT_DIR, dir_name, "*.jpg"))

    # detect faces on each images
    for i, image_file in enumerate(image_files):
        print(image_file)
        img = cv2.imread(image_file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        coors = detector(img_rgb, 1)
        
        # save images for each images
        for j, coors in enumerate(coors):
            top = coors.top()
            bottom = coors.bottom()
            left = coors.left()
            right = coors.right()
            
            # skip images which have the size of below 80 px
            if right - left < 80 and bottom - top:
                pass
            else:
                try:
                    dst_img = img[top:bottom, left:right]
                    #face_img = cv2.resize(dst_img, (96, 96))
                    img_name = dir_name + str(i) + '_' + str(j) + '.jpg'
                    target_dir = os.path.join(OUTPUT_DIR, dir_name,  img_name)
                    #cv2.imwrite(target_dir, face_img)
                    cv2.imwrite(target_dir, dst_img)
                except:
                    print("error")

