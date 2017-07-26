import os
import sys
import random
import shutil
import cv2
from tqdm import tqdm
sys.path.append('../')
from functions.settings import * 

# set the number of files you need par directory
min_num = 30

# set the ratio for trainding data 
ratio = 0.8

# resize param
resize = True

# image size
n_cols, n_rows, n_channels = (256, 256, 3)

dir_list = os.listdir(OUTPUT_DIR)

if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    if not os.path.exists(VALID_DIR):
        os.mkdir(VALID_DIR)

for file_dir in tqdm(dir_list):
    
    file_list = os.listdir(os.path.join(OUTPUT_DIR, file_dir))
    
    if len(file_list) <= min_num:
        continue
    else:
        if not os.path.exists(os.path.join(TRAIN_DIR, file_dir)):
            os.mkdir(os.path.join(TRAIN_DIR, file_dir))
        if not os.path.exists(os.path.join(VALID_DIR, file_dir)):
            os.mkdir(os.path.join(VALID_DIR, file_dir))

        random.shuffle(file_list)
        # choose the number of files you need
        nes_file = file_list[:min_num]

        for i in range(int(min_num * ratio)):
            
            if resize is True:
                img = cv2.imread(os.path.join(OUTPUT_DIR, file_dir, nes_file[i]))
                dst_img = cv2.resize(img, (n_cols, n_rows))
                cv2.imwrite(os.path.join(TRAIN_DIR, file_dir, str(i) + '.jpg'), dst_img)
            else:
                shutil.copy(os.path.join(OUTPUT_DIR, file_dir, nes_file[i]), os.path.join(TRAIN_DIR, file_dir, str(i) + '.jpg'))
        
        for i in range(int(min_num * ratio), min_num):

            if resize is True:
                img = cv2.imread(os.path.join(OUTPUT_DIR, file_dir, nes_file[i]))
                dst_img = cv2.resize(img, (n_cols, n_rows))
                cv2.imwrite(os.path.join(VALID_DIR, file_dir, str(i) + '.jpg'), dst_img)
            else:
                shutil.copy(os.path.join(OUTPUT_DIR, file_dir, nes_file[i]), os.path.join(VALID_DIR, file_dir, str(i) + '.jpg'))
