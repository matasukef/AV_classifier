import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
sys.path.append('../setup')
from settings import *

num_img = 0

n_rows, n_cols, n_channels = (96, 96, 3)

if not os.path.exists(TRAIN_DIR):
    print('{0} is not found.'.format(TRAIN_DIR))
    sys.exit(1)

mean = np.zeros((n_rows, n_cols, 3))

dir_list = os.listdir(TRAIN_DIR)

for file_dir in tqdm(dir_list):

    file_list = os.listdir(os.path.join(TRAIN_DIR, file_dir))
    
    for img in file_list:
        
        dst_img = cv2.imread(os.path.join(TRAIN_DIR, file_dir, img))
        mean += dst_img
        num_img += 1


mean_img = mean / num_img
cv2.imwrite(os.path.join(IMAGE_DIR, 'mean_face.jpg'), mean_img)
