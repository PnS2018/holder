"""Function to define our training Dataset"""
"""load_train_set and load_valid_set"""
"""output train: (train_x,train_y,Feature_number, shape) """
"""output valid: (valid_x,valid_y) """
"""shape = (width, height,numberchannel)"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

#
#ALL variables are here except the ones for declaring model Size
#and the threshold for no object.
#The variables for the Model are for now inside MODEL_TRAINER


Feature_number = 10
PicturesPFeature_train = 150
PicturesPFeature_test = 30
color = 0
resize_x = 0.5
resize_y = 0.5
shape = (int(480*resize_x), int(640*resize_y), 1) #1 because greyscale
#train options
batch_size = 20
epochs = 20
rot_range = 20
width_range = 0.2
height_range = 0.2
h_flip = True
v_flip = True

def get_num_of_classes():
    return Feature_number
def get_resize():
    output = (resize_x, resize_y)
    return output


def train_options():
    output = (batch_size, epochs, rot_range, width_range, height_range, h_flip,v_flip)
    return output

def load_train_set():
    train_x = np.zeros((Feature_number*PicturesPFeature_train, int(resize_x*480), int(resize_y*640)), dtype=np.uint8)
    train_y = np.zeros((Feature_number*PicturesPFeature_train, 1))
    for i in range (0,Feature_number):
        for k in range (0,PicturesPFeature_train):
            if i == 0:
                feature = 'Pi_Pictures/Train/Balls/0_picture'
            elif i == 1:
                feature = 'Pi_Pictures/Train/Bottles/1_picture'
            elif i == 2:
                feature = 'Pi_Pictures/Train/Cans/2_picture'
            elif i == 3:
                feature = 'Pi_Pictures/Train/Cups/3_picture'
            elif i == 4:
                feature = 'Pi_Pictures/Train/Face/4_picture'
            elif i == 5:
                feature = 'Pi_Pictures/Train/Pens/5_picture'
            elif i == 6:
                feature = 'Pi_Pictures/Train/Phone/6_picture'
            elif i == 7:
                feature = 'Pi_Pictures/Train/Shoes/7_picture'
            elif i == 8:
                feature = 'Pi_Pictures/Train/Silverware/8_picture'
            elif i == 9:
                feature = 'Pi_Pictures/Train/Yoghurt/9_picture'
            train_y[i*PicturesPFeature_train+k] = int(i)
            string = feature + str(k) + '.png'
            train_x[i*PicturesPFeature_train+k] = cv2.resize(cv2.imread(string, 0), (0,0), fx=resize_x, fy=resize_y)

    output = (train_x, train_y, Feature_number, shape)
    return output

def load_valid_set():
    valid_x = np.zeros((Feature_number*PicturesPFeature_test, int(resize_x*480), int(resize_y*640)), dtype=np.uint8)
    valid_y = np.zeros((Feature_number*PicturesPFeature_test, 1))
    for i in range (0,Feature_number):
        for k in range (0,PicturesPFeature_test):
            if i == 0:
                feature = 'Pi_Pictures/Test/Balls/0_picture'
            elif i == 1:
                feature = 'Pi_Pictures/Test/Bottles/1_picture'
            elif i == 2:
                feature = 'Pi_Pictures/Test/Cans/2_picture'
            elif i == 3:
                feature = 'Pi_Pictures/Test/Cups/3_picture'
            elif i == 4:
                feature = 'Pi_Pictures/Test/Face/4_picture'
            elif i == 5:
                feature = 'Pi_Pictures/Test/Pens/5_picture'
            elif i == 6:
                feature = 'Pi_Pictures/Test/Phone/6_picture'
            elif i == 7:
                feature = 'Pi_Pictures/Test/Shoes/7_picture'
            elif i == 8:
                feature = 'Pi_Pictures/Test/Silverware/8_picture'
            elif i == 9:
                feature = 'Pi_Pictures/Test/Yoghurt/9_picture'
            valid_y[i*PicturesPFeature_test+k] = int(i)
            string = feature + str(k) + '.png'
            valid_x[i*PicturesPFeature_test+k] = cv2.resize(cv2.imread(string, 0), (0,0), fx=resize_x, fy=resize_y)


    output = (valid_x, valid_y)
    return output

