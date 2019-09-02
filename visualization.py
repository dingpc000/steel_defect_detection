# -*- coding: utf-8 -*-
import cv2 as cv
import os
import pandas as pd
import numpy as np
import shutil

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def visualization():
    train_path = 'D://dataset/severstal-steel-defect-detection/train_images/'
    save_path = 'D://dataset/severstal-steel-defect-detection/visualization/'
    mask_path = 'D://dataset/severstal-steel-defect-detection/mask_images/'
    csv_name = 'D://dataset/severstal-steel-defect-detection/train.csv'
    color_dict={1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
    if not os.path.exists(save_path):
        os.mkdir(save_path,0o777)
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    del_file(save_path)
    train_list = os.listdir(train_path)
    label = pd.read_csv(csv_name)
    #filename_class = label['ImageId_ClassId']
    #encodepixle = label['EncodedPixels']
    for index,row in label.iterrows():
        filename=row['ImageId_ClassId'].split('_')[0]
        class_name = int(row['ImageId_ClassId'].split('_')[1])
        encodedpixel = row['EncodedPixels']
        if type(encodedpixel)!=type('str'):
            #print(type(encodedpixel))
            continue
        #if np.isnan(encodedpixel)
        # print(type(encodedpixel))
        # print(encodedpixel)
        if os.path.exists(save_path + filename):
            img = cv.imread(save_path + filename)
        else:
            img = cv.imread(train_path + filename)

        #img = cv.imread(train_path + filename)
        #print(encodedpixel)
        encodedpixel=encodedpixel.split(' ')
        mask = np.zeros((256, 1600, 1), np.uint8)
        mask_image = np.zeros((256,1600,3),np.uint8)
        for i in range(0,len(encodedpixel),2):
            start= int(encodedpixel[i])-1
            run_length = int(encodedpixel[i+1])
            lie = int(start/256)
            hang = int(start-lie*256)
            for k in range(run_length):
                mask_image[hang,lie]=color_dict[class_name]
                mask[hang,lie]= class_name
                # img[hang,lie] = img[hang,lie]+color_dict[class_name]
                #print('hang',hang,'lie',lie,'len',run_length,'k',k)
                hang = hang+1
                if hang >=256:
                    hang = 0
                    lie =lie+1
        # cv.imshow('mask',mask)
        # cv.waitKey(0)
        img = cv.addWeighted(img,0.5,mask_image,0.5,0)
        cv.imwrite(mask_path+filename,mask)
        #cv.imwrite(save_path+'mask'+filename,mask)
        #cv.imwrite(save_path+filename,img)

if __name__=='__main__':
    visualization()