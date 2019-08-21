# -*- coding: utf-8 -*-
import cv2 as cv
import os
import pandas as pd
import numpy as np
import shutil
def visualization():
    train_path = 'data/train_images/'
    save_path = 'data/visualization/'
    csv_name = 'data/train.csv'
    color_dict={1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,255,0)}
    shutil.rmtree(save_path)
    os.mkdir(save_path,0o777)
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
        mask = np.zeros((256, 1600, 3), np.uint8)
        for i in range(0,len(encodedpixel),2):
            start= int(encodedpixel[i])-1
            run_length = int(encodedpixel[i+1])
            lie = int(start/256)
            hang = int(start-lie*256)
            for k in range(run_length):
                mask[hang,lie]=color_dict[class_name]
                # img[hang,lie] = img[hang,lie]+color_dict[class_name]
                #print('hang',hang,'lie',lie,'len',run_length,'k',k)
                hang = hang+1
                if hang >=256:
                    hang = 0
                    lie =lie+1
        # cv.imshow('mask',mask)
        # cv.waitKey(0)
        img = cv.addWeighted(img,0.5,mask,0.5,0)
        #cv.imwrite(save_path+'mask'+filename,mask)
        cv.imwrite(save_path+filename,img)

if __name__=='__main__':
    visualization()