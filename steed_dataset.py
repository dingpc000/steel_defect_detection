from torch.utils.data import Dataset
import torch
import os
from cv2 import imread,imwrite
import numpy as np
import torchvision
class steel_dataset(Dataset):
    def __init__(self,train = True,train_path='D://dataset/severstal-steel-defect-detection/train_images/',test_path='D://dataset/severstal-steel-defect-detection/test_images/',mask_path= 'D://dataset/severstal-steel-defect-detection/mask/'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_list = os.listdir(train_path)
        self.test_list = os.listdir(test_path)
        self.mask_path = mask_path
        self.train = train

    def __getitem__(self, index):
        if self.train:
            filename  = self.train_list[index].split('/')[-1]
            img, mask_img=imread(self.train_path+filename),imread(self.mask_path+filename)
            if mask_img is None:
                mask_img=np.zeros((256,1600,1), np.uint8)
                # imwrite(self.mask_path+filename,mask_img)
            #print(img,mask_img)
            #print(img.shape,mask_img.shape)
            img ,mask_img=torchvision.transforms.functional.to_tensor(img),torchvision.transforms.functional.to_tensor(mask_img)
            return img ,mask_img
        else:
            img = imread(self.test_list[index])
            img = torchvision.transforms.functional.to_tensor(img)
            return img
    def __len__(self):
        if self.train:
            return len(self.train_list)

        else :
            return len(self.test_list)

if __name__=='__main__':
    print('kk')
    steel = steel_dataset()
    dataloader = torch.utils.data.DataLoader(dataset=steel,batch_size = 1)
    print('k')
    for data in dataloader:
        pass