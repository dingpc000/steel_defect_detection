import torch.nn as nn
import torch
import torchvision
import os
from network import *
from torchsummary import summary
def train():
    train_img_path = 'data/train_images/'
    train_list = os.listdir(train_img_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg =torchvision.models.vgg16()


def test():
    test_img_path = 'data/test_images/'
    test_list = os.listdir(test_img_path)


if __name__=='__main__':
    train()
    test()