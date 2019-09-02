import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import os
from FCN32s import *
from torchsummary import summary
from torch.utils.data import DataLoader
from steed_dataset import steel_dataset
from measure import *
def train(lr=0.01,n_epoch = 80):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = steel_dataset()
    model = FCN32s()
    model.to(device)
    dataloader=DataLoader(train_set,batch_size=5,shuffle=False)
    criterior = nn.NLLLoss2d()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    for epoch in range(n_epoch):
        if epoch > 0 and epoch % 50 == 0:
            optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0
        model = model.train()
        for data in dataloader:
            img = data[0].cuda()
            label = data[1].cuda()
            output = model(img)
            pre = F.log_softmax(output, dim=1)
            loss= criterior(pre,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss
            label_pred = pre.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 5)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            epoch, train_loss / len(dataloader), train_acc / len(dataloader), train_mean_iu / len(dataloader)))
        print(epoch_str + ' lr: {}'.format(optimizer.learning_rate))

def test():
    test_set = steel_dataset(train=False)


if __name__=='__main__':
    train()