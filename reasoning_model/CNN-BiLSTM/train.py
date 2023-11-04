from const import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,dataloader
import numpy as np
import pickle
from tqdm import tqdm
from model import Cnn_BiLSTM,SentEncoder,myDataset,my_collate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#LOAD TRAIN_DATA
print("LOAD TRAIN_DATA....")
print("========导入训练数据============")
with open('cnn-bilstm_data.pickle','rb') as handle:
    data = pickle.load(handle)
with open('label.pickle','rb') as handle:
    label = pickle.load(handle)

train_dataset=myDataset(label[:1800],data[:1800])
train_loader=dataloader.DataLoader(dataset=train_dataset,batch_size=1,shuffle=False,collate_fn=my_collate)
test_dataset=myDataset(label[1800:],data[1800:])
test_loader=dataloader.DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,collate_fn=my_collate)


model=Cnn_BiLSTM()
# model.load_state_dict(torch.load('model_pth/cb_7.pth'))
bce=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{device}")
model=model.to(device)
loss_list=[]
acc_list=[]
preds=[]
label_list=[]
print("训练过程:")
for epoch in range(NUM_EPOCHS):
    running_loss=0
    model=model.train()
    for data in tqdm(train_loader):
        path_data,label=data
        path_data=torch.tensor(path_data).squeeze(0).to(device)
        label=torch.tensor(label).float().squeeze().to(device)
        label_list.append(label)
        query=torch.tensor([4001]).to(device)
        optimizer.zero_grad()
        predict_score=model(path_data,query)
        preds.append(predict_score)
        loss=bce(predict_score,label)
        loss.backward()
        optimizer.step()
        running_loss=running_loss+loss.item()
    running_loss=running_loss/len(train_loader)
    loss_list.append(running_loss)
    loss_f=open('loss.txt','a')
    print(f"epoch:{epoch+1}   loss:{running_loss}")
    loss_f.write(f"epoch:{epoch+1}   loss:{running_loss}\n")
    loss_f.close()
    torch.save(model.state_dict(),f"model_pth/cb_{epoch+1}.pth")
    model.eval()
    right_number=0
    acc_list=[]
    predict_list=[]
    label_list=[]
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            path_data,label=test_data
            path_data=torch.tensor(path_data).squeeze(0).to(device)
            label=torch.tensor(label).float().squeeze().to(device)
            query=torch.tensor([4001]).to(device)
            predict_score=model(path_data,query)
            if predict_score>0.12:
                predict_label=1
            else:
                predict_label=0
            if predict_label==label.item():
                right_number+=1
            predict_list.append(predict_label)
            label_list.append(label.cpu())        
        acc=right_number/len(test_loader)
        cm=confusion_matrix(predict_list,label_list)
        print(classification_report(predict_list,label_list,digits=4))
    acc_list.append(acc)
    print(f"Test Accuracy: {acc}")
    print(cm)