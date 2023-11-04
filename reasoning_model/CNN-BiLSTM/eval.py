from const import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,dataloader
import numpy as np
import pickle
from tqdm import tqdm
from model import myDataset,Cnn_BiLSTM,my_collate
#LOAD TEST_DATA
with open('cnn-bilstm_data.pickle','rb') as handle:
    data = pickle.load(handle)
with open('label.pickle','rb') as handle:
    label = pickle.load(handle)
test_dataset=myDataset(label[1800:],data[1800:])
test_loader=dataloader.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,collate_fn=my_collate)


model=Cnn_BiLSTM()
model.load_state_dict(torch.load('model_pth/cb_18.pth'))
model.eval()
right_number=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
print(f"设备:{device}")
acc_list=[]
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
predict_list=[]
label_list=[]
predict_s_list=[]
with torch.no_grad():
    for test_data in tqdm(test_loader):
        path_data,label=test_data
        path_data=torch.tensor(path_data).squeeze(0).to(device)
        label=torch.tensor(label).float().squeeze().to(device)
        query=torch.tensor([4001]).to(device)
        predict_score=model(path_data,query)
        if predict_score>0.2:
            predict_label=1
        else:
            predict_label=0
        if predict_label==label.item():
            right_number+=1
        predict_s_list.append(predict_score)
        predict_list.append(predict_label)
        label_list.append(label.cpu())        
    acc=right_number/len(test_loader)
    cm=confusion_matrix(predict_list,label_list)
    print(classification_report(predict_list,label_list,digits=4))
    predict_f=open('predict.txt','a+')
    label_list = [float(tensor_item) for tensor_item in label_list]
    predict_s_list = [float(tensor_item) for tensor_item in predict_s_list]
    predict_f.write(f"predict_list:{predict_s_list}\nlabel_list:{label_list}\n")
acc_list.append(acc)
print(f"Test Accuracy: {acc}")
print(cm)