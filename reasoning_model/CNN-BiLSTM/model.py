
from const import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,dataloader
import numpy as np
import pickle
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class myDataset(Dataset):
    def __init__(self,label_list,path_list):
        self.label=label_list
        self.path_data=path_list
    def __getitem__(self,idx):
        label=self.label[idx]
        path_data=self.path_data[idx]
        return path_data,label
    def __len__(self):
        return len(self.path_data)
    
def my_collate(batch):
    data=[item[0] for item in batch]
    target=[item[1] for item in batch]
    return [data,target]


class SentEncoder(nn.Module):
    def __init__(self):
        super(SentEncoder,self).__init__()
        self.embedding_A = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.conv1=nn.Conv1d(EMBEDDING_DIM,NUM_FILTERS,KERNEL_SIZE)
        self.pooling=nn.MaxPool1d(POOLING_SIZE,1)
        self.bilstm=nn.LSTM(input_size=int(EMBEDDING_DIM/2),hidden_size=LSTM_HIDDEN_UNITS, bidirectional=True,)
    def forward(self,x):
        x=self.embedding_A(x)
        x=x.unsqueeze(0)
        x=x.permute(0,2,1)
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pooling(x)
        x=x.permute(0,2,1).squeeze(0)
        _,(h_n,c_n)=self.bilstm(x)
        x=torch.cat([h_n[0],h_n[1]],dim=0)
        return x

class Cnn_BiLSTM(nn.Module):
    def __init__(self):
        super(Cnn_BiLSTM,self).__init__()
        self.dense1=nn.Linear(EMBEDDING_DIM*2,EMBEDDING_DIM)
        self.tanh=nn.Tanh()
        self.dense2=nn.Linear(EMBEDDING_DIM,1)
        self.softmax=nn.Softmax(dim=0)
        self.relu=nn.ReLU()
        self.sig=nn.Sigmoid()
        self.sentEncoder=SentEncoder().to(device)
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    def forward(self,path_data,query):
        context_encoded=torch.zeros((path_data.shape[0],EMBEDDING_DIM)).to(device)#每一个tensor都表示一个关系和实体
        for i,path in enumerate(path_data):        
            path_encoded=self.sentEncoder(path)
            context_encoded[i]=path_encoded
        query_encoded=self.embedding(query)
        u=query_encoded
        for k in range(2):#两层注意力机制
            u_rep=u.repeat(context_encoded.shape[0],1)#每一行都一样(2,100)
            output=torch.cat((context_encoded,u_rep),dim=1)#(2,200)
            tanh=self.tanh(self.dense1(output))
            score=self.dense2(tanh)
            score=score.view(-1)
            alpha=self.softmax(score)
            alpha=alpha.unsqueeze(1)
            o=torch.sum(alpha*context_encoded,dim=0)
            u=u.squeeze(0)
            u=torch.cat((u,o),dim=0)
            u=self.dense1(u)
        u=self.relu(u)
        u=self.dense2(u)
        prediction=self.sig(u)
        return prediction.squeeze()