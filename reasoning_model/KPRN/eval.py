import pickle
import torch
import argparse
import random
import mmap
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from os import mkdir
import pandas as pd
import numpy as np
import constants.consts as consts
from model import KPRN, train, predict
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def hit_at_k(ranked_tuples, k):
    '''
    Checks if the pos interaction occured in the top k scores
    '''
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            return 1
    return 0

def ndcg_at_k(ranked_tuples, k):
    '''
    Article on ndcg: http://ethen8181.github.io/machine-learning/recsys/2_implicit.html
    ndcg_k = DCG_k / IDCG_k
    Say i represents index of or tag=1 in the top k, then since only one contribution to summation
    DCG_k = rel_i / log(i+1)
    IDCG_k = rel_i / log(1+1) since the first item is best rank
    ndcg_k = log(2) / log(i+1)
    Note we use log(2) / log(i+2) since indexing from 0
    '''
    for i,(score, tag) in enumerate(ranked_tuples[:k]):
        if tag == 1:
            return math.log(2) / math.log(i + 2)
    return 0


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def main():
    '''
    Evaluation function for kprn model testing and training
    '''
    print("Evaluation Starting")
    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                4100,100,4020, consts.TAG_SIZE,False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is", device)
    model_path='model/kprn.pth'
    
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model = model.to(device)

    hit_at_k_scores = defaultdict(list)
    ndcg_at_k_scores = defaultdict(list)
    max_k = 15
    acc=0
    file_path = 'path_data/test_interactions.txt'  
    with open('path_data/label.pickle','rb') as handle:
        label=pickle.load(handle)
    with open(file_path, 'r') as file:
        prediction_scores = predict(model, file_path,50, device,no_rel=False,gamma=1)
        target_scores=label[1800:]    
        #merge prediction scores and target scores into tuples, and rank
        # merged = list(zip(prediction_scores, target_scores))
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.metrics import accuracy_score

        # merged = list(zip(prediction_scores, target_scores))

        # X = [[x] for x, y in merged]
        # y = [y for x, y in merged]

        # clf = LogisticRegression(random_state=0).fit(X, y)
        # y_pred = clf.predict(X)

        # accuracy = accuracy_score(y, y_pred)
        # print(accuracy)
        # s_merged = sorted(merged, key=lambda x: x[0], reverse=True)
        pred_list=[]
        correct=0
        for i,score in enumerate(prediction_scores):
            if score>0.28:
                flag=1
            else:
                flag=0
            pred_list.append(flag)
            if  flag==target_scores[i]:
                correct=correct+1
        accuracy=correct/len(target_scores)
        print("Accuracy: ", accuracy)
        label_list = [float(tensor_item) for tensor_item in target_scores]
        predict_s_list = [float(tensor_item) for tensor_item in prediction_scores]
        predict_f=open('predict.txt','a+')
        cm=confusion_matrix(pred_list,target_scores)
        print(cm)
        print(classification_report(pred_list,target_scores,digits=4))
        predict_f.write(f"predict_list:{predict_s_list}\nlabel_list:{label_list}\n")
        # # saving scores
        # scores_cols = ['model', 'test_file', 'k', 'hit', 'ndcg']
        # scores_df = pd.DataFrame(scores, columns = scores_cols)
        # scores_path = 'model_scores.csv'
        # try:
        #     model_scores = pd.read_csv(scores_path)
        # except FileNotFoundError:
        #     model_scores = pd.DataFrame(columns=scores_cols)
        # model_scores=model_scores.append(scores_df, ignore_index = True, sort=False)
        # model_scores.to_csv(scores_path,index=False)


if __name__ == "__main__":
    main()
