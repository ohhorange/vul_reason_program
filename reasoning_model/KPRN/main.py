import pickle
import torch
import argparse
import random
import mmap
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from os import mkdir

import constants.consts as consts
from model import KPRN, train, predict
from eval import hit_at_k, ndcg_at_k

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
    Main function for kprn model testing and training
    '''
    print("Main Loaded")
    random.seed(1)
    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                4100,100,4020, consts.TAG_SIZE,False)

    model_path="model/kprn.pth"
    print("Training Starting")
    model = train(model,"path_data/train_interactions.txt",batch_size=50,epochs=30,model_path=model_path,load_checkpoint=False,not_in_memory=False,lr=0.002,l2_reg=0.0001,gamma=1,no_rel=False)

if __name__ == "__main__":
    main()
