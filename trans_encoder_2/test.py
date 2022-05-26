import time
import sklearn
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import numpy as np
import pandas as pd
from model_new import Transformer
from optim_new import ScheduledOptim
from dataset_new import SignalDataset
from config import *
from FocalLoss import FocalLoss
#from entropy import *
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from roc_new import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os




if __name__ == '__main__':
    
    model_name = 'transform.chkpt'
    train_file = 'train_all.csv'
    test_file = 'test_all.csv'
    raw_train = pd.read_csv(train_file, header=None).values
    raw_test = pd.read_csv(test_file, header=None).values
    whole_data = raw_train 
    time_start_i = time.time()
    raw_train, raw_valid, _, _ = train_test_split(whole_data, list(whole_data[:, 0]), test_size=0.3,
                                                          random_state=42,stratify=list(whole_data[:, 0]))
    print("raw_train shape", raw_train.shape)
    print("raw_valid shape", raw_valid.shape)
    _, raw_test, _, _ = train_test_split(raw_test, list(raw_test[:, 0]), test_size=0.99,
                                                     random_state=42,stratify=list(raw_test[:, 0]))


    print("raw_valid shape", raw_valid.shape)
    print("raw_test shape", raw_test.shape)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = SignalDataset(raw_train)
    valid_data = SignalDataset(raw_valid)
    test_data = SignalDataset(raw_test)
    train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=4,
                                  shuffle=True)
        #print('train_loader:', train_loader)
    valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  num_workers=4,
                                  shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 shuffle=True)

    for batch in tqdm(train_loader):
        print(batch[0].shape)
        break
