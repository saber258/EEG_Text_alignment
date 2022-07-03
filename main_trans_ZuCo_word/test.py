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
from dataset_new import EEGDataset, TextDataset
from config import *
from FocalLoss import FocalLoss
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from roc_new import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

FL = FocalLoss(class_num=3, gamma=1.5, average=False)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def cal_loss(pred, label, device):

    cnt_per_class = np.zeros(3)

    loss = F.cross_entropy(pred, label, reduction='sum')
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    cnt_per_class = [cnt_per_class[j] + pred.eq(j).sum().item() for j in range(class_num)]
    return loss, n_correct, cnt_per_class


def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)

    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1:class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, list(pre_i), list(rec_i), list(F1_i)


def train_epoch(train_loader, device, model, optimizer, total_num):
    all_labels = []
    all_res = []
    model.train()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    
    
    
    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig, label, = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        pred = model(sig)
        all_labels.extend(label.cpu().numpy())
        all_res.extend(pred.max(1)[1].cpu().numpy())
        loss, n_correct, cnt = cal_loss(pred, label, device)
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_correct += n_correct
        cnt_per_class += cnt
        cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cnt_per_class, cm


def eval_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            sig, label, = map(lambda x: x.to(device), batch)
            pred = model(sig)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt
    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cnt_per_class, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4


def test_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pres = []
    all_recs = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            sig, label, = map(lambda x: x.to(device), batch)

            pred = model(sig)  
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt


    np.savetxt('all_pred.txt',all_pred)
    np.savetxt('all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    plot_roc(all_labels,all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))




if __name__ == '__main__':
    
    model_name = 'eeg_transform.chkpt'
    
    # --- Preprocess
    df = pd.read_csv('df.csv')

    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42, test_size = 0.33)
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)

    df_train_text = df_train[[emotion, 'new_words']]
    df_train_eeg = df_train[eeg]

    df_val_text = df_val[[emotion, 'new_words']]
    df_val_eeg = df_val[eeg]

    # --- Save CSV
    df_train_text.to_csv('df_train_text.csv', header = None, index = False, index_label = False)
    df_train_eeg.to_csv('df_train_eeg.csv', header = None, index = False, index_label = False)

    df_val_text.to_csv('df_val_text.csv', header = None, index = False, index_label = False)
    df_val_eeg.to_csv('df_val_eeg.csv', header = None, index = False, index_label=False)

    # --- Load CSV
    df_train_text = pd.read_csv('df_train_text.csv', header = None).values
    df_train_eeg = pd.read_csv('df_train_eeg.csv', header = None).values

    df_val_text= pd.read_csv('df_val_text.csv', header = None).values
    df_val_eeg = pd.read_csv('df_val_eeg.csv', header = None).values


    for r in range(1):
        time_start_i = time.time()


        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # --- Text
        train_text = TextDataset(
            texts = df_train_text[:, 1],
            labels = df_train_text[:, 0],
            tokenizer = tokenizer,
            max_len = MAX_LEN
        )
        val_text = TextDataset(
            texts = df_val_text[:, 1],
            labels = df_val_text[:, 0],
            tokenizer = tokenizer,
            max_len = MAX_LEN
        )
        train_loader_text = DataLoader(dataset=train_text,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        valid_loader_text = DataLoader(dataset=val_text,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        # --- EEG
        train_eeg = EEGDataset(
            raw_data = df_train_eeg
        )

        val_eeg = EEGDataset(
            raw_data = df_val_eeg
        )
        # --- Dataloader EEG
        train_loader_eeg = DataLoader(dataset=train_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        valid_loader_eeg = DataLoader(dataset=val_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)

        model = Transformer(device=device, d_feature=train_text.text_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        print(model)
