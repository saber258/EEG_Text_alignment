# -*- coding: utf-8 -*-
import time
import sklearn
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tqdm import tqdm
import numpy as np
import pandas as pd
from model_new import Transformer
from optim_new import ScheduledOptim
from dataset_new import EEGDataset, TextDataset, Linear
from config import *
from FocalLoss import FocalLoss
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from roc_new import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler

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
    # print(total_pred)
    total_true = cm.sum(1)
    # print(total_true)

    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1: class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    print(pre_i)
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    print(rec_i)
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


    np.savetxt(f'{emotion}_{model_name_base}_all_pred.txt',all_pred)
    np.savetxt(f'{emotion}_{model_name_base}_all_label.txt', all_labels)
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
    model_name_base = 'baseline_text_DCCA'
    model_name = f'{emotion}_baseline_text_DCCA.chkpt'
    
    # --- Preprocess
    df_train1 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred_train.txt')
    df_train1_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label_train.txt')

    df_val1 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred_val.txt')
    df_val1_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label_val.txt')

    df_test1 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred_test.txt')
    df_test1_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label_test.txt')

    df_train2 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred2_train.txt')
    df_train2_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label2_train.txt')

    df_val2 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred2_val.txt')
    df_val2_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label2_val.txt')

    df_test2 = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_pred2_test.txt')
    df_test2_lbl = np.loadtxt('arousal2_trans_baseline_DCCA_only_all_label2_test.txt')

    for r in range(1):
        time_start_i = time.time()


        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # --- Text
        train_text = EEGDataset(
            signal = df_train1,
            label = df_train1_lbl
        )

        val_text = EEGDataset(
            signal = df_val1,
            label = df_val1_lbl
        )

        test_text = EEGDataset(
          signal = df_test1,
          label = df_test1_lbl
        )
        
        # --- Sampler
        target = df_train1_lbl.astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    

        # --- Loader
        train_loader_text = DataLoader(dataset=train_text,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  sampler = sampler)

        valid_loader_text = DataLoader(dataset=val_text,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        test_loader_text = DataLoader(dataset=test_text,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        # --- EEG
        train_eeg = EEGDataset(
            signal = df_train2,
            label = df_train2_lbl
        )

        val_eeg = EEGDataset(
            signal = df_val2,
            label = df_val2_lbl
        )

        test_eeg = EEGDataset(
          signal = df_test2,
          label = df_test2_lbl
        )
        # --- Dataloader EEG

        # --- Sampler

        target = df_train2_lbl.astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader_eeg = DataLoader(dataset=train_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  sampler = sampler)
      
        valid_loader_eeg = DataLoader(dataset=val_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        
        test_loader_eeg = DataLoader(dataset=test_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        
        model = Linear(device=device, d_feature=3, d_model=3,
                            class_num=class_num)

        model = nn.DataParallel(model)
        model = model.to(device)

        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4 ,lr = 1e-5), 3, warm_steps)
        
        train_accs = []
        valid_accs = []
        eva_indis = []
        train_losses = []
        valid_losses = []
        
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_acc, train_cnt, train_cm = train_epoch(train_loader_text, device, model, optimizer, train_text.__len__())

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            start = time.time()
            valid_loss, valid_acc, valid_cnt, valid_cm, eva_indi = eval_epoch(valid_loader_text, device, model, val_text.__len__())

            valid_accs.append(valid_acc)
            eva_indis.append(eva_indi)
            valid_losses.append(valid_loss)

            model_state_dict = model.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'config_file': 'config',
                'epoch': epoch_i}

            if eva_indi >= max(eva_indis):
                torch.save(checkpoint, str(r)+model_name)
                print('    - [Info] The checkpoint file has been updated.')

        
            print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                         elapse=(time.time() - start) / 60))
            print("train_cm:", train_cm)
            print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                         elapse=(time.time() - start) / 60))
            print("valid_cm:", valid_cm)
        
        
        print('ALL DONE')               
        time_consume = (time.time() - time_start_i)
        print('total ' + str(time_consume) + 'seconds')
        plt.plot(valid_losses)
        plt.xlabel('epoch')
        plt.ylim([0.0, 2])
        plt.ylabel('valid loss')
        plt.title('loss change curve')

        plt.savefig(f'{emotion}_{model_name_base}results_%s.png'%r)
        

        test_model_name = str(r) + model_name
        model = Linear(device=device, d_feature=3, d_model=3,
                            class_num=class_num)
        model = nn.DataParallel(model)

        chkpoint = torch.load(test_model_name, map_location='cuda')
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(test_loader_text, device, model, test_text.__len__())

