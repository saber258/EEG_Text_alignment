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
from dataset_new import EEGDataset, TextDataset, Text_EEGDataset
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

FL = FocalLoss(class_num=2, gamma=1.5, average=False)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def cal_loss(pred, label, device):

    cnt_per_class = np.zeros(2)

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

        sig, _, label, = map(lambda x: x.to(device), batch)
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
            sig, _, label, = map(lambda x: x.to(device), batch)
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

            sig, _, label, = map(lambda x: x.to(device), batch)

            pred = model(sig)  
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt


    np.savetxt(f'baselines/eeg/{emotion}_{model_name_base}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/eeg/{emotion}_{model_name_base}_all_label.txt', all_labels)
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
    model_name_base = 'baseline_onlyeeg'
    model_name = f'{emotion}_baseline_onlyeeg.chkpt'
    
    # --- Preprocess
    df = pd.read_csv('df.csv')
    one_hot = pd.get_dummies(df[emotion])

    df = df.drop('angry_trans', axis = 1)
    df = df.join(one_hot)
    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.3, shuffle = True, stratify = y)
    ros = RandomOverSampler(random_state=2)
    X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

    

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True, stratify = y_val)
    df_test = pd.concat([X_test, y_test], axis = 1)
    df_train = pd.concat([X_resampled_text, y_resampled_text], axis = 1)
    # df_train = pd.concat([X_train, y_train], axis = 1)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = pd.concat([X_val, y_val], axis = 1)

    df_train_text = df_train[[emotion, 'new_words']]
    df_train_eeg = df_train[eeg]

    df_val_text = df_val[[emotion, 'new_words']]
    df_val_eeg = df_val[eeg]

    df_test_text = df_test[[emotion, 'new_words']]
    df_test_eeg = df_test[eeg]

    # --- Save CSV
    df_train_text.to_csv('df_train_text.csv', header = None, index = False, index_label = False)
    df_train_eeg.to_csv('df_train_eeg.csv', header = None, index = False, index_label = False)

    df_val_text.to_csv('df_val_text.csv', header = None, index = False, index_label = False)
    df_val_eeg.to_csv('df_val_eeg.csv', header = None, index = False, index_label=False)


    df_test_text.to_csv('df_test_text.csv', header = None, index = False, index_label = False)
    df_test_eeg.to_csv('df_test_eeg.csv', header = None, index = False, index_label=False)

    # --- Load CSV
    df_train_text = pd.read_csv('df_train_text.csv', header = None).values
    df_train_eeg = pd.read_csv('df_train_eeg.csv', header = None).values

    df_val_text= pd.read_csv('df_val_text.csv', header = None).values
    df_val_eeg = pd.read_csv('df_val_eeg.csv', header = None).values

    df_test_text= pd.read_csv('df_test_text.csv', header = None).values
    df_test_eeg = pd.read_csv('df_test_eeg.csv', header = None).values


    for r in range(1):
        time_start_i = time.time()


        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # --- Text and EEG
        train_text_eeg = Text_EEGDataset(
            texts = df_train_text[:,1:],
            labels = df_train_text[:,0],
            tokenizer = tokenizer,
            max_len = MAX_LEN,
            signals = df_train_eeg[:, 1:]
        )
        val_text_eeg = Text_EEGDataset(
            texts = df_val_text[:, 1:],
            labels = df_val_text[:, 0],
            tokenizer = tokenizer,
            max_len = MAX_LEN,
            signals = df_val_eeg[:, 1:]
        )

        test_text_eeg = Text_EEGDataset(
          texts = df_test_text[:, 1:],
          labels = df_test_text[:, 0],
          tokenizer = tokenizer,
          max_len = MAX_LEN,
          signals = df_test_eeg[:, 1:]

        )
        
        # --- Sampler
        target = df_train_text[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


        # --- Loader
        train_loader_text_eeg = DataLoader(dataset=train_text_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  sampler = sampler)

        valid_loader_text_eeg = DataLoader(dataset=val_text_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        test_loader_text_eeg = DataLoader(dataset=test_text_eeg,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
    
        
        model = Transformer(device=device, d_feature=48, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

        model = nn.DataParallel(model)
        model = model.to(device)

        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4 ,lr = 1e-5, weight_decay = 1e-6), d_model, warm_steps)
        
        train_accs = []
        valid_accs = []
        eva_indis = []
        train_losses = []
        valid_losses = []
        
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_acc, train_cnt, train_cm = train_epoch(train_loader_text_eeg, device, model, optimizer, train_text_eeg.__len__())

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            start = time.time()
            valid_loss, valid_acc, valid_cnt, valid_cm, eva_indi = eval_epoch(valid_loader_text_eeg, device, model, val_text_eeg.__len__())

            valid_accs.append(valid_acc)
            eva_indis.append(eva_indi)
            valid_losses.append(valid_loss)

            model_state_dict = model.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'config_file': 'config',
                'epoch': epoch_i}

            if eva_indi >= max(eva_indis):
                torch.save(checkpoint, 'baselines/eeg/' + str(r)+model_name)
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
        fig1 = plt.figure('Figure 1')
        plt.plot(train_losses, label = 'train')
        plt.plot(valid_losses, label= 'valid')
        plt.xlabel('epoch')
        plt.ylim([0.0, 2])
        plt.ylabel('loss')
        plt.legend(loc ="upper right")
        plt.title('loss change curve')

        plt.savefig(f'baselines/eeg/{emotion}_{model_name_base}results_%s_loss.png'%r)

        fig2 = plt.figure('Figure 2')
        plt.plot(train_accs, label = 'train')
        plt.plot(valid_accs, label = 'valid')
        plt.xlabel('epoch')
        plt.ylim([0.0, 1])
        plt.ylabel('accuracy')
        plt.legend(loc ="upper right")
        plt.title('accuracy change curve')

        plt.savefig(f'baselines/eeg/{emotion}_{model_name_base}results_%s_acc.png'%r)
        

        test_model_name = 'baselines/eeg/' + str(r) + model_name
        model = Transformer(device=device, d_feature=48, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
                            class_num=class_num)
        model = nn.DataParallel(model)

        chkpoint = torch.load(test_model_name, map_location='cuda')
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(test_loader_text_eeg, device, model, test_text_eeg.__len__())

