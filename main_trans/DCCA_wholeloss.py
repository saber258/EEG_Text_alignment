# -*- coding: utf-8 -*-
import time
import sklearn
from sklearn.metrics import confusion_matrix
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import numpy as np
import pandas as pd
from model_new import Transformer, Transformer2
from optim_new import ScheduledOptim
from dataset_new import EEGDataset, TextDataset, Fusion
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
from CCA import cca_loss, DeepCCA

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

FL = FocalLoss(class_num=2, gamma=1.5, average=False)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



def cal_loss(pred1, label1, pred2, device):

    cnt_per_class = np.zeros(2)

    loss = model.loss
    loss = loss(pred1, pred2)

    CEloss1 = F.cross_entropy(pred1, label1, reduction='sum')
    CEloss2 = F.cross_entropy(pred2, label1, reduction = 'sum')

    loss = CEloss1 + CEloss2 + loss

    pred1 = pred1.max(1)[1]
    pred2 = pred2.max(1)[1]
    n_correct1 = pred1.eq(label1).sum().item()
    n_correct2 = pred2.eq(label1).sum().item()
    n_correct = n_correct1 + n_correct2
    # cnt_per_class = [cnt_per_class[j] + pred.eq(j).sum().item() for j in range(class_num)]
    return loss, n_correct#, cnt_per_class


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


def train_epoch(train_loader1, train_loader2, device, model, optimizer, total_num, total_num2):
    all_labels = []
    all_res = []
    all_res2 = []
    model.train()
    total_loss = 0
    total_correct = 0
    #cnt_per_class = np.zeros(class_num)

  
    dataloader_iterator = iter(train_loader1)
  
    for i, data1 in enumerate(train_loader2):

      try:
          data2 = next(dataloader_iterator)
      except StopIteration:
          dataloader_iterator = iter(train_loader1)
          data2 = next(dataloader_iterator)

      sig1, label1 = map(lambda x: x.to(device), data2)
      sig2, label2 = map(lambda x: x.to(device), data1)

      
      optimizer.zero_grad()
      pred1, pred2 = model(sig1, sig2)
      all_labels.extend(label1.cpu().numpy())
      all_res.extend(pred1.max(1)[1].cpu().numpy())
      all_res2.extend(pred2.max(1)[1].cpu().numpy())
      loss, n_correct = cal_loss(pred1, label1, pred2, device)
      
      loss.backward()
      optimizer.step_and_update_lr()
      total_loss += loss.item()
      total_correct += n_correct
    
      cm = confusion_matrix(all_labels, all_res)
      cm2 = confusion_matrix(all_labels, all_res2)
      

    train_loss = total_loss / (total_num + total_num2)
    train_acc = total_correct / (total_num + total_num2)

    return train_loss, train_acc, cm, cm2 


def eval_epoch(valid_loader1, valid_loader2, device, model, total_num, total_num2):
    model.eval()

    all_labels = []
    all_res = []
    all_res2 = []
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)

    with torch.no_grad():
     
        dataloader_iterator = iter(valid_loader1)
  
        for i, data1 in enumerate(valid_loader2):

          try:
            data2 = next(dataloader_iterator)
          except StopIteration:
            dataloader_iterator = iter(valid_loader1)
            data2 = next(dataloader_iterator)
      
          sig1, label1 = map(lambda x: x.to(device), data2)
          sig2, label2 = map(lambda x: x.to(device), data1)
        
          pred1, pred2 = model(sig1, sig2)
          all_labels.extend(label1.cpu().numpy())
          all_res.extend(pred1.max(1)[1].cpu().numpy())
          all_res2.extend(pred2.max(1)[1].cpu().numpy())
          loss, n_correct = cal_loss(pred1, label1, pred2, device)

   
          total_loss += loss.item()
          total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    cm2 = confusion_matrix(all_labels, all_res2)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    print()
    acc_SP2, pre_i2, rec_i2, F1_i2 = cal_statistic(cm2)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP2))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i2))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i2))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i2))
    valid_loss = total_loss / (total_num + total_num2)
    valid_acc = total_correct / (total_num + total_num2)
    return valid_loss, valid_acc, cm, cm2, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4


def test_epoch(valid_loader, valid_loader2, device, model, total_num, total_num2):
    all_labels = []
    all_res = []
    all_pres = []
    all_pred = []
    all_pred2 = []
    all_res2 = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
      
        dataloader_iterator = iter(valid_loader)
  
        for i, data1 in enumerate(valid_loader2):

          try:
            data2 = next(dataloader_iterator)
          except StopIteration:
            dataloader_iterator = iter(valid_loader)
            data2 = next(dataloader_iterator)

          sig1, label1 = map(lambda x: x.to(device), data2)
          sig2, label2 = map(lambda x: x.to(device), data1)
          pred1, pred2 = model(sig1, sig2)  
          all_labels.extend(label1.cpu().numpy())
          all_res.extend(pred1.max(1)[1].cpu().numpy())
          all_res2.extend(pred2.max(1)[1].cpu().numpy())
          all_pred.extend(pred1.cpu().numpy())
          all_pred2.extend(pred2.cpu().numpy())
          loss, n_correct = cal_loss(pred1, label1, pred2, device)
  

          total_loss += loss.item()
          total_correct += n_correct

    np.savetxt(f'{emotion}_{model_name_base}_all_pred.txt',all_pred)
    np.savetxt(f'{emotion}_{model_name_base}_all_pred2.txt',all_pred2)
    np.savetxt(f'{emotion}_{model_name_base}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    plot_roc(all_labels,all_pred)
    all_pred2 = np.array(all_pred2)
    plot_roc(all_labels,all_pred2)
    cm = confusion_matrix(all_labels, all_res)
    cm2 = confusion_matrix(all_labels, all_res2)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    print()
    acc_SP2, pre_i2, rec_i2, F1_i2 = cal_statistic(cm2)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP2))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i2))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i2))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i2))
    test_acc = total_correct / (total_num + total_num2)
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))
    total_loss = total_loss / (total_num + total_num2)
    print(f'Test loss: {total_loss}')


if __name__ == '__main__':
    model_name_base = 'baseline_DCCA_CE'
    model_name = f'{emotion}_baseline_DCCA_CE.chkpt'
    
    # --- Preprocess
    df = pd.read_csv('df.csv')

    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.3, shuffle = True)
    ros = RandomOverSampler(random_state=2)
    X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True)
    df_test = pd.concat([X_test, y_test], axis = 1)
    df_train = pd.concat([X_resampled_text, y_resampled_text], axis = 1)
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

        # --- Text
        train_text = TextDataset(
            texts = df_train_text[:,1:],
            labels = df_train_text[:,0],
            tokenizer = tokenizer,
            max_len = MAX_LEN
        )
        val_text = TextDataset(
            texts = df_val_text[:, 1:],
            labels = df_val_text[:, 0],
            tokenizer = tokenizer,
            max_len = MAX_LEN
        )

        test_text = TextDataset(
          texts = df_test_text[:, 1:],
          labels = df_test_text[:, 0],
          tokenizer = tokenizer,
          max_len = MAX_LEN

        )
        train_loader_text = DataLoader(dataset=train_text,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  #shuffle=True)
        valid_loader_text = DataLoader(dataset=val_text,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  #shuffle=True)
        test_loader_text = DataLoader(dataset=test_text,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  #shuffle=True)
        # --- EEG
        train_eeg = EEGDataset(
            signal = df_train_eeg[:, 1:],
            label = df_train_eeg[:, 0]
        )

        val_eeg = EEGDataset(
            signal = df_val_eeg[:, 1:],
            label = df_val_eeg[:, 0]
        )

        test_eeg = EEGDataset(
          signal = df_test_eeg[:, 1:],
          label = df_test_eeg[:, 0]
        )
        # --- Dataloader EEG
        train_loader_eeg = DataLoader(dataset=train_eeg,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  # shuffle=True )
        valid_loader_eeg = DataLoader(dataset=val_eeg,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  # shuffle=True)
        
        test_loader_eeg = DataLoader(dataset=test_eeg,
                                  batch_size=batch_size,
                                  num_workers=2)#,
                                  # shuffle=True)
                
        #print(train_eeg[0], train_text[0])
        
        model1 = Transformer(device=device, d_feature=train_text.text_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model2 = Transformer2(device=device, d_feature=train_eeg.sig_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        
        chkpt1 = torch.load(torchload, map_location = 'cuda')
        chkpt2 = torch.load(torchload2, map_location = 'cuda')

        model1.load_state_dict(chkpt1['model'])
        model2.load_state_dict(chkpt2['model'])


        # model2 = model2.to(device)
        # model1 = model1.to(device)

        model = DeepCCA(model1, model2, outdim_size, use_all_singular_values).to(device)
      

        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4, lr = 1e-5), d_model, warm_steps)
        
        train_accs = []
        valid_accs = []
        eva_indis = []
        train_losses = []
        valid_losses = []
        
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_acc, train_cm, train_cm2 = train_epoch(train_loader_text, train_loader_eeg, device, model, optimizer, train_text.__len__(), train_eeg.__len__())
      

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            start = time.time()
            valid_loss, valid_acc, valid_cm, valid_cm2, eva_indi = eval_epoch(valid_loader_text,valid_loader_eeg, device, model, val_text.__len__(), val_eeg.__len__())

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
            print('train_cm:', train_cm2)
            print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                         elapse=(time.time() - start) / 60))
            print("valid_cm:", valid_cm)
            print('valid_cm:', valid_cm2)
        
        
        print('ALL DONE')               
        time_consume = (time.time() - time_start_i)
        print('total ' + str(time_consume) + 'seconds')
        plt.plot(valid_losses)
        plt.xlabel('epoch')
        plt.ylim([-1,1])
        plt.ylabel('valid loss')
        plt.title('loss change curve')

        plt.savefig(f'{emotion}_{model_name_base}results_%s.png'%r)
        

        test_model_name = str(r) + model_name
        chkpoint = torch.load(test_model_name, map_location='cuda')
        model= DeepCCA(model1, model2, outdim_size, use_all_singular_values)
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(test_loader_text, test_loader_eeg, device, model, test_text.__len__(), test_eeg.__len__())

