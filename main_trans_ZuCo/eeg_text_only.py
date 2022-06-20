# -*- coding: utf-8 -*-
import time
import sklearn
from sklearn.metrics import confusion_matrix
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tqdm import tqdm
import numpy as np
import pandas as pd
from model_new import Transformer, Transformer2
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


def train_epoch(train_loader1, train_loader2, device, model, model2, optimizer, optimizer2, total_num, total_num2):
    all_labels_text = []
    all_res_text = []
    all_labels_eeg = []
    all_res_eeg = []
    model.train()
    model2.train()
    total_loss_text = 0
    total_correct_text = 0
    total_loss_eeg = 0
    total_correct_eeg = 0
    #cnt_per_class = np.zeros(class_num)

  
    dataloader_iterator = iter(train_loader1)
  
    for i, data1 in enumerate(train_loader2):

      try:
          data2 = next(dataloader_iterator)
      except StopIteration:
          dataloader_iterator = iter(train_loader1)
          data2 = next(dataloader_iterator)

      sig1, label1 = map(lambda x: x.to(device), data2)
      
      optimizer.zero_grad()
      pred1 = model(sig1)
      all_labels_text.extend(label1.cpu().numpy())
      all_res_text.extend(pred1.max(1)[1].cpu().numpy())
      loss1, n_correct1, cnt1 = cal_loss(pred1, label1, device)
      #print(cnt1)
      
      sig2, label2 = map(lambda x: x.to(device), data1)
      optimizer2.zero_grad()
      pred2 = model2(sig2)
      all_labels_eeg.extend(label2.cpu().numpy())
      all_res_eeg.extend(pred2.max(1)[1].cpu().numpy())
      loss2, n_correct2, cnt2 = cal_loss(pred2, label2, device)
      # print(cnt2)
      
    
      
      loss1.backward()
      loss2.backward()
      optimizer.step_and_update_lr()
      optimizer2.step_and_update_lr()
      total_loss_text += loss1.item()
      total_correct_text += n_correct1
      total_loss_eeg += loss2.item()
      total_correct_eeg += n_correct2
  
  
      cm_text = confusion_matrix(all_labels_text, all_res_text)
      cm_eeg = confusion_matrix(all_labels_eeg, all_res_eeg)
      

    train_loss_text = total_loss_text / total_num
    train_acc_text = total_correct_text / total_num

    train_loss_eeg = total_loss_eeg / total_num2
    train_acc_eeg = total_correct_eeg / total_num2


    return train_loss_text, train_acc_text, cm_text, train_loss_eeg, train_acc_eeg, cm_eeg


def eval_epoch(valid_loader1, valid_loader2, device, model, model2, total_num, total_num2):
    model.eval()
    model2.eval()
    all_labels_text = []
    all_res_text = []
    total_loss_text = 0
    total_correct_text = 0
    all_labels_eeg = []
    all_res_eeg = []
    total_loss_eeg = 0
    total_correct_eeg = 0

    with torch.no_grad():
     
        dataloader_iterator = iter(valid_loader1)
  
        for i, data1 in enumerate(valid_loader2):

          try:
            data2 = next(dataloader_iterator)
          except StopIteration:
            dataloader_iterator = iter(valid_loader1)
            data2 = next(dataloader_iterator)
      
          sig1, label1 = map(lambda x: x.to(device), data2)
        
          pred1 = model(sig1)
          all_labels_text.extend(label1.cpu().numpy())
          all_res_text.extend(pred1.max(1)[1].cpu().numpy())
          loss1, n_correct1, cnt1 = cal_loss(pred1, label1, device)

          sig2, label2 = map(lambda x: x.to(device), data1)
          optimizer.zero_grad()
          pred2 = model2(sig2)
          all_labels_eeg.extend(label2.cpu().numpy())
          all_res_eeg.extend(pred2.max(1)[1].cpu().numpy())
          loss2, n_correct2, cnt2 = cal_loss(pred2, label2, device)
          
          
          total_loss_text += loss1.item()
          total_correct_text += n_correct1

          total_loss_eeg += loss2.item()
          total_correct_eeg += n_correct2

    cm_text = confusion_matrix(all_labels_text, all_res_text)
    cm_eeg = confusion_matrix(all_labels_eeg, all_res_eeg)

    acc_SP_text, pre_i_text, rec_i_text, F1_i_text = cal_statistic(cm_text)
    acc_SP_eeg, pre_i_eeg, rec_i_eeg, F1_i_eeg = cal_statistic(cm_eeg)

    print('TEXT: ')
    print('acc_SP is : {acc_SP_text}'.format(acc_SP_text=acc_SP_text))
    print('pre_i is : {pre_i_text}'.format(pre_i_text=pre_i_text))
    print('rec_i is : {rec_i_text}'.format(rec_i_text=rec_i_text))
    print('F1_i is : {F1_i_text}'.format(F1_i_text=F1_i_text))
    print()
    print('EEG: ')
    print('acc_SP is : {acc_SP_eeg}'.format(acc_SP_eeg=acc_SP_eeg))
    print('pre_i is : {pre_i_eeg}'.format(pre_i_eeg=pre_i_eeg))
    print('rec_i is : {rec_i_eeg}'.format(rec_i_eeg=rec_i_eeg))
    print('F1_i is : {F1_i_eeg}'.format(F1_i_eeg=F1_i_eeg))
    valid_loss_text = total_loss_text / total_num
    valid_acc_text = total_correct_text / total_num
    valid_loss_eeg = total_loss_eeg /total_num2
    valid_acc_eeg = total_loss_eeg / total_num2
    return valid_loss_text, valid_acc_text, cm_text, sum(rec_i_text[1:]) * 0.6 + sum(pre_i_text[1:]) * 0.4, valid_loss_eeg, valid_acc_eeg, cm_text, sum(rec_i_eeg[1:]) * 0.6 + sum(pre_i_eeg[1:]) * 0.4


def test_epoch(valid_loader, valid_loader2, device, model, model2, total_num, total_num2):
    all_labels_text = []
    all_res_text = []
    all_pres_text = []
    all_recs_text = []
    all_pred_text = []
    all_labels_eeg = []
    all_res_eeg = []
    all_pres_eeg = []
    all_recs_eeg = []
    all_pred_eeg = []
    model.eval()
    model2.eval()
    total_loss_text = 0
    total_correct_text = 0
    total_loss_eeg = 0
    total_correct_eeg = 0
    with torch.no_grad():
      
        dataloader_iterator = iter(valid_loader)
  
        for i, data1 in enumerate(valid_loader2):

          try:
            data2 = next(dataloader_iterator)
          except StopIteration:
            dataloader_iterator = iter(valid_loader)
            data2 = next(dataloader_iterator)

          sig1, label1 = map(lambda x: x.to(device), data2)
          pred1 = model(sig1)  
          all_labels_text.extend(label1.cpu().numpy())
          all_res_text.extend(pred1.max(1)[1].cpu().numpy())
          all_pred_text.extend(pred1.cpu().numpy())
          loss1, n_correct1, cnt1 = cal_loss(pred1, label1, device)

            
          sig2, label2 = map(lambda x: x.to(device), data1)
          pred2 = model2(sig2)  
          all_labels_eeg.extend(label2.cpu().numpy())
          all_res_eeg.extend(pred2.max(1)[1].cpu().numpy())
          all_pred_eeg.extend(pred2.cpu().numpy())
          loss2, n_correct2, cnt2 = cal_loss(pred2, label2, device)

          
          total_loss_text += loss1.item()
          total_correct_text += n_correct1
          total_loss_eeg += loss2.item()
          total_correct_eeg += n_correct2
            #cnt_per_class += (cnt1 + cnt2)

    np.savetxt(f'{emotion}_{model_name_base}_all_pred.txt',all_pred_text)
    np.savetxt(f'{emotion}_{model_name_base}_all_label.txt', all_labels_text)
    np.savetxt(f'{emotion}_{model_name_base2}_all_pred.txt',all_pred_eeg)
    np.savetxt(f'{emotion}_{model_name_base2}_all_label.txt', all_labels_eeg)
    all_pred_text = np.array(all_pred_text)
    plot_roc(all_labels_text,all_pred_text)
    cm_text = confusion_matrix(all_labels_text, all_res_text)
    
    all_pred_eeg = np.array(all_pred_eeg)
    plot_roc(all_labels_eeg,all_pred_eeg)
    cm_eeg = confusion_matrix(all_labels_eeg, all_res_eeg)
    print("test_cm_text:", cm_text)
    acc_SP_text, pre_i_text, rec_i_text, F1_i_text = cal_statistic(cm_text)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP_text))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i_text))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i_text))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i_text))
    test_acc_text = total_correct_text / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc_text))
    print()
    print("test_cm_eeg:", cm_eeg)
    acc_SP_eeg, pre_i_eeg, rec_i_eeg, F1_i_eeg = cal_statistic(cm_eeg)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP_eeg))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i_eeg))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i_eeg))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i_eeg))
    test_acc_eeg = total_correct_eeg / total_num2
    print('test_acc is : {test_acc}'.format(test_acc=test_acc_eeg))

if __name__ == '__main__':
    model_name_base = 'baseline_text'
    model_name_base2 = 'baseline_eeg'
    plot_name = 'baseline'
    model_name = f'{emotion}_baseline_simul_text.chkpt'
    model_name2 = f'{emotion}_baseline_simul_eeg.chkpt'
    
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
        
        # --- Sampler
        target = df_train_text[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
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

        # --- Sampler

        target = df_train_eeg[:, 0].astype('int')
        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[target]
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
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
        
        model = Transformer(device=device, d_feature=train_text.text_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model2 = Transformer2(device=device, d_feature=train_eeg.sig_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        model2 = model2.to(device)
        model = model.to(device)

        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4, lr = 1e-5), d_model, warm_steps)
        
        optimizer2 = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model2.parameters()),
                 betas=(0.9, 0.98), eps=1e-4, lr = 1e-5), d_model, warm_steps)
        
        train_accs_text = []
        valid_accs_text = []
        eva_indis_text = []
        train_losses_text = []
        valid_losses_text = []

        train_accs_eeg = []
        valid_accs_eeg = []
        eva_indis_eeg = []
        train_losses_eeg = []
        valid_losses_eeg = []
        
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss_text, train_acc_text, train_cm_text, train_loss_eeg, train_acc_eeg, train_cm_eeg = train_epoch(train_loader_text, train_loader_eeg, device, model, model2, optimizer, optimizer2, train_text.__len__(), train_eeg.__len__())
      

            train_accs_text.append(train_acc_text)
            train_losses_text.append(train_loss_text)

            train_accs_eeg.append(train_acc_eeg)
            train_losses_eeg.append(train_loss_eeg)
            start = time.time()
            valid_loss_text, valid_acc_text, valid_cm_text, eva_indi_text, valid_loss_eeg, valid_acc_eeg, valid_cm_eeg, eva_indi_eeg = eval_epoch(valid_loader_text,valid_loader_eeg, device, model, model2, val_text.__len__(), val_eeg.__len__())

            valid_accs_text.append(valid_acc_text)
            eva_indis_text.append(eva_indi_text)
            valid_losses_text.append(valid_loss_text)

            valid_accs_eeg.append(valid_acc_eeg)
            eva_indis_eeg.append(eva_indi_eeg)
            valid_losses_eeg.append(valid_loss_eeg)

            model_state_dict = model.state_dict()
            model_state_dict2 = model2.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'config_file': 'config',
                'epoch': epoch_i
                }
            checkpoint2 = {
                'model': model_state_dict2,
                'config_file': 'config',
                'epoch': epoch_i
                }

            if eva_indi_text >= max(eva_indis_text):
                torch.save(checkpoint, str(r)+model_name)
                
                print('    - [Info] The checkpoint file has been updated.')
            if eva_indi_eeg >= max(eva_indis_eeg):
              torch.save(checkpoint2, str(r)+model_name2)
              print('     - [Info] The checkpoint file has been updated.')

            print('TEXT')
            print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=train_loss_text, accu=100 * train_acc_text,
                                                         elapse=(time.time() - start) / 60))
            print("train_cm:", train_cm_text)
            print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=valid_loss_text, accu=100 * valid_acc_text,
                                                         elapse=(time.time() - start) / 60))
            print("valid_cm:", valid_cm_text)
            print()
            print('EEG')
            print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=train_loss_eeg, accu=100 * train_acc_eeg,
                                                         elapse=(time.time() - start) / 60))
            print("train_cm:", train_cm_text)
            print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                      'elapse: {elapse:3.3f} min'.format(loss=valid_loss_eeg, accu=100 * valid_acc_eeg,
                                                         elapse=(time.time() - start) / 60))
            print("valid_cm:", valid_cm_eeg)
        
        
        print('ALL DONE')               
        time_consume = (time.time() - time_start_i)
        print('total ' + str(time_consume) + 'seconds')
       
        plt.plot(valid_losses_text, label = 'Text')
        plt.plot(valid_losses_eeg, label = 'EEG')
        plt.xlabel('epoch')
        plt.ylim([0.0, 1])
        plt.ylabel('valid loss')
        plt.title('loss change curve')
        plt.legend()

        plt.savefig(f'{emotion}_{plot_name}results_%s.png'%r)

        

    
        model = Transformer(device=device, d_feature=test_text.text_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
                            class_num=class_num)
        model2 = Transformer2(device=device, d_feature=test_eeg.sig_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
                            class_num=class_num)
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        chkpoint = torch.load(str(r) + model_name, map_location='cuda')
        chkpoint2 = torch.load(str(r) + model_name2, map_location='cuda')
        model.load_state_dict(chkpoint['model'])
        model2.load_state_dict(chkpoint2['model'])
        model = model.to(device)
        model2 = model2.to(device)
        test_epoch(test_loader_text, test_loader_eeg, device, model, model2, test_text.__len__(), test_eeg.__len__())

