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
from dataset_new import EEGDataset, TextDataset, Fusion, Text_EEGDataset
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

FL = FocalLoss(class_num=3, gamma=1.5, average=False)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



def cal_loss(pred1, label1, pred2, device):

    cnt_per_class = np.zeros(3)

    loss = model.loss
    loss = loss(pred1, pred2)

    pred1 = pred1.max(1)[1]
    pred2 = pred2.max(1)[1]
    n_correct1 = pred1.eq(label1).sum().item()
    n_correct = n_correct1
    return loss, n_correct



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


def train_epoch(train_loader1, device, model, optimizer, total_num, total_num2):
    model.train()
    all_labels_train = []
    all_res = []
    all_res2 = []
    all_pred_train = []
    all_pred2_train =[]
    total_loss = 0
    total_correct = 0
    #cnt_per_class = np.zeros(class_num)

  
    for batch in tqdm(train_loader1, mininterval=100, desc='- (Training)  ', leave=False): 

      sig2, sig1, label1, = map(lambda x: x.to(device), batch)

      
      optimizer.zero_grad()
      pred1, pred2 = model(sig1, sig2)
      all_labels_train.extend(label1.cpu().numpy())
      all_res.extend(pred1.max(1)[1].cpu().numpy())
      all_res2.extend(pred2.max(1)[1].cpu().numpy())
      all_pred_train.extend(pred1.detach().cpu().numpy())
      all_pred2_train.extend(pred2.detach().cpu().numpy())
      
      loss, n_correct1 = cal_loss(pred1, label1, pred2, device)
      
      loss.backward()
      optimizer.step_and_update_lr()
      total_loss += loss.item()
      total_correct += (n_correct1)
      
    train_loss = total_loss / total_num 

    return train_loss, all_pred_train, all_pred2_train, all_labels_train


def eval_epoch(valid_loader1, device, model, total_num, total_num2):
    model.eval()

    all_labels_val = []
    all_res = []
    all_res2=[]
    all_pred_val = []
    all_pred2_val = []
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
     
      for batch in tqdm(valid_loader1, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        pred1, pred2 = model(sig1, sig2)
        all_labels_val.extend(label1.cpu().numpy())
        all_res.extend(pred1.max(1)[1].cpu().numpy())
        all_res2.extend(pred2.max(1)[1].cpu().numpy())
        all_pred_val.extend(pred1.detach().cpu().numpy())
        all_pred2_val.extend(pred2.detach().cpu().numpy())
        loss, n_correct1 = cal_loss(pred1, label1, pred2, device)

  
        total_loss += loss.item()
        total_correct += (n_correct1)
    valid_loss = total_loss / total_num
    return valid_loss, all_pred_val, all_pred2_val, all_labels_val

def test_epoch(valid_loader, device, model, total_num, total_num2):
    all_labels = []
    all_labels2 = []
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
      
      for batch in tqdm(valid_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        pred1, pred2 = model(sig1, sig2)  
        all_labels.extend(label1.cpu().numpy())
        all_res.extend(pred1.max(1)[1].cpu().numpy())
        all_res2.extend(pred2.max(1)[1].cpu().numpy())
        all_pred.extend(pred1.cpu().numpy())
        all_pred2.extend(pred2.cpu().numpy())
        loss, n_correct1 = cal_loss(pred1, label1, pred2, device)


        total_loss += loss.item()
        total_correct += (n_correct1)

    np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred_test.txt',all_pred)
    np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred2_test.txt',all_pred2)
    np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_label_test.txt', all_labels)
    all_pred = np.array(all_pred)
    all_pred2 = np.array(all_pred2)
    total_loss = total_loss / total_num 
    print(f'Test loss: {total_loss}')


if __name__ == '__main__':
    model_name_base = 'baseline_DCCA_only'
    model_name = f'{emotion}_baseline_DCCA_only.chkpt'
    
    # --- Preprocess
    df = pd.read_csv('df.csv')

    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.5, stratify = y)
    ros = RandomOverSampler(random_state=2)
    X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, stratify = y_val)
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
        
        # --- model
        model1 = Transformer(device=device, d_feature=32, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model2 = Transformer2(device=device, d_feature=48, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        
        chkpt1 = torch.load(torchload, map_location = 'cuda')
        chkpt2 = torch.load(torchload2, map_location = 'cuda')

        model1.load_state_dict(chkpt1['model'])
        model2.load_state_dict(chkpt2['model'])

        model = DeepCCA(model1, model2, outdim_size, use_all_singular_values).to(device)

        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay = 1e-6), d_model, warm_steps)
        
        train_accs = []
        valid_accs = []
        eva_indis = []
        train_losses = []
        valid_losses = []
        all_pred_train = []
        all_pred_val= []
        all_pred2_train = []
        all_pred2_val = []
        all_labels_train = []
        all_labels_val = []

        
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, pred_train, pred2_train, labels_train = train_epoch(train_loader_text_eeg, device, model, optimizer, train_text_eeg.__len__(), train_text_eeg.__len__())
            
            all_pred_train.extend(pred_train)
            all_pred2_train.extend(pred2_train)
            all_labels_train.extend(labels_train)

            train_losses.append(train_loss)
            start = time.time()
            valid_loss, pred_val, pred2_val, labels_val = eval_epoch(valid_loader_text_eeg, device, model, val_text_eeg.__len__(), val_text_eeg.__len__())

            valid_losses.append(valid_loss)
            all_pred_val.extend(pred_val)
            all_pred2_val.extend(pred2_val)
            all_labels_val.extend(labels_val)

            model_state_dict = model.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'config_file': 'config',
                'epoch': epoch_i}


            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, 'baselines/DCCA/'+str(r)+model_name)
    
                print('    - [Info] The checkpoint file has been updated.')

        
            print('  - (Training)  loss: {loss: 8.5f} '
                      'elapse: {elapse:3.3f} min'.format(loss=train_loss, 
                                                         elapse=(time.time() - start) / 60))
            
            print('  - (Validation)  loss: {loss: 8.5f}, '
                      'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                         elapse=(time.time() - start) / 60))
            
            

        
        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred_train.txt',all_pred_train)
        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred2_train.txt',all_pred2_train)
        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_label_train.txt', all_labels_train)

        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred_val.txt',all_pred_val)
        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_pred2_val.txt',all_pred2_val)
        np.savetxt(f'baselines/DCCA/{emotion}_{model_name_base}_all_label_val.txt', all_labels_val)
        print('ALL DONE')               
        time_consume = (time.time() - time_start_i)
        print('total ' + str(time_consume) + 'seconds')
        fig1 = plt.figure('Figure 1')
        plt.plot(train_losses, label = 'train')
        plt.plot(valid_losses, label= 'valid')
        plt.xlabel('epoch')
        plt.ylim([-1, 1])
        plt.ylabel('loss')
        plt.legend(loc ="upper right")
        plt.title('loss change curve')

        plt.savefig(f'baselines/DCCA/{emotion}_{model_name_base}results_%s_loss.png'%r)

        fig2 = plt.figure('Figure 2')
        plt.plot(train_accs, label = 'train')
        plt.plot(valid_accs, label = 'valid')
        plt.xlabel('epoch')
        plt.ylim([0.0, 1])
        plt.ylabel('accuracy')
        plt.legend(loc ="upper right")
        plt.title('accuracy change curve')

        plt.savefig(f'baselines/DCCA/{emotion}_{model_name_base}results_%s_acc.png'%r)
        
        

        test_model_name = 'baselines/DCCA/'+str(r) + model_name
        chkpoint = torch.load(test_model_name, map_location='cuda')
        model= DeepCCA(model1, model2, outdim_size, use_all_singular_values)
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(test_loader_text_eeg, device, model, test_text_eeg.__len__(), test_text_eeg.__len__())

