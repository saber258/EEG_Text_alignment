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
from dataset_new import EEGDataset, TextDataset, Fusion, Text_EEGDataset, Linear
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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
r=0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

FL = FocalLoss(class_num=3, gamma=1.5, average=False)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



def cal_loss(pred1, label1, pred2, device):

    cnt_per_class = np.zeros(3)

    loss2 = nn.CosineSimilarity()
    loss2 = loss2(pred1, pred2)
    loss2 = torch.sum(loss2)
    # print(loss)
    loss2 = -loss2

    loss1 = F.cross_entropy(pred2, label1, reduction = 'sum')
    loss = loss2 + loss1

    pred1 = pred1.max(1)[1]
    pred2 = pred2.max(1)[1]
    n_correct3 = pred1.eq(label1).sum().item()
    n_correct = n_correct3
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
    all_labels = []
    all_res = []
   
    total_loss = 0
    total_correct = 0
    #cnt_per_class = np.zeros(class_num)

  
    for batch in tqdm(train_loader1, mininterval=0.5, desc='- (Validation)  ', leave=False):

      sig2, sig1, label1, = map(lambda x: x.to(device), batch)

      
      optimizer.zero_grad()
      pred1, pred2 = model(sig1, sig2)
      all_labels.extend(label1.cpu().numpy())
      all_res.extend(pred2.max(1)[1].cpu().numpy())
    
      loss, n_correct1 = cal_loss(pred1, label1, pred2, device)
      
      
      loss.backward()
      optimizer.step_and_update_lr()
      total_loss += loss.item()
      total_correct += (n_correct1)
  
  
      cm = confusion_matrix(all_labels, all_res)
      

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num

    return train_loss, train_acc, cm


def eval_epoch(valid_loader1, device, model, total_num, total_num2):
    model.eval()

    all_labels = []
    all_res = []
    all_pred = []

    total_loss = 0
    total_correct = 0

    with torch.no_grad():
     
      for batch in tqdm(valid_loader1, mininterval=0.5, desc='- (Validation)  ', leave=False):

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)
      
        pred1, pred2 = model(sig1, sig2)
        all_labels.extend(label1.cpu().numpy())
        all_res.extend(pred2.max(1)[1].cpu().numpy())
        all_pred.extend(pred2.cpu().numpy())
        loss, n_correct1 = cal_loss(pred1, label1, pred2,device)
        

  
        total_loss += loss.item()
        total_correct += (n_correct1)

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm,sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels


def test_epoch(valid_loader, device, model, total_num, total_num2):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
     
      for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)
        pred1, pred2 = model(sig1, sig2)  
        all_labels.extend(label1.cpu().numpy())
        all_res.extend(pred2.max(1)[1].cpu().numpy())
        all_pred.extend(pred2.cpu().numpy())
        loss, n_correct1 = cal_loss(pred1, label1, pred2, device)


        total_loss += loss.item()
        total_correct += (n_correct1)

    np.savetxt(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}_all_pred.txt',all_pred)

    np.savetxt(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}_all_label.txt', all_labels)
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
    total_loss = total_loss / total_num
    print(f'Test loss: {total_loss}')


if __name__ == '__main__':
    model_name_base = 'baseline_fusion_cossim_eeg_trans'
    model_name = f'{emotion}_baseline_fusion_cossim_eeg_trans.chkpt'
    
    # --- Preprocess
    df = pd.read_csv(f'preprocessed_eeg/{patient}_mean.csv')

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
    df_train_eeg_label = df_train[[emotion]]
    df_train_eeg = df_train.iloc[:, 3:]
    df_train_eeg = pd.concat([df_train_eeg_label, df_train_eeg], axis=1)

    df_val_text = df_val[[emotion, 'new_words']]
    df_val_eeg_label = df_val[[emotion]]
    df_val_eeg = df_val.iloc[:, 3:]

    df_val_eeg = pd.concat([df_val_eeg_label, df_val_eeg], axis=1)

    df_test_text = df_test[[emotion, 'new_words']]
    df_test_eeg_label = df_test[[emotion]]
    df_test_eeg = df_test.iloc[:, 3:]
    df_test_eeg = pd.concat([df_test_eeg_label, df_test_eeg], axis=1)

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
    
    model1 = Transformer(device=device, d_feature=32, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
    model2 = Transformer2(device=device, d_feature=838, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
    # model1 = nn.DataParallel(model1)
    # model2 = nn.DataParallel(model2)
    # model1 = Linear(device, 32, class_num)
    # model2 = Linear(device, 839, class_num)
    
    
    model1 = nn.DataParallel(model1)
    model2 = nn.DataParallel(model2)
    
    # chkpt1 = torch.load(torchload, map_location = 'cuda')
    # chkpt2 = torch.load(torchload2, map_location = 'cuda')

    # model1.load_state_dict(chkpt1['model'])
    # model2.load_state_dict(chkpt2['model'])


    model2 = model2.to(device)
    model1 = model1.to(device)

    model = Fusion(model1, model2).to(device)
  

    
    optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
              betas=(0.9, 0.98), eps=1e-4, lr = 1e-4, weight_decay = 1e-3), d_model, warm_steps)
    
    train_accs = []
    valid_accs = []
    eva_indis = []
    train_losses = []
    valid_losses = []
    pred_val = []
    label_val = []
    
    for epoch_i in range(epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_acc, train_cm = train_epoch(train_loader_text_eeg, device, model, optimizer, train_text_eeg.__len__(), train_text_eeg.__len__())
  

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        start = time.time()
        valid_loss, valid_acc, valid_cm, eva_indi, all_pred_val, all_label_val = eval_epoch(valid_loader_text_eeg, device, model, val_text_eeg.__len__(), val_text_eeg.__len__())

        pred_val.extend(all_pred_val)
        label_val.extend(all_label_val)
        valid_accs.append(valid_acc)
        eva_indis.append(eva_indi)
        valid_losses.append(valid_loss)

        model_state_dict = model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'config_file': 'config',
            'epoch': epoch_i}


        if eva_indi >= max(eva_indis):
            torch.save(checkpoint, 'baselines/fusion_cossim_ds/'+str(r)+model_name)

            print('    - [Info] The checkpoint file has been updated.')

    
        print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                      elapse=(time.time() - start) / 60))
        print("train_cm:", train_cm)
        
        print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                      elapse=(time.time() - start) / 60))
        print("valid_cm:", valid_cm)
        writer.add_scalar('Accuracy', train_acc, epoch_i)
        writer.add_scalar('Loss', train_loss, epoch_i)
    
    
    np.savetxt(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}_all_pred_val.txt',pred_val)
    np.savetxt(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}_all_label_val.txt', label_val)
    print('ALL DONE')               
    time_consume = (time.time() - time_start_i)
    print('total ' + str(time_consume) + 'seconds')
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(valid_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0, 2])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')

    plt.savefig(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}results_%s_loss.png'%r)

    fig2 = plt.figure('Figure 2')
    plt.plot(train_accs, label = 'train')
    plt.plot(valid_accs, label = 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 1])
    plt.ylabel('accuracy')
    plt.legend(loc ="upper right")
    plt.title('accuracy change curve')

    plt.savefig(f'baselines/fusion_cossim_ds/{emotion}_{model_name_base}results_%s_acc.png'%r)
    

    test_model_name = 'baselines/fusion_cossim_ds/'+str(r) + model_name
    chkpoint = torch.load(test_model_name, map_location='cuda')
    model = Fusion(model1, model2).to(device)
    model.load_state_dict(chkpoint['model'])
    model = model.to(device)
    test_epoch(test_loader_text_eeg, device, model, test_text_eeg.__len__(), test_text_eeg.__len__())
