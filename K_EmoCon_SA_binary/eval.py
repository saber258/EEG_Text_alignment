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
from model_new import *
from optim_new import ScheduledOptim
from dataset_new import Text_EEGDataset
from config import *
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from roc_new import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os
from transformers import BertTokenizer, BertModel
from imblearn.over_sampling import RandomOverSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def cal_loss(pred, pred2, label, device):


    loss = F.cross_entropy(pred2, label, reduction='sum')
    pred = pred2.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    return loss, n_correct


def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)

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

    return acc_SP, np.average(pre_i), np.average(rec_i), np.average(F1_i)


def test_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Test)  ', leave=False):

            sig2, sig1, label, = map(lambda x: x.to(device), batch)

            pred, pred2 = model(sig1, sig2) 
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred2.max(1)[1].cpu().numpy())
            all_pred.extend(pred2.cpu().numpy())
            loss, n_correct = cal_loss(pred, pred2, label, device)

            total_loss += loss.item()
            total_correct += n_correct

    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = (total_correct / total_num) 
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))

def get_embeddings(df, device):
  words = df
  
  marked_texts = []
  
  for i in words:
    marked_text = "[CLS] " + i + " [SEP]"
    marked_texts.append(marked_text)

  tokenized = []
  for i in marked_texts:
    tokenized_text = tokenizer.tokenize(i)
    tokenized.append(tokenized_text)

  index_token = []

  for i in tokenized:
    index_token.append(tokenizer.convert_tokens_to_ids(i))
  
  segments = []

  for i in tokenized:
    segments.append([1] * len(i))

  tokens_tensors = []
  for i in index_token:
    tokens_tensor = torch.tensor([i])
    tokens_tensors.append(tokens_tensor)

  segment_tensors = []
  for i in segments:
    segments_tensors = torch.tensor([i])
    segment_tensors.append(segments_tensors)

  model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, 
                                  ).to(device)

  model.eval()

  output = []
  hidden_state = []
  for i in range(len(tokens_tensors)):

    with torch.no_grad():

      outputs = model(tokens_tensors[i], segment_tensors[i])
      output.append(outputs)

      hidden_states = outputs[2]
      hidden_state.append(hidden_states)

  embeddings = []
  for i in range(len(hidden_state)):
    token_vecs = hidden_state[i][-2][0]
    embedding = torch.mean(token_vecs, dim=0)
    embeddings.append(embedding)

  return embeddings

if __name__ == '__main__':
    
    # --- Preprocess
    df = pd.read_csv('df.csv')

    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 2, test_size = 0.3, shuffle = True, stratify = y)
    ros = RandomOverSampler(random_state=2)
    X_resampled_text, y_resampled_text = ros.fit_resample(X_train, y_train)

    

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state= 2, test_size = 0.5, shuffle = True, stratify = y_val)
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

        embeddings_train = get_embeddings(df_train_text[:,1])
        embeddings_val = get_embeddings(df_val_text[:,1])
        embeddings_test = get_embeddings(df_test_text[:,1])
        

        # --- Text and EEG
        train_text_eeg = Text_EEGDataset(
            texts = embeddings_train,
            labels = df_train_text[:,0],
            tokenizer = tokenizer,
            max_len = MAX_LEN,
            signals = df_train_eeg[:, 1:]
        )
        val_text_eeg = Text_EEGDataset(
            texts = embeddings_val,
            labels = df_val_text[:, 0],
            tokenizer = tokenizer,
            max_len = MAX_LEN,
            signals = df_val_eeg[:, 1:]
        )

        test_text_eeg = Text_EEGDataset(
        texts = embeddings_test,
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
        model2 = Transformer2(device=device, d_feature=48, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
    

        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        
        model1 = model1.to(device)
        model2 = model2.to(device)

        # --- Choose a model to evaluate
        
        #model = DeepCCA(model1, model2, outdim_size, use_all_singular_values).to(device)

        # model = CAM(model1, model2).to(device)

        # model = Fusion(device=device, model1 = model1, model2 = model2,
        # d_feature =6, d_model=d_model, d_inner=d_inner,
        # n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num).to(device)

        # model = DeepCCA_fusion(model3, outdim_size = outdim_size, use_all_singular_values = False, d_feature = 6, d_model = d_model, d_inner = d_inner,
        #     n_layers=num_layers, n_head = num_heads, d_k=64, d_v=64, dropout = 0.1,
        #     class_num=3, device=torch.device('cuda'))

        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-4 ,lr = 1e-5), d_model, warm_steps)

        chkpoint = torch.load(torchload3, map_location='cuda')
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(valid_loader_text_eeg, device, model, val_text_eeg.__len__())

