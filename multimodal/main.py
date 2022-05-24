import numpy as np
import pandas as pd
from data_prepare import *
from configs import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 






if __name__ == '__main__':

    df = pd.read_csv(csv)
    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_test, y_val = train_test_split(X, y, random_state = 42, test_size = 0.33)
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)
    
    df_train_text = df_train[[emotion, 'new_words']]
    df_train_eeg = df_train[eeg]

    df_val_text = df_val[[emotion, 'new_words']]
    df_val_eeg = df_val[eeg]

    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    Multimodal_dataset_train = MultiModalDataset(texts = df_text.new_words.to_numpy(),
                               eeg = df_eeg.to_numpy(),
                               labels = df_text[emotion].to_numpy(),
                               tokenizer = tokenizer,
                               max_len = MAX_LEN)
    
    Multimodal_dataset_val = MultiModalDatset(texts = df_val.new_words.to_numpy(),
                                              eeg = df_eeg.to_numpy(),
                                              labels = df_text[emotion].to_numpy(),
                                              tokenizer = tokenizer,
                                              max_lan = MAX_LEN)
    

    print(len(Multimodal_dataset[:]['eeg']))
