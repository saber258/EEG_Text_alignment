import numpy as np
import pandas as pd
from data_prepare import *
from configs import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torchsampler import ImbalancedDatasetSampler






if __name__ == '__main__':

    # --- Create Train and Validation
    df = pd.read_csv(csv)

    X = df.drop([emotion], axis = 1)
    y= df[[emotion]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42, test_size = 0.33)
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_val = pd.concat([X_val, y_val], axis = 1)
    
    df_train_text = df_train[[emotion, 'new_words']]
    df_train_eeg = df_train[eeg]

    df_val_text = df_val[[emotion, 'new_words']]
    df_val_eeg = df_val[eeg]

    # --- Tokenizer    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # --- Create Dataset
    Multimodal_dataset_train = MultiModalDataset(texts = df_train_text.new_words.to_numpy(),
                               eeg = df_train_eeg.to_numpy(),
                               labels = df_train_text[emotion].to_numpy(),
                               tokenizer = tokenizer,
                               max_len = MAX_LEN)
    
    Multimodal_dataset_val = MultiModalDataset(texts = df_val_text.new_words.to_numpy(),
                                              eeg = df_val_eeg.to_numpy(),
                                              labels = df_val_text[emotion].to_numpy(),
                                              tokenizer = tokenizer,
                                              max_len = MAX_LEN)

    # --- DataLoader
    train_loader = DataLoader(dataset = Multimodal_dataset_train,
                              batch_size = batch_size,
                              num_workers = 2,
                              sampler = ImbalancedDatasetSampler(Multimodal_dataset_train),
                              shuffle = True)
    val_loader = DataLoader(dataset = Multimodal_dataset_val,
                            batch_size = batch_size,
                            num_workers = 2,
                            shuffle = True)
      
    
    
    

    
    

    print(Multimodal_dataset_train[:]['eeg'])
