import numpy as np
import pandas as pd
from data_prepare import *
from configs import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

df = pd.read_csv('df.csv')

y = df[['happy_trans']]

X_text = df[['new_words']]
df_eeg = df[['delta', 'lowAlpha',
       'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'middleGamma', 'theta']]

df_text = pd.concat([y, X_text], axis = 1)


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

Multimodal_dataset = MultiModalDataset(texts = df_text.new_words.to_numpy(),
                           eeg = df_eeg.to_numpy(),
                           labels = df_text.happy_trans.to_numpy(),
                           tokenizer = tokenizer,
                           max_len = MAX_LEN)


if __name__ == '__main__':
    print(Multimodal_dataset[0])
