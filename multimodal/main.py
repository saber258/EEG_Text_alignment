import numpy as np
import pandas as pd
from data_prepare import *
from configs import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel







if __name__ == '__main__':

    df = pd.read_csv(csv)
    df_text = df[[emotion, 'new_words']]
    df_eeg = df[eeg]

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    Multimodal_dataset = MultiModalDataset(texts = df_text.new_words.to_numpy(),
                               eeg = df_eeg.to_numpy(),
                               labels = df_text[emotion].to_numpy(),
                               tokenizer = tokenizer,
                               max_len = MAX_LEN)

    print(Multimodal_dataset[:]['eeg', 'input_ids'])
