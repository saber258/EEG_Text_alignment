import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from config import *
import torch.nn.functional as F

class EEGDataset(Dataset):
    def __init__(self, signal, label):

        self._signal = torch.FloatTensor(signal)
        self._label = torch.LongTensor(label)


    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._label)

    @property
    def sig_len(self):
        return self._signal.shape[1]

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._signal[idx], self._label[idx]

class TextDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  @property
  def n_insts(self):
    return len(self.labels)

  @property
  def text_len(self):
    return 32

  def __len__(self):
    return self.n_insts

  def __getitem__(self, item):
    text = str(self.texts[item])
    label = self.labels[item]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      truncation = True,
      return_attention_mask=True
      #return_tensors='pt',
    )
    return torch.FloatTensor(encoding['input_ids']).flatten(), torch.tensor(label, dtype=torch.long)


class Fusion(nn.Module):
  def __init__(self, model1, model2):
    super(Fusion, self).__init__()
    self.model1 = model1
    self.model2 = model2
    self.classifier = nn.Linear(4, class_num)

  def forward(self, x1, x2):
    x1 = self.model1(x1)
    x2 = self.model2(x2)
    x = torch.cat((x1, x2), dim = 1)
    out = self.classifier(x)
    return out


class Whole(nn.Module):
  def __init__(self,model1, model2):
    super(Whole, self).__init__()
    self.model1 = model1
    self.model2 = model2

  def forward(self, x1, x2):
    x1 = self.model1(x1)
    x2 = self.model2(x2)

    return x1, x2

    
