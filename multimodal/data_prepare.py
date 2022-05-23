import numpy as np
import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
  def __init__(self, eeg, texts, labels, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.eeg = eeg

  def __len__(self):
    return len(self.texts), len(self.eeg)

  def __getitem__(self, item):
    text = str(self.texts[item])
    eeg = self.eeg[item]
    label = self.labels[item]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'eeg' : torch.tensor(eeg, dtype = float),
      'labels': torch.tensor(label, dtype=torch.long)
    }
