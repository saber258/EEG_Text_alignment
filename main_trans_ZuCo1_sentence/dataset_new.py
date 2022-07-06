from google.colab import output
import numpy as np
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import *
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from model_new import Encoder, Encoder2, Transformer, Transformer3, Encoder3
import scipy.stats as stats

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
      add_special_tokens=False,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      truncation = True,
      return_attention_mask=True
      #return_tensors='pt',
    )
    return torch.FloatTensor(encoding['input_ids']).flatten(), torch.tensor(label, dtype=torch.long)

class Text_EEGDataset(Dataset):
  def __init__(self, texts, signals, labels, tokenizer, max_len):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.signals = torch.FloatTensor(signals)

  @property
  def n_insts(self):
    return len(self.labels)

  @property
  def text_len(self):
    return 32
  
  def sig_len(self):
    return self.signals.shape[1]

  def __len__(self):
    return self.n_insts

  def __getitem__(self, item):
    text = str(self.texts[item])
    label = self.labels[item]
    signal = self.signals[item]

    input_ids = [self.tokenizer.encode(text, add_special_tokens=False,max_length=MAX_LEN, padding = 'max_length', truncation = True, return_token_type_ids = False, return_attention_mask = True)]   
    input_ids = np.array(input_ids)
    input_ids = stats.zscore(input_ids, axis=None, nan_policy='omit')
    return signal, torch.FloatTensor(input_ids).flatten(), torch.tensor(label, dtype=torch.long)


class Linear(nn.Module):
  def __init__(self, device, d_feature, class_num):
      super(Linear, self).__init__()

      # self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
      self.batchnorm = nn.BatchNorm1d(128)
      self.bn = nn.BatchNorm1d(64)
      self.linear1_linear = nn.Linear(d_feature, 128)
      self.hidden = nn.Linear(128, 64)
      self.dropout = nn.Dropout(0.25)
      self.classifier = nn.Linear(64, class_num)
  def forward(self,x1):
    # x1 = self.linear1_cov(x1)
    # x1 = x1.contiguous().view(x1.size()[0], -1)
    x1 = self.linear1_linear(x1)
    x1 = self.batchnorm(x1)
    x1 = self.dropout(x1)
    x1 = self.hidden(x1)
    x1 = self.bn(x1)
    x1 = self.dropout(x1)
    out = self.classifier(F.relu(x1))

    return out


# class Fusion(nn.Module):
#   def __init__(self, device, model1, model2,
#             d_feature, d_model, d_inner,
#             n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
#             class_num=3):
#     super(Fusion, self).__init__()
#     self.device = device
#     self.model1 = model1
#     self.model2 = model2
#     self.Transformer = Transformer3(device=device, d_feature=6, d_model=d_model, d_inner=d_inner,
#                             n_layers=n_layers, n_head=n_head, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
#     self.classifier = nn.Linear(6, class_num)
#     self.bn = nn.BatchNorm1d(6)
#     self.dropout = nn.Dropout(0.25)

#     # self.linear1_cov = nn.Conv1d(8, 1, kernel_size=1)
#     # self.linear1_linear = nn.Linear(4, class_num)
#     # # self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
#     # # self.linear2_linear = nn.Linear(d_feature, class_num)

#   def forward(self, x1, x2):
#     x1 = self.model1(x1)
#     x2 = self.model2(x2)
    
#     x = torch.cat((x1, x2), dim = 1)
#     # x = self.bn(x)
#     # x = self.dropout(x)

#     # out = self.linear1_cov(x)

#     out = self.classifier(x)

#     # out = self.Transformer(x)
#     return out, x1, x2

class Fusion(nn.Module):
    def __init__(self, model1, model2):
        super(Fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.classifier = nn.Linear(6, 3)
       
      

    def forward(self, x1, x2):
        
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        
        

        return output1, output2

class Whole(nn.Module):
  def __init__(self,model1, model2):
    super(Whole, self).__init__()
    self.model1 = model1
    self.model2 = model2

  def forward(self, x1, x2):
    x1 = self.model1(x1)
    x2 = self.model2(x2)

    return x1, x2



class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    


class TransformerFusion(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device, model1, model2,
            d_feature1, d_feature2, d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
            class_num=3):

        super().__init__()

        self.encoder3 = Encoder3(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device
        self.Transformer = Transformer3(device=device, d_feature=80, d_model=d_model, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
        self.model1 = model1
        self.model2 = model2
        self.d_feature = d_feature
        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)

    def forward(self, src_seq1, src_seq2):

        enc_output1, _ = self.model1(src_seq1)
        print(enc_output1)
        print(enc_output1.size())
        enc_output2, _ = self.model2(src_seq2)
        print(enc_output2)
        print(enc_output2.size())
        src_seq = torch.cat((enc_output1, enc_output2), dim = 1)
        
      
        b, l, _ = src_seq.size()
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder3(src_seq, src_pos)

        dec_output = enc_output
        res = self.linear1_cov(dec_output)
        res = res.contiguous().view(res.size()[0], -1)
        res = self.linear1_linear(res)

        return res
