import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import *
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from model_new import Encoder, Encoder2


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
  def __init__(self, model1, model2, d_feature = 40, d_model = 16, class_num = class_num):
    super(Fusion, self).__init__()
    self.model1 = model1
    self.model2 = model2
    self.classifier = nn.Linear(4, class_num)
    # self.linear1_cov = nn.Conv1d(8, 1, kernel_size=1)
    # self.linear1_linear = nn.Linear(4, class_num)
    # # self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
    # # self.linear2_linear = nn.Linear(d_feature, class_num)

  def forward(self, x1, x2):
    x1 = self.model1(x1)
    x2 = self.model2(x2)
    x = torch.cat((x1, x2), dim = 1)
    # out = self.linear1_cov(x)
    # out = self.linear1_linear(out)
    out = self.classifier(x)
    return out, x1, x2


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
            self, device,
            d_feature1, d_feature2, d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
            class_num=2):

        super().__init__()

        self.encoder = Encoder(d_feature1, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.encoder2 = Encoder2(d_feature2, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device
        self.d_feature = d_feature
        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)

    def forward(self, src_seq1, src_seq2):
        b, l = src_seq1.size()
        src_pos1 = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos1 = src_pos1.to(self.device)

        b2, l2 = src_seq2.size()
        src_pos2 = torch.LongTensor(
            [list(range(1, l2 + 1)) for i in range(b2)]
        )
        src_pos2 = src_pos2.to(self.device)

        enc_output1, *_ = self.encoder(src_seq1, src_pos1)
        enc_output2, *_ = self.encoder2(src_seq2, src_pos2)
        enc_output = torch.cat((enc_output1, enc_output2), dim = 1)
        dec_output = enc_output
        res = self.linear1_cov(dec_output)
        res = res.contiguous().view(res.size()[0], -1)
        res = self.linear1_linear(res)
        return res
