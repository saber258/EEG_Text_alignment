from google.colab import output
import numpy as np
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers.utils.dummy_flax_objects import FlaxAlbertForSequenceClassification
from transformers.utils.dummy_pt_objects import TextDatasetForNextSentencePrediction
from config import *
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from block_new import get_sinusoid_encoding_table
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_new import Encoder, Encoder2, Transformer, Transformer3, Encoder3
import scipy.stats as stats
from torch.autograd import Variable
from transformers import BertModel

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
    return 768
  
  def sig_len(self):
    return self.signals.shape[1]

  def __len__(self):
    return self.n_insts

  def __getitem__(self, item):
    text = self.texts[item]
    label = self.labels[item]
    signal = self.signals[item]
    text = torch.FloatTensor(text)
    # print(text.shape)
    
    return signal, text, torch.tensor(label, dtype=torch.long)

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

class BiLSTM(nn.Module):
    #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, device, embedding_dim = 32, hidden_dim1 = 128, hidden_dim2 = 64, output_dim = 3, n_layers =2,
                 dropout = 0.3, bidirectional = True, pad_index = 1):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        
        # n_position = vocab_size + 1
        # self.embedding = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(n_position, 30, padding_idx=0),
        #     freeze=True)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        b, l = text.size()
        # print(b)
        # print(l)
        text_lengths = torch.tensor([32]*b).cpu()
        src_pos = torch.LongTensor(
            [list(range(0, l)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)
        # print(src_pos)
        # print(text)
        embedded = self.embedding(src_pos)
        # print(text.shape)
        # print(embedded.shape)
        # print('hello')
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True) 
        # print(packed_embedded)
        packed_output, (hidden, cell) = self.lstm(text)
        # print(packed_output)
        # print(packed_output.shape)
        # cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(packed_output)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

# class BiLSTM(nn.Module):

#     # define all the layers used in model
#     def __init__(self, vocab_size, device, embedding_dim = 30, lstm_units = 64, hidden_dim =32, num_classes=3, lstm_layers =2,
#                  bidirectional = True, dropout = 0.3, pad_index = 0, batch_size = 64):
#         super().__init__()
#         self.device = device
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
#         self.lstm = nn.LSTM(embedding_dim,
#                             lstm_units,
#                             num_layers=lstm_layers,
#                             bidirectional=bidirectional,
#                             batch_first=True)
#         num_directions = 2 if bidirectional else 1
#         self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.lstm_layers = lstm_layers
#         self.num_directions = num_directions
#         self.lstm_units = lstm_units


#     def init_hidden(self, batch_size):
#         h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)),
#                 Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)))
#         return h.to(self.device), c.to(self.device)

#     def forward(self, text):
#         batch_size = text.shape[0]
#         h_0, c_0 = self.init_hidden(batch_size)
#         b, l = text.size()
#         text_lengths = torch.tensor([48]*b).cpu()
#         src_pos = torch.LongTensor(
#             [list(range(1, l)) for i in range(b)]
#         )
#         src_pos = src_pos.to(self.device)
#         # print(src_pos)
#         embedded = self.embedding(src_pos)
#         # embedded = self.embedding(text)
#         packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)
#         output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
#         output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
#         out = output_unpacked[:, -1, :]
#         rel = self.relu(out)
#         dense1 = self.fc1(rel)
#         drop = self.dropout(dense1)
#         preds = self.fc2(drop)
#         return preds


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


class MLP(nn.Module):
  #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, embed_size = 1, hidden_size2 = 256, hidden_size3 = 128, hidden_size4 = 64, 
    output_dim = class_num, dropout = 0.3, max_document_length = 32):
        super().__init__()
        # embedding and convolution layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size*max_document_length, hidden_size2)  # dense layer
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)  # dense layer
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  # dense layer
        self.fc4 = nn.Linear(hidden_size4, output_dim)  # dense layer

    def forward(self, text):
        # text shape = (batch_size, num_sequences)
        b, l = text.size()
 
        # text_lengths = torch.tensor([48]*b).cpu()
        # embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        # x = embedded.view(embedded.shape[0], -1)  # x = Flatten()(x)
        x = self.relu(self.fc1(text))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        return preds

class MLP2(nn.Module):
  #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, embed_size = 1, hidden_size2 = 256, hidden_size3 = 128, hidden_size4 = 64, 
    output_dim = class_num, dropout = 0.3, max_document_length = 838):
        super().__init__()
        # embedding and convolution layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size*max_document_length, hidden_size2)  # dense layer
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)  # dense layer
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  # dense layer
        self.fc4 = nn.Linear(hidden_size4, output_dim)  # dense layer

    def forward(self, text):
        # text shape = (batch_size, num_sequences)
        b, l = text.size()
 
        # text_lengths = torch.tensor([48]*b).cpu()
        # embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        # x = embedded.view(embedded.shape[0], -1)  # x = Flatten()(x)
        x = self.relu(self.fc1(text))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        return preds


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
#                         nn.BatchNorm2d(out_channels),
#                         nn.ReLU())
#         self.conv2 = nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
#                         nn.BatchNorm2d(out_channels))
#         self.downsample = downsample
#         self.relu = nn.ReLU()
#         self.out_channels = out_channels
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes = 10):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
#                         nn.BatchNorm2d(64),
#                         nn.ReLU())
#         self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
#         self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
#         self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
#         self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512, num_classes)
        
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
            
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x

# class Block(nn.Module):
#     def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
#         assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
#         super(Block, self).__init__()
#         self.num_layers = num_layers
#         if self.num_layers > 34:
#             self.expansion = 4
#         else:
#             self.expansion = 1
#         # ResNet50, 101, and 152 include additional layer of 1x1 kernels
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         if self.num_layers > 34:
#             self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         else:
#             # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
#             self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.relu = nn.ReLU()
#         self.identity_downsample = identity_downsample

#     def forward(self, x):
#         identity = x
#         if self.num_layers > 34:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)

#         if self.identity_downsample is not None:
#             identity = self.identity_downsample(identity)

#         x += identity
#         x = self.relu(x)
#         return x


# class ResNet(nn.Module):
#     def __init__(self, num_layers, block, image_channels, num_classes):
#         assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
#                                                      f'to be 18, 34, 50, 101, or 152 '
#         super(ResNet, self).__init__()
#         if num_layers < 50:
#             self.expansion = 1
#         else:
#             self.expansion = 4
#         if num_layers == 18:
#             layers = [2, 2, 2, 2]
#         elif num_layers == 34 or num_layers == 50:
#             layers = [3, 4, 6, 3]
#         elif num_layers == 101:
#             layers = [3, 4, 23, 3]
#         else:
#             layers = [3, 8, 36, 3]
#         self.in_channels = 64
#         self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # ResNetLayers
#         self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
#         self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
#         self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
#         self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * self.expansion, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x

#     def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
#         layers = []

#         identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
#                                             nn.BatchNorm2d(intermediate_channels*self.expansion))
#         layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
#         self.in_channels = intermediate_channels * self.expansion # 256
#         for i in range(num_residual_blocks - 1):
#             layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
#         return nn.Sequential(*layers)


# def ResNet18(img_channels=3, num_classes=1000):
#     return ResNet(18, Block, img_channels, num_classes)


# def ResNet34(img_channels=3, num_classes=1000):
#     return ResNet(34, Block, img_channels, num_classes)


# def ResNet50(img_channels=3, num_classes=3):
#     return ResNet(50, Block, img_channels, num_classes)


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        # print(x)
        # print(x.shape)
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        
        return out    