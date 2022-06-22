batch_size = 64
d_model = 128
# d_model = 16
# num_layers = 12
# num_heads = 12
num_layers = 1
num_heads = 1
class_num = 3
d_inner = 256
# d_inner = 32
dropout = 0.1
warm_steps = 4000
fea_num = 7
epoch = 200
PAD = 0
KS = 3

Fea_PLUS = 2
# SIG_LEN = 839
SIG_LEN = 32
SIG_LEN2 = 839
SIG_LEN3 = 6
MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
emotion = 'sentiment'
csv = 'df.csv'
patient = 'ZAB'




torchload = 'baselines/text/0arousal_trans_baseline_onlytext.chkpt'
torchload2 = 'baselines/eeg/0arousal_trans_baseline_onlyeeg.chkpt'
torchload3 = 'baselines/DCCA/0sentiment_baseline_DCCA_only_trans.chkpt'

outdim_size = class_num
use_all_singular_values = False


