batch_size = 64
d_model = 16
# d_model = 16
# num_layers = 12
# num_heads = 12
num_layers = 1
num_heads = 2
class_num = 3
d_inner = 32
# d_inner = 32
dropout = 0.3
warm_steps = 2000
fea_num = 7
epoch = 200
PAD = 0
KS = 3

Fea_PLUS = 2
# SIG_LEN = 832
SIG_LEN = 768
SIG_LEN2 = 832
SIG_LEN3 = 6
MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
emotion = 'sentiment'
csv = 'df.csv'
patient = 'ZAB'




# torchload = 'baselines/text/0sentiment_baseline_onlytext.chkpt'
# torchload2 = 'baselines/eeg/0sentiment_baseline_onlyeeg.chkpt'

torchload = 'baselines/text/0sentiment_baseline_onlytext.chkpt'
torchload2 = 'baselines/eeg/0sentiment_baseline_onlyeeg.chkpt'
# torchload3 = 'baselines/fusion_cossim/0sentiment_baseline_fusion_cossim_trans.chkpt'
# torchload3 = 'baselines/fusion_wd/0sentiment_baseline_fusion_wd_trans.chkpt'
# torchload3 = 'baselines/text_eeg_fusion/0sentiment_baseline_fusion_linout.chkpt'
# torchload3 = 'baselines/DCCA_fusion/0sentiment_baseline_DCCA_fusion_trans.chkpt'
torchload3 = 'baselines/fusion_cossim_ds/0sentiment_baseline_fusion_cossim_eeg_trans.chkpt'
# torchload3 = 'baselines/fusion_wd_ds/0sentiment_baseline_fusion_wd_text_trans.chkpt'
# torchload3 = 'baselines/DCCA_ds/0sentiment_baseline_onlyeeg_trans.chkpt'
outdim_size = class_num
use_all_singular_values = False


