batch_size = 8
# d_model = 64
d_model = 1024
# num_layers = 4
# num_heads = 5
num_layers = 12
num_heads = 12
class_num = 2
# d_inner = 512
d_inner = 768
dropout = 0.1
warm_steps = 4000
fea_num = 7
epoch = 100
PAD = 0
KS = 3

Fea_PLUS = 2
SIG_LEN = 32
SIG_LEN2 = 8
MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
emotion = 'angry2_trans'
csv = 'df.csv'
eeg = [emotion, 'delta2', 'lowAlpha2', 'highAlpha2', 'lowBeta2', 'highBeta2',
'lowGamma2', 'middleGamma2', 'theta2']
# eeg = [emotion, 'delta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta',
# 'lowGamma', 'middleGamma', 'theta']

# torchload = 'baselines/text/0angry2_trans_baseline_onlytext_transform.chkpt'
# torchload2 = 'baselines/eeg/0angry2_trans_baseline_onlyeeg_transform.chkpt'

torchload = 'baselines/text/0angry2_trans_baseline_onlytext.chkpt'
torchload2 = 'baselines/eeg/0angry2_trans_baseline_onlyeeg.chkpt'

outdim_size = class_num
use_all_singular_values = False

####
'''
finished
TEXT
happy1 
happy2  

nervous1 
nervous2 

sad
sad2

angry
angry2


arousal1 
arousal2 

valence1 
valence2 

EEG

happy2
happy1

sad2
sad

angry1
angry2

nervous1
nervous2

arousal1
arousal2

valence1
valence2

'''
###