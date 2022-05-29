batch_size = 16
# d_model = 64
d_model = 128
# num_layers = 4
# num_heads = 5
num_layers = 2
num_heads = 4
class_num = 3
# d_inner = 512
d_inner = 256
dropout = 0.1
warm_steps = 4000 
fea_num = 7
epoch = 100
PAD = 0
KS = 3
Fea_PLUS = 2
SIG_LEN = 32
MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
emotion = 'valence2_trans'
csv = 'df.csv'
eeg = [emotion, 'delta2', 'lowAlpha2', 'highAlpha2', 'lowBeta2', 'highBeta2',
'lowGamma2', 'middleGamma2', 'theta2']


####
'''
finished
TEXT
happy1
happy2

nervous1
nervous2

arousal1
arousal2

valence1
valence2
'''
###