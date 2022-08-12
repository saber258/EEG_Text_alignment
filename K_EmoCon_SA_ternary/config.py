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
# SIG_LEN = 48
SIG_LEN = 32
SIG_LEN2 = 48
SIG_LEN3 = 6
MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
emotion = 'arousal_trans'
csv = 'df.csv'
patient = 'ZAB'

eeg = [emotion, 'delta0', 'lowAlpha0', 'highAlpha0','lowBeta0','highBeta0', 'lowGamma0', 'middleGamma0', 'theta0',
         'delta1', 'lowAlpha1', 'highAlpha1', 'lowBeta1', 'highBeta1', 'lowGamma1', 'middleGamma1', 'theta1',
         'delta2', 'lowAlpha2', 'highAlpha2', 'lowBeta2', 'highBeta2', 'lowGamma2', 'middleGamma2', 'theta2',
         'delta3', 'lowAlpha3', 'highAlpha3', 'lowBeta3', 'highBeta3', 'lowGamma3', 'middleGamma3', 'theta3',
         'delta4', 'lowAlpha4', 'highAlpha4', 'lowBeta4', 'highBeta4', 'lowGamma4', 'middleGamma4', 'theta4',
         'delta5', 'lowAlpha5', 'highAlpha5', 'lowBeta5', 'highBeta5', 'lowGamma5', 'middleGamma5', 'theta5']

# eeg = [emotion, 'delta0_2', 'lowAlpha0_2', 'highAlpha0_2','lowBeta0_2','highBeta0_2', 'lowGamma0_2', 'middleGamma0_2', 'theta0_2',
#          'delta1_2', 'lowAlpha1_2', 'highAlpha1_2', 'lowBeta1_2', 'highBeta1_2', 'lowGamma1_2', 'middleGamma1_2', 'theta1_2',
#          'delta2_2', 'lowAlpha2_2', 'highAlpha2_2', 'lowBeta2_2', 'highBeta2_2', 'lowGamma2_2', 'middleGamma2_2', 'theta2_2',
#          'delta3_2', 'lowAlpha3_2', 'highAlpha3_2', 'lowBeta3_2', 'highBeta3_2', 'lowGamma3_2', 'middleGamma3_2', 'theta3_2',
#          'delta4_2', 'lowAlpha4_2', 'highAlpha4_2', 'lowBeta4_2', 'highBeta4_2', 'lowGamma4_2', 'middleGamma4_2', 'theta4_2',
#          'delta5_2', 'lowAlpha5_2', 'highAlpha5_2', 'lowBeta5_2', 'highBeta5_2', 'lowGamma5_2', 'middleGamma5_2', 'theta5_2']



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


