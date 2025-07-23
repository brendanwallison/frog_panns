# sample_rate = 32000
# clip_samples = sample_rate * 30
sample_rate = 16000
clip_samples = sample_rate * 3

# https://github.com/qiuqiangkong/panns_inference/issues/8
# Hi Debayan,

#      This is because the window_size, etc. should be the same as the
# checkpoint.  Try:

# CHECKPOINT_PATH="Cnn14_16k_mAP=0.438.pth"   # Trained by a later code
# version, achieves higher mAP than the paper.
# wget -O $CHECKPOINT_PATH
# https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1
# MODEL_TYPE="Cnn14_16k"
# CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging \
#     --sample_rate=16000 \
#     --window_size=512 \
#     --hop_size=160 \
#     --mel_bins=64 \
#     --fmin=50 \
#     --fmax=8000 \
#     --model_type=$MODEL_TYPE \
#     --checkpoint_path=$CHECKPOINT_PATH \
#     --audio_path='resources/R9_ZSCveAHg_7s.wav' \
#     --cuda

mel_bins = 64
fmin = 50
fmax = 8000
window_size = 512
hop_size = 160

window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

labels = ['no_frogs', 'white_dot']
# labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 
#     'pop', 'reggae', 'rock']
    
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)