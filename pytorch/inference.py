import os
import csv
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

# 🐸 Dataset: slices one audio file into 3s clips
class FrogClipDataset(Dataset):
    def __init__(self, audio_path, clip_duration=3.0, sample_rate=16000):
        self.audio_path = audio_path
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.clip_samples = int(clip_duration * sample_rate)

        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        self.waveform = waveform.squeeze(0)
        self.total_samples = self.waveform.shape[0]
        self.num_clips = self.total_samples // self.clip_samples

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.clip_samples
        clip = self.waveform[start:start + self.clip_samples]
        start_time_sec = start / self.sample_rate
        return clip.float(), os.path.basename(self.audio_path), start_time_sec
    
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model, label_mode='count', threshold=0.5):
        super().__init__()
        assert label_mode in ['binary', 'count'], "label_mode must be 'binary' or 'count'"
        self.model = model
        self.label_mode = label_mode
        self.threshold = threshold

    def forward(self, x):
        logits = self.model(x)

        if self.label_mode == 'binary':
            # logits: [batch_size, 1] or [batch_size]
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            scores = torch.sigmoid(logits)  # probability of presence
            preds = (scores >= self.threshold).long()  # 0 or 1

        elif self.label_mode == 'count':
            # logits: [batch_size, 5]
            scores = F.softmax(logits, dim=-1)  # class probabilities
            preds = torch.argmax(scores, dim=-1)  # predicted bin index

        return preds, scores

# 🧠 SoftMax + label mapping
def predict(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds, probs

# 📄 Main loop: sequential per-file inference with CSV output
def run_inference(audio_dir, model, output_csv, batch_size=256, num_workers=0, device='cuda'):
    model.eval()
    audio_files = sorted(f for f in os.listdir(audio_dir) if f.endswith('.wav'))

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'start_time_sec', 'predicted_category', 'score_0', 'score_1', 'score_2', 'score_3', 'score_4+'])

        for fname in audio_files:
            path = os.path.join(audio_dir, fname)
            dataset = FrogClipDataset(path)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

            for clips, file_names, start_times in loader:
                clips = clips.to(device)
                with torch.no_grad():
                    logits = model(clips)
                    preds, probs = predict(logits)

                for i in range(clips.size(0)):
                    label = ['0', '1', '2', '3', '4+'][preds[i].item()]
                    scores = probs[i].cpu().numpy()
                    writer.writerow([
                        file_names[i],
                        round(start_times[i].item(), 1),
                        label,
                        *[round(s, 3) for s in scores]
                    ])

def build_model(model_type, classes_num, checkpoint_path, device='cuda'):
    # Instantiate the model
    Model = eval(model_type)
    model = Model(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        freeze_base=False,
        classes_num=classes_num
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # or 'cuda' if needed

    # If checkpoint is a full dict with metadata
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Move to device
    model.to(device)
    return model

def predict(model, x):
    logits = model(x)
    return torch.nn.functional.softmax(logits, dim=-1)

def batch_inference(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, 'inference_results.csv')

    # Build model
    classes_num = 1 if args.label_mode == 'binary' else 5
    model = build_model(
        model_type=args.model_type,
        classes_num=classes_num,
        checkpoint_path=args.checkpoint_path,
        device=device
    )

    # Run inference
    run_inference(
        audio_dir=args.dataset_dir,
        model=model,
        output_csv=output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )

    print(f'Inference results saved to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train frog call classifier.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory for batch inference.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output Directory for inference results.')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--label_mode', type=str, choices=['binary', 'count'], default='count')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    batch_inference(args)