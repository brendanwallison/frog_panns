from frogcall_dataset import FrogCallDataset
from torch.utils.data import DataLoader

import os
from frogcall_dataset import FrogCallDataset
from torch.utils.data import DataLoader
from scipy.io.wavfile import write as wavwrite
import numpy as np

def save_clips_from_batch(clips, labels, audio_files, starts, sample_rate, out_dir):
    for i in range(len(clips)):
        label = labels[i].item()
        if label == 1:
            subfolder = 'white_dot'
        elif label == 0.5:
            subfolder = 'white_dot_q2'
        else:
            subfolder = 'no_frog_call'
        save_folder = os.path.join(out_dir, subfolder)
        os.makedirs(save_folder, exist_ok=True)
        # Unique filename: audio file + start time
        base = os.path.splitext(audio_files[i])[0]
        start_sec = starts[i] / sample_rate
        filename = f"{base}_start_{start_sec:.2f}s.wav"
        filepath = os.path.join(save_folder, filename)
        wavwrite(filepath, sample_rate, (clips[i].numpy() * 32767).astype(np.int16))

def main():
    audio_dir = 'data/Data'
    annotation_dir = 'data/annotations'
    out_dir = 'frogcall_extracts'
    dataset = FrogCallDataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        clip_duration=3.0,
        sample_rate=16000,
        train=True,
        test_split=0.2,
        pos_ratio=0.5,
        random_seed=42
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    # For each batch, extract and save clips
    for batch_idx, batch in enumerate(loader):
        clips, labels, audio_files, starts = batch
        save_clips_from_batch(clips, labels, audio_files, starts, dataset.sample_rate, out_dir)
        print(f"Saved batch {batch_idx}")
        # Optionally, limit number of batches
        if batch_idx > 5: break

    print("Done")

# def main():
#     audio_dir = 'data/Data'
#     annotation_dir = 'data/annotations'
#     dataset = FrogCallDataset(
#         audio_dir=audio_dir,
#         annotation_dir=annotation_dir,
#         clip_duration=3.0,
#         sample_rate=16000,
#         train=True,
#         test_split=0.2,
#         in_memory=True,
#         pos_ratio=0.5,
#         random_seed=42
#     )

#     print(f"Total samples: {len(dataset)}")
#     for i in range(5):
#         clip, label = dataset[i]
#         print(f"Sample {i}: clip shape = {clip.shape}, label = {label}")

#     # Optional: test DataLoader batching
#     loader = DataLoader(dataset, batch_size=4, shuffle=True)
#     batch = next(iter(loader))
#     print(f"Batched clip shape: {batch[0].shape}, Batched label shape: {batch[1].shape}")

#     for i in range(10):
#         clip, label = dataset[i]
#         entry = dataset.positive_clips[i] if label > 0 else dataset.negative_clips[i]
#         audio_idx = entry['audio_idx']
#         start = entry['start']
#         audio_file = dataset.audio_files[audio_idx]
#         print(f"Sample {i}:")
#         print(f"  Audio file: {audio_file}")
#         print(f"  Clip start (s): {start / dataset.sample_rate:.2f}")
#         print(f"  Clip end (s): {(start + int(dataset.clip_duration * dataset.sample_rate)) / dataset.sample_rate:.2f}")
#         print(f"  Label: {label}")
#         # Optionally, print corresponding annotation rows
#         annotations = dataset._load_annotations(audio_file)
#         print(annotations)
#         print("-" * 40)

#     print("Done")

if __name__ == "__main__":
    main()