import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import hashlib
import torchaudio
from concurrent.futures import ThreadPoolExecutor

class RangeSampler:
    def __init__(self, ranges):
        self.ranges = ranges
        self.cumulative_sizes = []
        total = 0
        for r in ranges:
            size = r['end'] - r['start']
            total += size
            self.cumulative_sizes.append(total)
        self.total = total

    def sample(self):
        idx = random.randint(0, self.total - 1)
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                r = self.ranges[i]
                offset = idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
                return {'audio_file': r['audio_file'], 'start': r['start'] + offset}

class FrogCallDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, clip_duration=3.0, sample_rate=16000,
                 train=True, test_split=0.2, pos_ratio=0.5, random_seed=None, label_mode='binary'):

        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.pos_ratio = pos_ratio
        self.clip_samples = int(self.clip_duration * self.sample_rate)
        self.train = train
        self.test_split = test_split
        self.random_seed = random_seed
        self.label_mode = label_mode

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.audio_files = self._load_or_create_split()
        self.metadata_path = self._get_metadata_path()
        if os.path.exists(self.metadata_path):
            self._load_metadata()
        else:
            self._compute_and_save_metadata()

        self.pos_sampler = RangeSampler(self.positive_ranges)
        self.neg_sampler = RangeSampler(self.negative_ranges)

    def _get_split_path(self):
        return os.path.join(self.audio_dir, f"dataset_split_seed{self.random_seed}_split{self.test_split}.json")

    def _load_or_create_split(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        split_path = self._get_split_path()
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split = json.load(f)
        else:
            all_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
            random.shuffle(all_files)
            split_idx = int(len(all_files) * (1 - self.test_split))
            split = {
                'train': all_files[:split_idx],
                'test': all_files[split_idx:]
            }
            with open(split_path, 'w') as f:
                json.dump(split, f)
        return split['train'] if self.train else split['test']

    def _get_metadata_path(self):
        key = "_".join(sorted(self.audio_files))
        split_tag = "train" if self.train else "test"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.audio_dir, f"clip_metadata_{split_tag}_{hash_key}.json")

    def _load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        self.positive_ranges = metadata['positive_ranges']
        self.negative_ranges = metadata['negative_ranges']
        self.audio_files = metadata['audio_files']

        all_audio_files = set(os.listdir(self.audio_dir))
        missing_files = set(r['audio_file'] for r in self.positive_ranges + self.negative_ranges
                            if r['audio_file'] not in all_audio_files)
        if missing_files:
            raise FileNotFoundError(f"Metadata references missing audio files: {sorted(missing_files)}")

        pos_total = sum(r['end'] - r['start'] for r in self.positive_ranges)
        neg_total = sum(r['end'] - r['start'] for r in self.negative_ranges)
        print(f"[{self.train and 'TRAIN' or 'TEST'}] Loaded {len(self.audio_files)} files with {pos_total} positive and {neg_total} negative clips.")

    def _compute_and_save_metadata(self):
        self.positive_ranges = []
        self.negative_ranges = []

        def process_file(audio_file):
            annotations = self._load_annotations(audio_file)
            audio_path = os.path.join(self.audio_dir, audio_file)
            try:
                info = torchaudio.info(audio_path)
            except Exception as e:
                print(f"Failed to read {audio_file}: {e}")
                return [], []
            total_samples = info.num_frames

            all_start = 0
            all_end = total_samples - self.clip_samples + 1
            all_indices = set(range(all_start, all_end))

            pos_indices = set()
            for _, row in annotations.iterrows():
                if 'white dot' not in row['Annotation']:
                    continue
                ann_start = float(row['Begin Time (s)'])
                ann_end = float(row['End Time (s)'])
                ann_duration = ann_end - ann_start
                min_overlap = 0.5 * ann_duration

                clip_start_min = ann_start + min_overlap - self.clip_duration
                clip_start_max = ann_end - min_overlap

                start_idx_min = max(0, int(clip_start_min * self.sample_rate))
                start_idx_max_raw = clip_start_max * self.sample_rate
                start_idx_max = min(int(start_idx_max_raw), total_samples - self.clip_samples)

                pos_indices.update(range(start_idx_min, start_idx_max + 1))

            neg_indices = sorted(all_indices - pos_indices)
            pos_indices = sorted(pos_indices)

            def to_ranges(start_list):
                if not start_list:
                    return []
                ranges = []
                current_start = start_list[0]
                prev = start_list[0]
                for s in start_list[1:]:
                    if s == prev + 1:
                        prev = s
                    else:
                        ranges.append({'audio_file': audio_file, 'start': current_start, 'end': prev + 1})
                        current_start = s
                        prev = s
                ranges.append({'audio_file': audio_file, 'start': current_start, 'end': prev + 1})
                return ranges

            return to_ranges(pos_indices), to_ranges(neg_indices)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_file, self.audio_files))

        for pos, neg in results:
            self.positive_ranges.extend(pos)
            self.negative_ranges.extend(neg)

        metadata = {
            'positive_ranges': self.positive_ranges,
            'negative_ranges': self.negative_ranges,
            'audio_files': self.audio_files,
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

        pos_total = sum(r['end'] - r['start'] for r in self.positive_ranges)
        neg_total = sum(r['end'] - r['start'] for r in self.negative_ranges)
        print(f"[{self.train and 'TRAIN' or 'TEST'}] Computed metadata for {len(self.audio_files)} files with {pos_total} positive and {neg_total} negative clips.")

    def compare_fast_vs_true_labels(self, num_samples=1000, threshold=0.05):
        fast_positive = 0
        true_positive = 0
        flipped_to_positive = 0
        flipped_to_negative = 0

        for _ in range(num_samples):
            use_pos = np.random.rand() < self.pos_ratio and self.pos_sampler.total > 0
            entry = self.pos_sampler.sample() if use_pos else self.neg_sampler.sample()

            audio_file = entry['audio_file']
            start = entry['start']
            audio_path = os.path.join(self.audio_dir, audio_file)

            try:
                waveform, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=self.clip_samples)
            except Exception as e:
                print(f"Failed to load {audio_file} at {start}: {e}")
                continue

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            clip = waveform.squeeze(0)

            annotations = self._load_annotations(audio_file)
            clip_start = start / self.sample_rate
            clip_end = clip_start + self.clip_duration
            true_label = self._has_frog_call(annotations, clip_start, clip_end)

            fast_label = 1.0 if use_pos else 0.0
            true_binary = 1.0 if true_label > threshold else 0.0

            fast_positive += int(fast_label == 1.0)
            true_positive += int(true_binary == 1.0)

            if fast_label == 1.0 and true_binary == 0.0:
                flipped_to_negative += 1
            elif fast_label == 0.0 and true_binary == 1.0:
                flipped_to_positive += 1

        print(f"\n🔍 Fast vs. True Label Comparison (threshold={threshold}):")
        print(f"Sampled: {num_samples}")
        print(f"Fast positives: {fast_positive}")
        print(f"True positives: {true_positive}")
        print(f"Flipped to negative: {flipped_to_negative}")
        print(f"Flipped to positive: {flipped_to_positive}")
        print(f"Agreement rate: {(num_samples - flipped_to_negative - flipped_to_positive) / num_samples:.3f}")

    def _load_annotations(self, audio_file):
        base = os.path.splitext(audio_file)[0]
        ann_filename = f"{base}.Table.1.selections.txt"
        ann_path = os.path.join(self.annotation_dir, ann_filename)
        if not os.path.exists(ann_path):
            return pd.DataFrame(columns=['Annotation', 'Begin Time (s)', 'End Time (s)'])
        df = pd.read_csv(ann_path, sep='\t')
        df['Annotation'] = df['Annotation'].astype(str).str.strip().str.lower()
        return df

    def _has_frog_call(self, annotations, clip_start, clip_end):
        max_overlap = 0
        confidence = 1
        for _, row in annotations.iterrows():
            if 'white dot' in row['Annotation']:
                ann_start = float(row['Begin Time (s)'])
                ann_end = float(row['End Time (s)'])
                overlap_start = max(clip_start, ann_start)
                overlap_end = min(clip_end, ann_end)
                overlap = max(0, overlap_end - overlap_start)
                fraction = overlap / (ann_end - ann_start)
                if fraction > max_overlap:
                    max_overlap = fraction
                    confidence = 0.75 if 'q2' in row['Annotation'] else 1
        return max_overlap * confidence
    
    def get_frog_call_weights(self, annotations, clip_start, clip_end):
        weights = []
        for _, row in annotations.iterrows():
            if 'white dot' in row['Annotation']:
                ann_start = float(row['Begin Time (s)'])
                ann_end = float(row['End Time (s)'])
                overlap_start = max(clip_start, ann_start)
                overlap_end = min(clip_end, ann_end)
                overlap = max(0, overlap_end - overlap_start)
                fraction = overlap / (ann_end - ann_start)
                confidence = 0.75 if 'q2' in row['Annotation'] else 1.0
                weight = fraction * confidence
                if weight > 0:
                    weights.append(weight)
        return weights  # list of floats in [0, 1]

    def soft_count_distribution(self, weights, max_bin=4):
        """
        Returns a soft count distribution over bins: [0, 1, 2, 3, 4+]
        """
        if not weights:
            dist = torch.zeros(max_bin + 1)
            dist[0] = 1.0
            return dist

        probs = torch.tensor(weights, dtype=torch.float32)  # shape: [N]
        dist = torch.tensor([1.0])  # start with 0 calls

        for p in probs:
            dist = torch.cat([dist * (1 - p), torch.zeros(1)]) + torch.cat([torch.zeros(1), dist * p])

        # Pad or truncate to max_bin
        if len(dist) <= max_bin:
            dist = torch.nn.functional.pad(dist, (0, max_bin + 1 - len(dist)))
        else:
            dist[max_bin] = dist[max_bin:].sum()
            dist = dist[:max_bin + 1]

        return dist / dist.sum()  # normalize

    def __len__(self):
        return 1000
    
    def true_length(self):
        return self.pos_sampler.total + self.neg_sampler.total

    def __getitem__(self, idx):
        if self.pos_sampler.total == 0 and self.neg_sampler.total == 0:
            raise RuntimeError("No clips available for sampling.")

        use_pos = np.random.rand() < self.pos_ratio and self.pos_sampler.total > 0
        entry = self.pos_sampler.sample() if use_pos else self.neg_sampler.sample()

        audio_file = entry['audio_file']
        start = entry['start']
        audio_path = os.path.join(self.audio_dir, audio_file)

        waveform, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=self.clip_samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        clip = waveform.squeeze(0)

        annotations = self._load_annotations(audio_file)
        clip_start = start / self.sample_rate
        clip_end = clip_start + self.clip_duration
        if self.label_mode == 'binary':
            label = self._has_frog_call(annotations, clip_start, clip_end)
        elif self.label_mode == 'count':
            weights = self.get_frog_call_weights(annotations, clip_start, clip_end)
            label = self.soft_count_distribution(weights)  # shape: [5]
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
        
        if isinstance(label, torch.Tensor):
            label = label.clone().detach().float()
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return clip.float(), label, audio_file, start

class FrogCallInferenceDataset(Dataset):
    def __init__(self, audio_dir, clip_duration=3.0, sample_rate=16000):
        self.audio_dir = audio_dir
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.clip_samples = int(clip_duration * sample_rate)

        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.index_map = self._build_index()

    def _build_index(self):
        index = []
        for audio_file in self.audio_files:
            audio_path = os.path.join(self.audio_dir, audio_file)
            try:
                info = torchaudio.info(audio_path)
                total_samples = info.num_frames
                num_clips = total_samples // self.clip_samples
                for i in range(num_clips):
                    start = i * self.clip_samples
                    index.append((audio_file, start))
            except Exception as e:
                print(f"Skipping {audio_file}: {e}")
        return index

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        audio_file, start = self.index_map[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=self.clip_samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        clip = waveform.squeeze(0)
        start_time_sec = start / self.sample_rate
        return clip.float(), audio_file, start_time_sec

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import librosa
# import pandas as pd
# import random

# class FrogCallDataset(Dataset):
#     def __init__(self, audio_dir, annotation_dir, clip_duration=3.0, sample_rate=16000, train=True, test_split=0.2, in_memory=True, pos_ratio=0.5, random_seed=None):
#         self.audio_dir = audio_dir
#         self.annotation_dir = annotation_dir
#         self.clip_duration = clip_duration
#         self.sample_rate = sample_rate
#         self.in_memory = in_memory
#         self.pos_ratio = pos_ratio

#         if random_seed is not None:
#             random.seed(random_seed)
#             np.random.seed(random_seed)
#         self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
#         random.shuffle(self.audio_files)
#         split_idx = int(len(self.audio_files) * (1 - test_split))
#         self.audio_files = self.audio_files[:split_idx] if train else self.audio_files[split_idx:]

#         # Precompute all possible clips and their labels
#         self.positive_clips = []
#         self.negative_clips = []

#         if self.in_memory:
#             self.memory_data = []
#             for audio_file in self.audio_files:
#                 audio, annotations = self._load_audio_and_annotations(audio_file)
#                 self.memory_data.append({'audio': audio, 'annotations': annotations, 'audio_file': audio_file})

#                 # Precompute clips for this audio
#                 total_samples = len(audio)
#                 clip_samples = int(self.clip_duration * self.sample_rate)
#                 # for start in range(0, total_samples - clip_samples + 1, clip_samples):
#                 for start in range(0, total_samples - clip_samples + 1, 1):
#                     clip_start = start / self.sample_rate
#                     clip_end = clip_start + self.clip_duration
#                     label = self._has_frog_call(annotations, clip_start, clip_end)
#                     entry = {'audio_idx': len(self.memory_data) - 1, 'start': start}
#                     if label > 0:
#                         self.positive_clips.append(entry)
#                     else:
#                         self.negative_clips.append(entry)
#         else:
#             for audio_idx, audio_file in enumerate(self.audio_files):
#                 audio_path = os.path.join(self.audio_dir, audio_file)
#                 audio, _ = librosa.load(audio_path, sr=self.sample_rate)
#                 annotations = self._load_annotations(audio_file)
#                 total_samples = len(audio)
#                 clip_samples = int(self.clip_duration * self.sample_rate)
#                 # for start in range(0, total_samples - clip_samples + 1, clip_samples):
#                 for start in range(0, total_samples - clip_samples + 1, 1):
#                     clip_start = start / self.sample_rate
#                     clip_end = clip_start + self.clip_duration
#                     label = self._has_frog_call(annotations, clip_start, clip_end)
#                     entry = {'audio_idx': audio_idx, 'start': start}
#                     if label > 0:
#                         self.positive_clips.append(entry)
#                     else:
#                         self.negative_clips.append(entry)

#     def _load_audio_and_annotations(self, audio_file):
#         audio_path = os.path.join(self.audio_dir, audio_file)
#         audio, _ = librosa.load(audio_path, sr=self.sample_rate)
#         annotations = self._load_annotations(audio_file)
#         return audio, annotations

#     def _load_annotations(self, audio_file):
#         base = os.path.splitext(audio_file)[0]
#         ann_filename = f"{base}.Table.1.selections.txt"
#         ann_path = os.path.join(self.annotation_dir, ann_filename)
#         if not os.path.exists(ann_path):
#             # Use the correct column name for empty DataFrame
#             return pd.DataFrame(columns=['Annotation', 'Begin Time (s)', 'End Time (s)'])
#         df = pd.read_csv(ann_path, sep='\t')
#         # Use the correct column name for annotation
#         df['Annotation'] = df['Annotation'].astype(str).str.strip().str.lower()
#         return df

#     def _sample_clip(self, audio, start):
#         clip_samples = int(self.clip_duration * self.sample_rate)
#         return audio[start:start+clip_samples], start / self.sample_rate

#     def _has_frog_call(self, annotations, clip_start, clip_end):
#         max_overlap = 0
#         confidence = 1
#         for _, row in annotations.iterrows():
#             if 'white dot' in row['Annotation']:
#                 ann_start = float(row['Begin Time (s)'])
#                 ann_end = float(row['End Time (s)'])
#                 overlap_start = max(clip_start, ann_start)
#                 overlap_end = min(clip_end, ann_end)
#                 overlap = max(0, overlap_end - overlap_start)
#                 fraction = overlap / (clip_end - clip_start)
#                 if fraction > max_overlap:
#                     max_overlap = fraction
#                     confidence = 0.75 if 'q2' in row['Annotation'] else 1
#         # If there is overlap, scale label by fraction and confidence
#         return max_overlap * confidence

#     def __len__(self):
#         # Return total number of possible clips (for infinite sampling, you could return a large number)
#         return len(self.positive_clips) + len(self.negative_clips)
#         # return 10000

#     def __getitem__(self, idx):
#         # ...existing logic to select entry...
#         # # Instead of random sampling, use idx to select from a combined list:
#         # if idx < len(self.positive_clips):
#         #     entry = self.positive_clips[idx]
#         #     label = 1 if 'q2' not in entry.get('annotation_label', '') else 0.5
#         # else:
#         #     entry = self.negative_clips[idx - len(self.positive_clips)]
#         #     label = 0
#         # Randomly sample according to pos_ratio
#         if np.random.rand() < self.pos_ratio and len(self.positive_clips) > 0:
#             entry = random.choice(self.positive_clips)
#             label = 1 if 'q2' not in entry.get('annotation_label', '') else 0.5
#         else:
#             entry = random.choice(self.negative_clips)
#             label = 0

#         audio_idx = entry['audio_idx']
#         start = entry['start']
#         audio_file = self.audio_files[audio_idx]
#         if self.in_memory:
#             audio = self.memory_data[audio_idx]['audio']
#         else:
#             audio, _ = librosa.load(os.path.join(self.audio_dir, audio_file), sr=self.sample_rate)
#         clip, _ = self._sample_clip(audio, start)
#         return torch.tensor(clip, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), audio_file, start