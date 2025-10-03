import os
import sys
import argparse
import time
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint

from models import Transfer_Cnn14_16k
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from frogcall_dataset import FrogCallDataset

def save_checkpoint(model, optimizer, iteration, filepath='checkpoints/checkpoint.pt'):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, filepath)

def evaluate(model, val_loader, device, criterion, label_mode='binary'):
    model.eval()
    val_loss = 0.0
    total = 0

    # Binary metrics
    tp = tn = fp = fn = 0

    # Count metrics (optional)
    count_error_sum = 0.0
    kl_div_sum = 0.0

    with torch.no_grad():
        for clips, labels, audio_file, start in val_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            output_dict = model(clips)
            logits = output_dict['clipwise_output']  # shape: [B, 1] or [B, 5]

            if label_mode == 'binary':
                # Binary loss
                loss = criterion(logits.squeeze(-1), labels.float())
                val_loss += loss.item() * clips.size(0)

                # Binary metrics
                binary_labels = (labels >= 0.5).long()
                probs = torch.sigmoid(logits.squeeze(-1))  # shape: [B]
                preds = (probs >= 0.5).long()

                tp += ((preds == 1) & (binary_labels == 1)).sum().item()
                tn += ((preds == 0) & (binary_labels == 0)).sum().item()
                fp += ((preds == 1) & (binary_labels == 0)).sum().item()
                fn += ((preds == 0) & (binary_labels == 1)).sum().item()

            elif label_mode == 'count':
                # Count loss
                log_probs = F.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, labels)
                val_loss += loss.item() * clips.size(0)

                # Expected count error
                bin_indices = torch.arange(labels.size(-1)).to(device)  # [0, 1, 2, 3, 4]
                expected_pred = torch.sum(torch.exp(log_probs) * bin_indices, dim=-1)
                expected_true = torch.sum(labels * bin_indices, dim=-1)
                count_error_sum += torch.abs(expected_pred - expected_true).sum().item()

                # KL divergence (already computed as loss)
                kl_div_sum += loss.item() * clips.size(0)

                # Binary-style metrics from softmax bins
                probs = torch.exp(log_probs)  # shape: [B, 5]
                pred_pos = probs[:, 1:].sum(dim=-1) >= 0.5  # sum of bins 1–4
                true_pos = labels[:, 1:].sum(dim=-1) >= 0.5

                tp += ((pred_pos == 1) & (true_pos == 1)).sum().item()
                tn += ((pred_pos == 0) & (true_pos == 0)).sum().item()
                fp += ((pred_pos == 1) & (true_pos == 0)).sum().item()
                fn += ((pred_pos == 0) & (true_pos == 1)).sum().item()
            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

            total += clips.size(0)

    avg_loss = val_loss / total if total > 0 else 0

    metrics = {'loss': avg_loss, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    if label_mode == 'count':
        metrics.update({
            'expected_count_error': count_error_sum / total if total > 0 else 0,
            'kl_divergence': kl_div_sum / total if total > 0 else 0
        })

    return metrics

def train(args):
    # Arguments & parameters
    audio_dir = args.dataset_dir
    annotation_dir = args.annotation_dir
    workspace = args.workspace
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    learning_rate_decay = args.learning_rate_decay
    batch_size = args.batch_size
    stop_iteration = args.stop_iteration
    label_mode = args.label_mode
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    pos_ratio = getattr(args, 'pos_ratio', 0.5)

    # Model
    classes_num = 1 if label_mode == 'binary' else 5

    Model = eval(model_type)
    model = Model(
        sample_rate=16000, window_size=512, hop_size=160, mel_bins=64,
        fmin=50, fmax=8000, classes_num=classes_num, freeze_base=freeze_base
    )

    if pretrained_checkpoint_path:
        print(f'Loading pretrained model from {pretrained_checkpoint_path}')
        model.load_from_pretrain(pretrained_checkpoint_path)

    model.to(device)

    test_split = 0.05  # Fixed test split for validation during training

    # Dataset & DataLoader
    train_dataset = FrogCallDataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        clip_duration=3.0,
        sample_rate=16000,
        train=True,
        test_split=test_split,
        pos_ratio=pos_ratio, 
        random_seed=42,
        label_mode=label_mode
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_dataset = FrogCallDataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        clip_duration=3.0,
        sample_rate=16000,
        train=False,
        test_split=test_split,
        pos_ratio=pos_ratio, 
        random_seed=42,
        label_mode=label_mode
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model.base.bn0.weight.requires_grad = False
    model.base.bn0.bias.requires_grad = False
    model.base.bn0.eval()  # Prevent running_mean/var updates
    for param in model.base.fc_audioset.parameters():
        param.requires_grad = False

    layer_groups = [
        model.base.conv_block1,
        model.base.conv_block2,
        model.base.conv_block3,
        model.base.conv_block4,
        model.base.conv_block5,
        model.base.conv_block6,
        model.base.fc1,
        model.fc_transfer
    ]

    params = []
    num_layers = len(layer_groups)

    for i, layer in enumerate(layer_groups):
        lr = learning_rate * (learning_rate_decay ** (num_layers - i - 1))
        layer_params = list(layer.parameters())
        if layer_params:  # Only add if non-empty
            for p in layer_params:
                p.requires_grad = True  # Ensure gradients are enabled
            params.append({'params': layer_params, 'lr': lr})

    # Total parameters in the model
    total_model_params = sum(p.numel() for p in model.parameters())

    # Total parameters in optimizer groups
    total_optimizer_params = sum(p.numel() for group in params for p in group['params'])

    print(f"Total model parameters: {total_model_params}")
    print(f"Total optimizer parameters: {total_optimizer_params}")
    
    # Loss & Optimizer
    if label_mode == 'binary':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.9))
    elif label_mode == 'count':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    
    optimizer = optim.Adam(params)

    iteration = 0
    model.train()
    best_val_loss = float('inf')
    while iteration < stop_iteration:
        for clips, labels, audio_file, start in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            output_dict = model(clips)
            logits = output_dict['clipwise_output']  # shape: [B, 1] or [B, 5]

            if label_mode == 'binary':
                # logits: [B, 1], labels: [B]
                loss = criterion(logits.squeeze(-1), labels.float())

            elif label_mode == 'count':
                # logits: [B, 5], labels: [B, 5]
                log_probs = F.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, labels)  # KLDivLoss expects log_probs vs. probs

            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
            # Run validation
            metrics = evaluate(model, val_loader, device, criterion, label_mode=label_mode)
            val_loss = metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, filepath='checkpoints/best_model.pt')
                print(f"Checkpoint saved at iteration {iteration} with val loss {val_loss:.4f}")
            print(f"Validation loss: {metrics['loss']:.4f}")
            if label_mode == 'binary':
                pprint.pprint({k: metrics[k] for k in ['tp', 'tn', 'fp', 'fn', 'loss']})
            elif label_mode == 'count':
                pprint.pprint({k: metrics[k] for k in ['tp', 'tn', 'fp', 'fn', 'expected_count_error', 'kl_divergence', 'loss']})

            iteration += 1
            if iteration >= stop_iteration:
                break

    print("Done Training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train frog call classifier.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser.add_argument('--annotation_dir', type=str, required=True, help='Directory of annotations.')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--pretrained_checkpoint_path', type=str, default=None)
    parser.add_argument('--freeze_base', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--augmentation', type=str, choices=['none', 'mixup'], default='none')
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--learning_rate_decay', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--stop_iteration', type=int, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--pos_ratio', type=float, default=0.5)
    parser.add_argument('--in_memory', action='store_true', default=True)
    parser.add_argument('--label_mode', type=str, choices=['binary', 'count'], default='count')

    args = parser.parse_args()
    train(args)