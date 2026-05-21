"""
LIAR Ablation with CALM-RV components
Supports --use_align, --use_calibration, --use_robust, --use_augmented
"""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.contrastive_loss import ContrastiveAlignmentLoss
from src.eval.metrics import classification_metrics

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                if 'text' in row and 'label' in row:
                    rows.append(row)
            except:
                continue
    return rows

class LiarDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(row['text'], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(LABEL_MAP[row['label']])
        item['id'] = row.get('id', str(idx))
        # Add dummy event_id for alignment (all same, so alignment does nothing)
        item['event_id'] = 'liar'
        item['text'] = row['text']
        return item

def get_representations(model, input_ids, attention_mask):
    # For RoBERTa
    if hasattr(model, 'roberta'):
        outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return outputs.last_hidden_state[:, 0, :]

def brier_loss(logits, labels, num_classes=3):
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes).float()
    return torch.mean((probs - one_hot) ** 2)

def simple_perturb(text, p=0.05):
    words = text.split()
    if len(words) < 3:
        return text
    if random.random() < p:
        idx = random.randint(0, len(words)-1)
        words[idx] = words[idx].lower()
    return ' '.join(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--use_align', action='store_true')
    parser.add_argument('--use_calibration', action='store_true')
    parser.add_argument('--use_robust', action='store_true')
    parser.add_argument('--use_augmented', action='store_true')
    parser.add_argument('--lambda_align', type=float, default=0.5)
    parser.add_argument('--lambda_cal', type=float, default=0.5)
    parser.add_argument('--lambda_robust', type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu') if args.force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cfg = yaml.safe_load(open(args.config, 'r'))
    data_path = ROOT / cfg['dataset']['path']
    rows = load_jsonl(data_path)
    print(f'Loaded {len(rows)} examples')

    # Load augmented data if requested
    if args.use_augmented:
        aug_path = ROOT / 'data/augmented/liar_augmented.jsonl'
        if not aug_path.exists():
            # Generate LIAR augmented data (optional)
            print('Augmented file not found, skipping augmentation')
        else:
            aug_rows = load_jsonl(aug_path)
            rows = rows + aug_rows
            print(f'Added {len(aug_rows)} augmented examples')

    random.shuffle(rows)
    train_rows = rows[:int(0.8 * len(rows))]
    val_rows = rows[int(0.8 * len(rows)):int(0.9 * len(rows))]
    test_rows = rows[int(0.9 * len(rows)):]

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model']['name'], num_labels=3)
    model.to(device)

    max_len = cfg['model'].get('max_length', 128)
    train_ds = LiarDataset(train_rows, tokenizer, max_len)
    val_ds = LiarDataset(val_rows, tokenizer, max_len)
    test_ds = LiarDataset(test_rows, tokenizer, max_len)

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    lr = float(cfg['training']['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epochs = cfg['training']['epochs']
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)

    best_f1 = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            texts = batch.pop('text', [])
            event_ids = batch.pop('event_id', [])
            batch.pop('id', None)
            labels = batch.pop('labels').to(device)
            input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss

            if args.use_align:
                reps = get_representations(model, input_batch['input_ids'], input_batch['attention_mask'])
                align_loss = contrastive_loss_fn(reps, labels, event_ids)
                loss += args.lambda_align * align_loss

            if args.use_calibration:
                cal_loss = brier_loss(outputs.logits, labels)
                loss += args.lambda_cal * cal_loss

            if args.use_robust:
                perturbed = [simple_perturb(t) for t in texts]
                pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
                pert_input = {k: v.to(device) for k, v in pert_enc.items() if hasattr(v, 'to')}
                pert_outputs = model(**pert_input)
                robust_loss = F.cross_entropy(pert_outputs.logits, labels)
                loss += args.lambda_robust * robust_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch.pop('id', None)
                batch.pop('text', None)
                batch.pop('event_id', None)
                labels = batch.pop('labels').numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
                logits = model(**input_batch).logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)
        all_logits = np.vstack(all_logits)
        val_metrics = classification_metrics(all_logits, np.array(all_labels))
        print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={val_metrics["macro_f1"]:.4f}')

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop('id', None)
            batch.pop('text', None)
            batch.pop('event_id', None)
            labels = batch.pop('labels').numpy()
            input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            logits = model(**input_batch).logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    all_logits = np.vstack(all_logits)
    test_metrics = classification_metrics(all_logits, np.array(all_labels))

    print('\n' + '='*60)
    print(f'LIAR Ablation: {args.exp_name}')
    print('='*60)
    print(f'Macro-F1: {test_metrics["macro_f1"]:.4f}')
    print(f'Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'ECE: {test_metrics["ece"]:.4f}')

    out_dir = ROOT / 'outputs/liar_ablation' / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == '__main__':
    main()