"""
Fast Temporal Evaluation for CALM-RV
Train on early events, test on later events
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
from src.eval.metrics import classification_metrics

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}

# Events in chronological order
TRAIN_EVENTS = ['gurlitt', 'ebola-essien', 'ferguson', 'ottawashooting', 'sydneysiege', 'charliehebdo']
TEST_EVENTS = ['germanwings-crash', 'putinmissing', 'prince-toronto']


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


class SimpleDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(row['text'], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(LABEL_MAP[row['label']])
        }


def train_and_eval(train_rows, test_rows, args, cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model']['name'], num_labels=3)
    model.to(device)
    
    # Simple train/val split
    random.shuffle(train_rows)
    val_rows = train_rows[:int(0.1 * len(train_rows))]
    train_rows = train_rows[int(0.1 * len(train_rows)):]
    
    train_loader = DataLoader(SimpleDataset(train_rows, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(SimpleDataset(val_rows, tokenizer), batch_size=16)
    test_loader = DataLoader(SimpleDataset(test_rows, tokenizer), batch_size=16)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_f1 = -1
    best_state = None
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = batch['labels'].to(device)
            input_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['labels'].numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                logits = model(**input_batch).logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)
        all_logits = np.vstack(all_logits)
        val_f1 = classification_metrics(all_logits, np.array(all_labels))['macro_f1']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch['labels'].numpy()
            input_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            logits = model(**input_batch).logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    all_logits = np.vstack(all_logits)
    return classification_metrics(all_logits, np.array(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--use_augmented', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cpu') if args.force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    cfg = yaml.safe_load(open(args.config))
    data_path = ROOT / cfg['dataset']['path']
    rows = load_jsonl(data_path)
    
    # Split by event
    by_event = {}
    for r in rows:
        by_event.setdefault(r['event_id'], []).append(r)
    
    train_rows = []
    for e in TRAIN_EVENTS:
        if e in by_event:
            train_rows.extend(by_event[e])
    
    test_rows = []
    for e in TEST_EVENTS:
        if e in by_event:
            test_rows.extend(by_event[e])
    
    # Add augmented data if requested
    if args.use_augmented:
        aug_path = ROOT / 'data/augmented/hard_augmented.jsonl'
        if aug_path.exists():
            train_rows += load_jsonl(aug_path)
    
    print(f'Train events: {TRAIN_EVENTS}')
    print(f'Test events: {TEST_EVENTS}')
    print(f'Train size: {len(train_rows)}, Test size: {len(test_rows)}')
    
    results = train_and_eval(train_rows, test_rows, args, cfg, device)
    
    print('\n' + '='*50)
    print('TEMPORAL EVALUATION RESULTS')
    print('='*50)
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ECE: {results['ece']:.4f}")


if __name__ == '__main__':
    main()