"""
LOEO Framework for Non-Transformer Models (LSTM, TextCNN)
Tests model-agnostic generalization
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
from transformers import AutoTokenizer

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import classification_metrics

# Import custom models
from src.models.lstm_model import LSTMRumourClassifier
from src.models.cnn_model import TextCNN

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "text" in row and "label" in row and "event_id" in row:
                    rows.append(row)
            except:
                continue
    return rows


class NonTransformerDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_MAP.get(row["label"], 0)),
            "id": row.get("id", str(idx)),
            "event_id": row.get("event_id", "unknown"),
            "text": row["text"]
        }
        return item


def brier_loss(logits, labels, num_classes=3):
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes).float()
    return torch.mean((probs - one_hot) ** 2)


def simple_perturb(text, p=0.05):
    words = text.split()
    if len(words) < 3:
        return text
    if random.random() < p:
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower()
    return " ".join(words)


def get_model(model_name, vocab_size, num_classes):
    if model_name == "lstm":
        return LSTMRumourClassifier(vocab_size=vocab_size, num_classes=num_classes)
    elif model_name == "textcnn":
        return TextCNN(vocab_size=vocab_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(train_rows, test_rows, test_event_name, args, cfg, device):
    # Load tokenizer for vocabulary
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    
    # Create model
    model = get_model(args.model_type, vocab_size, cfg["model"]["num_labels"])
    model.to(device)
    
    # Split train into train/val
    random.shuffle(train_rows)
    val_size = int(0.1 * len(train_rows))
    val_rows = train_rows[:val_size]
    train_rows = train_rows[val_size:]
    
    # Load augmented data if requested
    if args.use_augmented:
        aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
        if aug_path.exists():
            augmented_rows = load_jsonl(aug_path)
            train_rows = train_rows + augmented_rows
            print(f"  Added {len(augmented_rows)} augmented examples")
    
    # Create datasets
    max_len = cfg["model"].get("max_length", 128)
    train_dataset = NonTransformerDataset(train_rows, tokenizer, max_len)
    val_dataset = NonTransformerDataset(val_rows, tokenizer, max_len)
    test_dataset = NonTransformerDataset(test_rows, tokenizer, max_len)
    
    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    lr_value = cfg["training"]["lr"]
    if isinstance(lr_value, str):
        lr_value = float(lr_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_value)
    
    epochs = cfg["training"].get("epochs", 5)
    
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids=input_ids)
            loss = F.cross_entropy(logits, labels)
            
            if args.use_calibration:
                cal_loss = brier_loss(logits, labels)
                loss = loss + args.lambda_cal * cal_loss
            
            if args.use_robust:
                texts = batch["text"]
                perturbed = [simple_perturb(t) for t in texts]
                pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                pert_input_ids = pert_enc["input_ids"].to(device)
                pert_logits = model(input_ids=pert_input_ids)
                robust_loss = F.cross_entropy(pert_logits, labels)
                loss = loss + args.lambda_robust * robust_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy()
                input_ids = batch["input_ids"].to(device)
                logits = model(input_ids=input_ids).cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        metrics = classification_metrics(all_logits, np.array(all_labels))
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].numpy()
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids=input_ids).cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    metrics = classification_metrics(all_logits, np.array(all_labels))
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model_type", required=True, choices=["lstm", "textcnn"])
    parser.add_argument("--test_event", default="ferguson")
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--use_robust", action="store_true")
    parser.add_argument("--use_augmented", action="store_true")
    parser.add_argument("--lambda_cal", type=float, default=0.5)
    parser.add_argument("--lambda_robust", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["training"]["seed"])
    
    print("="*60)
    print(f"NON-TRANSFORMER LOEO: {args.model_type.upper()}")
    print(f"Test event: {args.test_event}")
    print(f"  Calibration: {args.use_calibration}")
    print(f"  Robust: {args.use_robust}")
    print(f"  Augmented: {args.use_augmented}")
    print("="*60)
    
    data_path = ROOT / cfg["dataset"]["path"]
    all_rows = load_jsonl(data_path)
    
    by_event = {}
    for row in all_rows:
        event = row["event_id"]
        if event not in by_event:
            by_event[event] = []
        by_event[event].append(row)
    
    test_rows = by_event.get(args.test_event, [])
    train_rows = []
    for event, rows in by_event.items():
        if event != args.test_event:
            train_rows.extend(rows)
    
    print(f"Train size: {len(train_rows)}")
    print(f"Test size: {len(test_rows)}")
    
    results = train_and_evaluate(train_rows, test_rows, args.test_event, args, cfg, device)
    
    print("\n" + "="*60)
    print(f"{args.model_type.upper()} RESULTS FOR {args.test_event.upper()}")
    print("="*60)
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    
    output_dir = ROOT / "outputs/loeo" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{args.test_event}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()