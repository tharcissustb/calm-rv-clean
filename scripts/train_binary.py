"""
Complete Binary Training Script for CALM-RV
Supports --force_cpu flag to avoid CUDA errors
Works for Twitter15/16, Weibo, Chinese_Rumor_Dataset
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import classification_metrics

# Binary label mapping (2 classes)
LABEL_MAP = {"true": 0, "false": 1}


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
                if "text" in row and "label" in row:
                    rows.append(row)
            except:
                continue
    return rows


class BinaryDataset(Dataset):
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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        
        # Handle label: convert string to int, default to false if unknown
        label_str = row["label"].lower()
        if label_str == "true" or label_str == "rumor" or label_str == "1":
            label_val = 0
        else:
            label_val = 1
        
        item["labels"] = torch.tensor(label_val, dtype=torch.long)
        item["id"] = row.get("id", str(idx))
        item["text"] = row["text"]
        return item


def brier_loss_binary(logits, labels, num_classes=2):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--use_calibration", action="store_true", help="Use calibration loss")
    parser.add_argument("--use_robust", action="store_true", help="Use robustness loss")
    parser.add_argument("--use_augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--lambda_cal", type=float, default=0.5, help="Calibration loss weight")
    parser.add_argument("--lambda_robust", type=float, default=0.5, help="Robustness loss weight")
    args = parser.parse_args()
    
    # Set device
    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["training"]["seed"])
    
    print("="*60)
    print(f"BINARY EXPERIMENT: {args.exp_name}")
    print(f"  Calibration: {args.use_calibration} (λ={args.lambda_cal})")
    print(f"  Robust: {args.use_robust} (λ={args.lambda_robust})")
    print(f"  Augmented: {args.use_augmented}")
    print("="*60)
    
    # Load data
    data_path = ROOT / cfg["dataset"]["path"]
    rows = load_jsonl(data_path)
    print(f"Loaded {len(rows)} examples")
    
    # Load augmented data if requested
    if args.use_augmented:
        aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
        if aug_path.exists():
            augmented_rows = load_jsonl(aug_path)
            rows = rows + augmented_rows
            print(f"Added {len(augmented_rows)} augmented examples")
            print(f"Total training examples: {len(rows)}")
    
    # Split data (80/10/10)
    random.shuffle(rows)
    train_rows = rows[:int(0.8 * len(rows))]
    val_rows = rows[int(0.8 * len(rows)):int(0.9 * len(rows))]
    test_rows = rows[int(0.9 * len(rows)):]
    
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")
    
    # Initialize tokenizer and model
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)
    
    # Create datasets
    max_len = cfg["model"].get("max_length", 128)
    train_dataset = BinaryDataset(train_rows, tokenizer, max_len)
    val_dataset = BinaryDataset(val_rows, tokenizer, max_len)
    test_dataset = BinaryDataset(test_rows, tokenizer, max_len)
    
    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer
    lr_value = cfg["training"]["lr"]
    if isinstance(lr_value, str):
        lr_value = float(lr_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_value)
    
    epochs = cfg["training"].get("epochs", 5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_f1 = -1
    best_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            texts = batch["text"]
            batch.pop("id")
            labels = batch["labels"].to(device)
            
            input_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss
            
            if args.use_calibration:
                cal_loss = brier_loss_binary(outputs.logits, labels, num_classes=2)
                loss = loss + args.lambda_cal * cal_loss
            
            if args.use_robust:
                perturbed = [simple_perturb(t) for t in texts]
                pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                pert_enc = {k: v.to(device) for k, v in pert_enc.items()}
                pert_outputs = model(**pert_enc)
                robust_loss = F.cross_entropy(pert_outputs.logits, labels)
                loss = loss + args.lambda_robust * robust_loss
            
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
                batch.pop("id")
                batch.pop("text")
                labels = batch["labels"].numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                outputs = model(**input_batch)
                logits = outputs.logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        metrics = classification_metrics(all_logits, np.array(all_labels))
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={metrics['macro_f1']:.4f}, val_ece={metrics['ece']:.4f}")
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Test best model
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("id")
            batch.pop("text")
            labels = batch["labels"].numpy()
            input_batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**input_batch)
            logits = outputs.logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    metrics = classification_metrics(all_logits, np.array(all_labels))
    
    print("\n" + "="*60)
    print(f"RESULTS FOR {cfg['dataset']['name'].upper()}")
    print("="*60)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")
    
    # Save results
    output_dir = ROOT / "outputs/runs" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()