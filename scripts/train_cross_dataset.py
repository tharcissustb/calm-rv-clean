"""
Cross-Dataset Validation for Paper 2
Train on one dataset, test on another to prove generalization

Supported datasets:
- pheme: Multi-event thread-based rumour verification
- rumoureval17: Multi-event thread-based (smaller)
- liar: Standalone claim verification (large)
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

from src.training.contrastive_loss import ContrastiveAlignmentLoss
from src.eval.metrics import classification_metrics

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
                if "text" in row and "label" in row:
                    rows.append(row)
            except:
                continue
    return rows


class RumourDataset(Dataset):
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
        item["labels"] = torch.tensor(LABEL_MAP.get(row["label"], 0))
        item["id"] = row.get("id", str(idx))
        item["event_id"] = row.get("event_id", "unknown")
        item["text"] = row["text"]
        return item


def get_representations(model, input_ids, attention_mask):
    outputs = model.roberta(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
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
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower()
    return " ".join(words)


def train_and_evaluate(train_rows, test_rows, test_name, args, cfg, device):
    """Train on train_rows, evaluate on test_rows"""
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)
    
    # Split train into train/val
    random.shuffle(train_rows)
    val_size = int(0.1 * len(train_rows))
    val_rows = train_rows[:val_size]
    train_rows = train_rows[val_size:]
    
    # Load augmented data if requested
    augmented_rows = []
    if args.use_augmented:
        aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
        if aug_path.exists():
            augmented_rows = load_jsonl(aug_path)
            if augmented_rows:
                train_rows = train_rows + augmented_rows
                print(f"  Added {len(augmented_rows)} augmented examples")
    
    # Create datasets
    max_len = cfg["model"].get("max_length", 128)
    train_dataset = RumourDataset(train_rows, tokenizer, max_len)
    val_dataset = RumourDataset(val_rows, tokenizer, max_len)
    test_dataset = RumourDataset(test_rows, tokenizer, max_len)
    
    batch_size = cfg["training"].get("batch_size", 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer
    lr_value = cfg["training"]["lr"]
    if isinstance(lr_value, str):
        lr_value = float(lr_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_value)
    
    epochs = cfg["training"].get("epochs", 3)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            texts = batch["text"]
            event_ids = batch["event_id"]
            labels = batch["labels"].to(device)
            
            input_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss
            
            reps = get_representations(model, input_batch["input_ids"], input_batch["attention_mask"])
            
            if args.use_align:
                align_loss = contrastive_loss_fn(reps, labels, event_ids)
                loss = loss + args.lambda_align * align_loss
            
            if args.use_calibration:
                cal_loss = brier_loss(outputs.logits, labels)
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
        
        # Validation
        model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                outputs = model(**input_batch)
                logits = outputs.logits.cpu().numpy()
                all_labels.extend(labels)
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        metrics = classification_metrics(all_logits, np.array(all_labels))
        
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
            labels = batch["labels"].numpy()
            input_batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**input_batch)
            logits = outputs.logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    metrics = classification_metrics(all_logits, np.array(all_labels))
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--train_dataset", required=True, choices=["pheme", "rumoureval17", "liar"])
    parser.add_argument("--test_dataset", required=True, choices=["pheme", "rumoureval17", "liar"])
    parser.add_argument("--use_align", action="store_true")
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--use_robust", action="store_true")
    parser.add_argument("--use_augmented", action="store_true")
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_cal", type=float, default=0.5)
    parser.add_argument("--lambda_robust", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["training"]["seed"])
    
    # Map dataset names to file paths
    dataset_paths = {
        "pheme": "data/pheme.jsonl",
        "rumoureval17": "data/rumoureval17.jsonl",
        "liar": "data/liar.jsonl",
    }
    
    print("="*60)
    print(f"CROSS-DATASET VALIDATION: {args.exp_name}")
    print(f"Train on: {args.train_dataset.upper()}")
    print(f"Test on: {args.test_dataset.upper()}")
    print(f"  Align: {args.use_align} (λ={args.lambda_align})")
    print(f"  Calibration: {args.use_calibration} (λ={args.lambda_cal})")
    print(f"  Robust: {args.use_robust} (λ={args.lambda_robust})")
    print(f"  Augmented: {args.use_augmented}")
    print("="*60)
    
    # Load data
    train_path = ROOT / dataset_paths[args.train_dataset]
    test_path = ROOT / dataset_paths[args.test_dataset]
    
    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)
    
    print(f"Train size ({args.train_dataset}): {len(train_rows)}")
    print(f"Test size ({args.test_dataset}): {len(test_rows)}")
    
    # Run cross-dataset validation
    results = train_and_evaluate(train_rows, test_rows, args.test_dataset, args, cfg, device)
    
    print("\n" + "="*60)
    print(f"CROSS-DATASET RESULTS: Train on {args.train_dataset.upper()} → Test on {args.test_dataset.upper()}")
    print("="*60)
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    
    # Save results
    output_dir = ROOT / "outputs/cross_dataset" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()