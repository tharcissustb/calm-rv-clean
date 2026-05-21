"""
Simple Training Script for Paper 2 Framework
Runs on CPU safely without CUDA errors
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
    """Extract [CLS] token representations"""
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


def evaluate(model, loader, device, output_dir, name):
    """Evaluate model and return metrics"""
    from src.eval.metrics import classification_metrics
    
    model.eval()
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy()
            batch_cuda = {k: v.to(device) for k, v in batch.items() if k not in ["id", "text", "event_id", "labels"]}
            outputs = model(**batch_cuda)
            logits = outputs.logits.cpu().numpy()
            all_labels.extend(labels)
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    metrics = classification_metrics(all_logits, np.array(all_labels))
    
    print(f"  {name}: F1={metrics['macro_f1']:.4f}, ECE={metrics['ece']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--use_align", action="store_true")
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--use_robust", action="store_true")
    parser.add_argument("--use_augmented", action="store_true")
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_cal", type=float, default=0.5)
    parser.add_argument("--lambda_robust", type=float, default=0.5)
    args = parser.parse_args()
    
    # Force CPU for stability
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["training"]["seed"])
    
    print("="*60)
    print(f"Experiment: {args.exp_name}")
    print(f"  Align: {args.use_align} (λ={args.lambda_align})")
    print(f"  Calibration: {args.use_calibration} (λ={args.lambda_cal})")
    print(f"  Robust: {args.use_robust} (λ={args.lambda_robust})")
    print(f"  Augmented: {args.use_augmented}")
    print("="*60)
    
    # Load data
    data_path = ROOT / cfg["dataset"]["path"]
    rows = load_jsonl(data_path)
    print(f"Loaded {len(rows)} original examples")
    
    # Load augmented data if requested
    augmented_rows = []
    if args.use_augmented:
        aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
        if aug_path.exists():
            augmented_rows = load_jsonl(aug_path)
            print(f"Loaded {len(augmented_rows)} augmented examples")
        else:
            print("Warning: Augmented data not found")
    
    # Split data
    random.shuffle(rows)
    train_rows = rows[:int(0.7 * len(rows))]
    val_rows = rows[int(0.7 * len(rows)):int(0.8 * len(rows))]
    test_rows = rows[int(0.8 * len(rows)):]
    
    # Add augmented data to training
    if args.use_augmented and augmented_rows:
        train_rows = train_rows + augmented_rows
        print(f"Training set: {len(train_rows)} (original: {len(train_rows)-len(augmented_rows)}, aug: {len(augmented_rows)})")
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)
    
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
    
    # Scheduler
    epochs = cfg["training"].get("epochs", 5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    
    # Training loop
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Prepare batch
            texts = batch["text"]
            event_ids = batch["event_id"]
            labels = batch["labels"].to(device)
            
            # Move inputs to device
            input_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss
            
            # Get representations for contrastive loss
            reps = get_representations(model, input_batch["input_ids"], input_batch["attention_mask"])
            
            # L_align
            if args.use_align:
                align_loss = contrastive_loss_fn(reps, labels, event_ids)
                loss = loss + args.lambda_align * align_loss
            
            # L_calibration
            if args.use_calibration:
                cal_loss = brier_loss(outputs.logits, labels)
                loss = loss + args.lambda_cal * cal_loss
            
            # L_robust
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
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, None, f"Epoch {epoch+1} Val")
        
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best model and test
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, test_loader, device, None, "Test")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"ECE: {test_metrics['ece']:.4f}")
    print(f"Brier: {test_metrics['brier']:.4f}")
    
    # Save results
    output_dir = ROOT / "outputs/runs" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()