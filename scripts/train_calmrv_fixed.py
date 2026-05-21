"""
CALM-RV Training with Proper Model Saving (FIXED)
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
        item["event_id"] = row.get("event_id", "unknown")
        item["text"] = row["text"]
        return item


def get_representations(model, input_ids, attention_mask):
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
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower()
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    
    # Load data
    data_path = ROOT / cfg["dataset"]["path"]
    rows = load_jsonl(data_path)
    print(f"Loaded {len(rows)} examples")
    
    # Split
    random.shuffle(rows)
    train_rows = rows[:int(0.7 * len(rows))]
    val_rows = rows[int(0.7 * len(rows)):int(0.8 * len(rows))]
    test_rows = rows[int(0.8 * len(rows)):]
    
    # Load augmented data
    aug_path = ROOT / "data/augmented/hard_augmented.jsonl"
    if aug_path.exists():
        augmented_rows = load_jsonl(aug_path)
        train_rows = train_rows + augmented_rows
        print(f"Added {len(augmented_rows)} augmented examples")
    
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    
    # Initialize model
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
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            event_ids = batch.pop("event_id", [])
            texts = batch.pop("text", [])
            labels = batch["labels"].to(device)
            
            # FIX: Remove labels from input_batch before passing to model
            input_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'to') and k != 'labels':
                    input_batch[k] = v.to(device)
            
            optimizer.zero_grad()
            outputs = model(**input_batch, labels=labels)
            loss = outputs.loss
            
            reps = get_representations(model, input_batch["input_ids"], input_batch["attention_mask"])
            align_loss = contrastive_loss_fn(reps, labels, event_ids)
            loss = loss + 0.5 * align_loss
            
            cal_loss = brier_loss(outputs.logits, labels)
            loss = loss + 0.5 * cal_loss
            
            perturbed = [simple_perturb(t) for t in texts]
            pert_enc = tokenizer(perturbed, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            pert_input = {k: v.to(device) for k, v in pert_enc.items() if hasattr(v, 'to')}
            pert_outputs = model(**pert_input)
            robust_loss = F.cross_entropy(pert_outputs.logits, labels)
            loss = loss + 0.3 * robust_loss
            
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
                batch.pop("event_id", None)
                batch.pop("text", None)
                labels_val = batch["labels"].numpy()
                input_batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to') and k != 'labels'}
                logits = model(**input_batch).logits.cpu().numpy()
                all_labels.extend(labels_val)
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        from src.eval.metrics import classification_metrics
        val_f1 = classification_metrics(all_logits, np.array(all_labels))['macro_f1']
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_f1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # SAVE MODEL IN HUGGINGFACE FORMAT
    output_dir = ROOT / f"outputs/models/{args.exp_name}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Model saved to: {output_dir}")
    print(f"   Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()