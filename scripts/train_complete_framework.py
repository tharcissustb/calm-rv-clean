"""
Complete Training Framework for Paper 2
Includes: L_clean, L_robust, L_cal, L_align + Augmented Data
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
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(force_cpu: bool):
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def simple_perturb(text, p=0.05):
    """Lightweight perturbation for robustness"""
    words = text.split()
    if len(words) < 3:
        return text
    if random.random() < p:
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower()
    return " ".join(words)


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
    """Extract [CLS] token representations for contrastive loss"""
    outputs = model.roberta(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    return outputs.last_hidden_state[:, 0, :]


def brier_loss(logits, labels, num_classes=3):
    """Multi-class Brier score"""
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=3).float()
    return torch.mean((probs - one_hot) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--use_align", action="store_true", help="Use contrastive alignment")
    parser.add_argument("--use_calibration", action="store_true", help="Use calibration loss")
    parser.add_argument("--use_robust", action="store_true", help="Use robustness loss")
    parser.add_argument("--use_augmented", action="store_true", help="Use augmented hard data")
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_cal", type=float, default=0.5)
    parser.add_argument("--lambda_robust", type=float, default=0.5)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    
    set_seed(cfg["training"]["seed"])
    device = choose_device(args.force_cpu)
    
    print("="*60)
    print("COMPLETE FRAMEWORK TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Align loss: {args.use_align} (λ={args.lambda_align})")
    print(f"Calibration loss: {args.use_calibration} (λ={args.lambda_cal})")
    print(f"Robust loss: {args.use_robust} (λ={args.lambda_robust})")
    print(f"Augmented data: {args.use_augmented}")
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
            print("Warning: Augmented data not found. Run augment_hard_events.py first.")
    
    # Split original data
    random.shuffle(rows)
    train_rows = rows[:int(0.7 * len(rows))]
    val_rows = rows[int(0.7 * len(rows)):int(0.8 * len(rows))]
    test_rows = rows[int(0.8 * len(rows)):]
    
    # Add augmented data to training set
    if args.use_augmented and augmented_rows:
        train_rows = train_rows + augmented_rows
        print(f"Training set size: {len(train_rows)} (original: {len(train_rows)-len(augmented_rows)}, augmented: {len(augmented_rows)})")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)
    
    # Create datasets
    train_dataset = RumourDataset(train_rows, tokenizer, cfg["model"]["max_length"])
    val_dataset = RumourDataset(val_rows, tokenizer, cfg["model"]["max_length"])
    test_dataset = RumourDataset(test_rows, tokenizer, cfg["model"]["max_length"])
    
   # Use smaller batch size on CPU
   batch_size = 8 if args.force_cpu else cfg["training"]["batch_size"]
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size)
   test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    
    # Training loop
    best_f1 = -1
    best_state = None
    
    from src.eval.evaluator import evaluate_and_save
    
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            texts = batch.pop("text")
            event_ids = batch.pop("event_id")
            ids = batch.pop("id")
            labels = batch.pop("labels").to(device)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch, labels=labels)
            loss = outputs.loss  # L_clean
            
            # Get representations for contrastive loss
            representations = get_representations(model, batch["input_ids"], batch["attention_mask"])
            
            # L_align: Contrastive alignment
            if args.use_align:
                align_loss = contrastive_loss_fn(representations, labels, event_ids)
                loss = loss + args.lambda_align * align_loss
            
            # L_cal: Brier score calibration
            if args.use_calibration:
                cal_loss = brier_loss(outputs.logits, labels)
                loss = loss + args.lambda_cal * cal_loss
            
            # L_robust: Perturbation robustness
            if args.use_robust:
                perturbed_texts = [simple_perturb(t, p=0.05) for t in texts]
                perturbed_enc = tokenizer(
                    perturbed_texts,
                    padding=True,
                    truncation=True,
                    max_length=cfg["model"]["max_length"],
                    return_tensors="pt"
                ).to(device)
                perturbed_outputs = model(**perturbed_enc)
                robust_loss = F.cross_entropy(perturbed_outputs.logits, labels)
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
        model.eval()
        all_ids, all_y_true, all_logits = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("text")
                batch.pop("event_id")
                batch_ids = batch.pop("id")
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits.detach().cpu().numpy()
                all_ids.extend(batch_ids)
                all_y_true.extend(labels.numpy())
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        
        # Save results
        output_dir = ROOT / "outputs/runs" / f"{args.exp_name}_seed{cfg['training']['seed']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = evaluate_and_save(output_dir, all_ids, all_y_true, all_logits, f"val_epoch{epoch+1}")
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={results['macro_f1']:.4f}, val_ece={results['ece']:.4f}")
        
        if results["macro_f1"] > best_f1:
            best_f1 = results["macro_f1"]
            best_state = model.state_dict().copy()
    
    # Test best model
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_ids, all_y_true, all_logits = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("text")
            batch.pop("event_id")
            batch_ids = batch.pop("id")
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_ids.extend(batch_ids)
            all_y_true.extend(labels.numpy())
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    test_results = evaluate_and_save(output_dir, all_ids, all_y_true, all_logits, "test")
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Macro-F1: {test_results['macro_f1']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"ECE: {test_results['ece']:.4f}")
    print(f"Brier: {test_results['brier']:.4f}")
    print("="*60)
    
    # Save summary
    summary = {
        "exp_name": args.exp_name,
        "use_align": args.use_align,
        "use_calibration": args.use_calibration,
        "use_robust": args.use_robust,
        "use_augmented": args.use_augmented,
        "lambda_align": args.lambda_align,
        "lambda_cal": args.lambda_cal,
        "lambda_robust": args.lambda_robust,
        "test_macro_f1": test_results["macro_f1"],
        "test_accuracy": test_results["accuracy"],
        "test_ece": test_results["ece"],
        "test_brier": test_results["brier"],
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()