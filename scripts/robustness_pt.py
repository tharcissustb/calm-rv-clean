"""
Robustness Evaluation for PyTorch Saved Models (.pt files)
"""

import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_from_pt(model_path, base_model_name="roberta-base", num_labels=3):
    """Load PyTorch model from .pt file"""
    # Create model with same architecture
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle different state dict keys (may have 'model.' prefix)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


def char_noise(text, noise_rate=0.05):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_rate and chars[i].isalpha():
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def word_dropout(text, drop_rate=0.1):
    words = text.split()
    if len(words) < 2:
        return text
    words = [w for w in words if random.random() > drop_rate]
    return " ".join(words) if words else text


def url_remove(text):
    import re
    return re.sub(r'https?://\S+|www\.\S+', '[URL]', text)


def mention_remove(text):
    import re
    return re.sub(r'@\w+', '[USER]', text)


def hashtag_remove(text):
    import re
    return re.sub(r'#\w+', '[HASHTAG]', text)


def all_perturbations(text):
    text = url_remove(text)
    text = mention_remove(text)
    text = hashtag_remove(text)
    text = char_noise(text, noise_rate=0.03)
    text = word_dropout(text, drop_rate=0.05)
    return text


def load_test_data(max_samples=200):
    """Load test data from PHEME (ferguson event)"""
    data_path = ROOT / "data/pheme.jsonl"
    texts, labels = [], []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('event_id') == 'ferguson':
                texts.append(row['text'])
                labels.append(LABEL_MAP[row['label']])
    
    # Sample for speed
    if len(texts) > max_samples:
        indices = random.sample(range(len(texts)), max_samples)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return texts, labels


def evaluate_robustness(model, tokenizer, test_texts, test_labels, device):
    """Evaluate model under perturbations"""
    model.eval()
    
    perturbations = [
        ("Clean (no perturbation)", lambda x: x),
        ("URL Removal", lambda x: url_remove(x)),
        ("Mention Removal", lambda x: mention_remove(x)),
        ("Hashtag Removal", lambda x: hashtag_remove(x)),
        ("Character Noise (5%)", lambda x: char_noise(x, 0.05)),
        ("Word Dropout (10%)", lambda x: word_dropout(x, 0.10)),
        ("All Combined", lambda x: all_perturbations(x)),
    ]
    
    results = {}
    
    for pert_name, pert_fn in perturbations:
        correct = 0
        for text, true_label in zip(test_texts, test_labels):
            perturbed = pert_fn(text)
            inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.argmax().item()
            if pred == true_label:
                correct += 1
        accuracy = correct / len(test_texts)
        results[pert_name] = accuracy
        print(f"{pert_name}: {accuracy:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", required=True, help="Path to .pt model file")
    parser.add_argument("--base_model", default="roberta-base")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_pt}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = load_model_from_pt(args.model_pt, args.base_model, num_labels=3)
    model.to(device)
    
    # Load test data
    test_texts, test_labels = load_test_data(max_samples=200)
    print(f"Loaded {len(test_texts)} test samples from ferguson event")
    
    # Evaluate
    results = evaluate_robustness(model, tokenizer, test_texts, test_labels, device)
    
    print("\n" + "="*50)
    print("ROBUSTNESS SUMMARY")
    print("="*50)
    for pert, acc in results.items():
        print(f"{pert}: {acc:.4f}")


if __name__ == "__main__":
    main()