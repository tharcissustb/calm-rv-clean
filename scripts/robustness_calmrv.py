"""
Robustness Evaluation for CALM-RV Model (HuggingFace Format)
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}


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
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def mention_remove(text):
    import re
    return re.sub(r'@\w+', '', text)


def hashtag_remove(text):
    import re
    return re.sub(r'#\w+', '', text)


def load_test_data(max_samples=200):
    data_path = ROOT / "data/pheme.jsonl"
    texts, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('event_id') == 'ferguson':
                texts.append(row['text'])
                labels.append(LABEL_MAP[row['label']])
    
    if len(texts) > max_samples:
        indices = random.sample(range(len(texts)), max_samples)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    return texts, labels


def evaluate(model_path, base_model="roberta-base"):
    device = torch.device("cpu")
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    texts, labels = load_test_data(max_samples=200)
    print(f"Loaded {len(texts)} test samples")
    
    perturbations = [
        ("Clean", lambda x: x),
        ("URL Removal", url_remove),
        ("Mention Removal", mention_remove),
        ("Hashtag Removal", hashtag_remove),
        ("Character Noise (5%)", lambda x: char_noise(x, 0.05)),
        ("Word Dropout (10%)", lambda x: word_dropout(x, 0.10)),
        ("All Combined", lambda x: char_noise(word_dropout(hashtag_remove(mention_remove(url_remove(x))), 0.05), 0.03)),
    ]
    
    results = {}
    for name, pert_fn in perturbations:
        correct = 0
        for text, label in zip(texts, labels):
            perturbed = pert_fn(text)
            inputs = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                pred = model(**inputs).logits.argmax().item()
            if pred == label:
                correct += 1
        acc = correct / len(texts)
        results[name] = acc
        print(f"{name}: {acc:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    
    results = evaluate(args.model_path)
    
    print("\n" + "="*50)
    print("CALM-RV ROBUSTNESS SUMMARY")
    print("="*50)
    for pert, acc in results.items():
        print(f"{pert}: {acc:.4f}")


if __name__ == "__main__":
    main()

