"""
Simple Adversarial Robustness Evaluation for CALM-RV
Tests character-level and word-level perturbations
Computes Attack Success Rate (ASR) and Accuracy Drop
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

LABEL_MAP = {"true": 0, "false": 1, "unverified": 2}
ID_TO_LABEL = {0: "true", 1: "false", 2: "unverified"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                if "text" in row and "label" in row:
                    rows.append(row)
            except:
                continue
    return rows


# ============================================================
# PERTURBATION FUNCTIONS
# ============================================================

def char_noise(text, noise_rate=0.05):
    """Add random character noise"""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_rate and chars[i].isalpha():
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def word_dropout(text, drop_rate=0.1):
    """Randomly drop words"""
    words = text.split()
    if len(words) < 2:
        return text
    words = [w for w in words if random.random() > drop_rate]
    return " ".join(words) if words else text


def word_shuffle(text, shuffle_rate=0.1):
    """Randomly shuffle adjacent words"""
    words = text.split()
    if len(words) < 3:
        return text
    for i in range(len(words) - 1):
        if random.random() < shuffle_rate:
            words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)


def url_remove(text):
    """Remove URLs from text"""
    import re
    return re.sub(r'https?://\S+|www\.\S+', '[URL]', text)


def mention_remove(text):
    """Remove @mentions"""
    import re
    return re.sub(r'@\w+', '[USER]', text)


def hashtag_remove(text):
    """Remove hashtags"""
    import re
    return re.sub(r'#\w+', '[HASHTAG]', text)


def all_perturbations(text):
    """Apply all perturbations"""
    text = url_remove(text)
    text = mention_remove(text)
    text = hashtag_remove(text)
    text = char_noise(text, noise_rate=0.03)
    text = word_dropout(text, drop_rate=0.05)
    return text


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_robustness(model, tokenizer, test_texts, test_labels, device, max_samples=200):
    """
    Evaluate model under different perturbations
    Returns:
        - Original accuracy
        - Accuracy under each perturbation
        - Attack Success Rate (ASR) for each perturbation
    """
    model.eval()
    
    perturbations = [
        ("Clean (no perturbation)", lambda x: x),
        ("URL Removal", lambda x: url_remove(x)),
        ("Mention Removal", lambda x: mention_remove(x)),
        ("Hashtag Removal", lambda x: hashtag_remove(x)),
        ("Character Noise (5%)", lambda x: char_noise(x, 0.05)),
        ("Word Dropout (10%)", lambda x: word_dropout(x, 0.10)),
        ("Word Shuffle (10%)", lambda x: word_shuffle(x, 0.10)),
        ("All Combined", lambda x: all_perturbations(x)),
    ]
    
    # Limit samples for speed
    if len(test_texts) > max_samples:
        indices = random.sample(range(len(test_texts)), max_samples)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    results = {}
    
    for pert_name, pert_fn in tqdm(perturbations, desc="Evaluating perturbations"):
        correct = 0
        predictions = []
        original_predictions = []
        
        for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
            # Original prediction (clean)
            inputs_clean = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs_clean = model(**inputs_clean)
                pred_clean = outputs_clean.logits.argmax().item()
            
            # Perturbed prediction
            perturbed_text = pert_fn(text)
            inputs_pert = tokenizer(perturbed_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs_pert = model(**inputs_pert)
                pred_pert = outputs_pert.logits.argmax().item()
            
            if pred_pert == true_label:
                correct += 1
            
            original_predictions.append(pred_clean)
            predictions.append(pred_pert)
        
        accuracy = correct / len(test_texts)
        
        # Calculate Attack Success Rate (ASR)
        # ASR = proportion of correctly classified clean examples that become incorrect after perturbation
        correctly_classified_clean = sum(1 for i, p in enumerate(original_predictions) if p == test_labels[i])
        flipped_wrong = sum(1 for i, p in enumerate(predictions) if original_predictions[i] == test_labels[i] and p != test_labels[i])
        asr = flipped_wrong / correctly_classified_clean if correctly_classified_clean > 0 else 0
        
        results[pert_name] = {
            "accuracy": accuracy,
            "attack_success_rate": asr,
            "correct": correct,
            "total": len(test_texts)
        }
        
        print(f"\n{pert_name}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(test_texts)})")
        print(f"  Attack Success Rate (ASR): {asr:.4f} ({flipped_wrong}/{correctly_classified_clean})")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--model_name", default="roberta-base", help="Base model name")
    parser.add_argument("--test_data", default="data/pheme.jsonl", help="Test data path")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3)
    model.to(device)
    model.eval()
    
    # Load test data
    test_rows = load_jsonl(Path(args.test_data))
    
    # Use only ferguson test samples for focused evaluation
    test_rows = [r for r in test_rows if r.get('event_id') == 'ferguson']
    
    if len(test_rows) > 300:
        test_rows = random.sample(test_rows, 300)
    
    test_texts = [r['text'] for r in test_rows]
    test_labels = [LABEL_MAP[r['label']] for r in test_rows]
    
    print(f"Loaded {len(test_texts)} test samples from ferguson event")
    
    # Evaluate robustness
    results = evaluate_robustness(model, tokenizer, test_texts, test_labels, device)
    
    # Save results
    output_path = Path("outputs/robustness") / f"robustness_results_seed{args.seed}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv for kk, vv in v.items()}
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY: ATTACK SUCCESS RATE (ASR)")
    print("="*60)
    print(f"{'Perturbation':<25} {'Accuracy':<12} {'ASR':<12}")
    print("-"*50)
    for pert_name, metrics in results.items():
        print(f"{pert_name:<25} {metrics['accuracy']:.4f}       {metrics['attack_success_rate']:.4f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    main()