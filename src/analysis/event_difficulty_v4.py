"""
Event Difficulty Detection - CPU Version (No GPU Memory Issues)
"""

import json
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc

def load_data(filepath):
    """Load JSONL with utf-8 encoding"""
    data_by_event = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                event = row['event_id']
                text = row['text']
                
                if event not in data_by_event:
                    data_by_event[event] = []
                data_by_event[event].append(text)
            except:
                continue
    
    return data_by_event

def get_embeddings_cpu(texts, model_name='roberta-base', batch_size=8, max_samples=500):
    """Get embeddings on CPU - slower but no memory issues"""
    
    # Force CPU
    device = torch.device('cpu')
    print(f"  Using device: CPU (slower but stable)")
    
    # Sample if too large
    if len(texts) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        print(f"  Sampled {len(texts)} texts")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="  Computing embeddings"):
            batch = texts[i:i+batch_size]
            
            encoded = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        
        # Clear memory
        del outputs, encoded
        gc.collect()
    
    # Move model to CPU and clear
    model.cpu()
    del model
    gc.collect()
    
    return np.vstack(all_embeddings)

def main():
    # Path to data
    data_path = Path("data/pheme.jsonl")
    
    print(f"Loading data from: {data_path}")
    data_by_event = load_data(data_path)
    
    events = list(data_by_event.keys())
    print(f"Found {len(events)} events: {events}")
    
    # Count samples
    print("\nSamples per event:")
    for event, texts in data_by_event.items():
        print(f"  {event}: {len(texts)}")
    
    # Compute difficulty for each event
    results = {}
    
    for test_event in events:
        print(f"\n{'='*50}")
        print(f"Test event: {test_event}")
        
        # Training texts from all other events
        train_texts = []
        for event, texts in data_by_event.items():
            if event != test_event:
                train_texts.extend(texts)
        
        test_texts = data_by_event[test_event]
        
        print(f"  Training samples: {len(train_texts)} (will sample 500 max)")
        print(f"  Test samples: {len(test_texts)} (will sample 300 max)")
        
        # Get embeddings on CPU
        print("  Computing training embeddings...")
        train_emb = get_embeddings_cpu(train_texts, max_samples=500)
        train_center = np.mean(train_emb, axis=0)
        
        print("  Computing test embeddings...")
        test_emb = get_embeddings_cpu(test_texts, max_samples=300)
        
        # Compute similarity
        similarities = cosine_similarity(test_emb, train_center.reshape(1, -1))
        avg_sim = np.mean(similarities)
        difficulty = 1 - avg_sim
        
        results[test_event] = {
            "avg_similarity": float(avg_sim),
            "difficulty": float(difficulty),
            "n_train": len(train_texts),
            "n_test": len(test_texts)
        }
        
        print(f"  Avg similarity: {avg_sim:.4f}")
        print(f"  Difficulty: {difficulty:.4f}")
        
        # Force memory cleanup
        gc.collect()
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS (sorted by difficulty - highest first)")
    print("="*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['difficulty'], reverse=True)
    for event, scores in sorted_results:
        print(f"{event:20s}: difficulty={scores['difficulty']:.4f} (sim={scores['avg_similarity']:.4f})")
    
    # Compare with your actual LOEO F1 scores
    print("\n" + "="*50)
    print("COMPARISON WITH ACTUAL MODEL PERFORMANCE")
    print("="*50)
    
    # From your Excel - actual LOEO macro_f1 values
    actual_f1 = {
        "charliehebdo": 0.313,      # from your loeo_pheme_roberta_ce sheet
        "ebola-essien": 0.148,
        "ferguson": 0.046,
        "germanwings-crash": 0.229,
        "gurlitt": 0.068,
        "ottawashooting": 0.300,    # actually 0.300 from your data
        "prince-toronto": 0.072,
        "putinmissing": 0.045,
        "sydneysiege": 0.299
    }
    
    print(f"\n{'Event':20s} {'Difficulty':>12s} {'Actual F1':>12s} {'Relationship':>20s}")
    print("-" * 65)
    
    for event in [e for e, _ in sorted_results]:
        diff = results[event]['difficulty']
        f1 = actual_f1.get(event, 0)
        
        # Check if difficulty correlates with low F1
        if diff > 0.0020 and f1 < 0.1:
            relationship = "✓ High Diff = Low F1"
        elif diff < 0.0019 and f1 > 0.25:
            relationship = "✓ Low Diff = High F1"
        elif diff > 0.0019 and f1 < 0.15:
            relationship = "Partial correlation"
        else:
            relationship = "Check manually"
            
        print(f"{event:20s} {diff:12.4f} {f1:12.3f} {relationship:20s}")
    
    # Save results
    output_path = Path("outputs/event_difficulty_v4.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print("\n" + "="*50)
    print("KEY INSIGHT FOR YOUR PAPER:")
    print("="*50)
    print("If difficulty correlates with low F1, then events that are")
    print("distributionally different from training cause model failure.")
    print("This motivates adaptive calibration for hard events.")

if __name__ == "__main__":
    main()