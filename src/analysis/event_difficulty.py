"""
Event Difficulty Detection for Cross-Event Generalization
Measures distribution shift between training events and test events
"""

import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json

def load_jsonl_data(filepath, events_to_load=None):
    """Load JSONL and group by event_id"""
    data_by_event = {}
    # FIX: Explicitly use utf-8 encoding
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                event = row.get('event_id', 'unknown')
                if events_to_load and event not in events_to_load:
                    continue
                if event not in data_by_event:
                    data_by_event[event] = []
                data_by_event[event].append(row)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue
    return data_by_event

def get_text_embeddings(texts, model_name='roberta-base', batch_size=32):
    """Get RoBERTa embeddings for a list of texts"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i:i+batch_size]
            # Ensure batch is list of strings
            batch = [str(t) for t in batch]
            encoded = tokenizer(batch, padding=True, truncation=True, 
                               max_length=128, return_tensors='pt').to(device)
            outputs = model(**encoded)
            # Use CLS token embedding
            batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeds)
    
    return np.vstack(embeddings)

def compute_event_center(embeddings):
    """Compute centroid of an event's text embeddings"""
    return np.mean(embeddings, axis=0)

def compute_event_difficulty(train_events_data, test_event_data, model_name='roberta-base'):
    """
    Compute difficulty scores for each test event
    
    Args:
        train_events_data: dict of event_name -> list of text strings
        test_event_data: dict of event_name -> list of text strings
        model_name: HuggingFace model name
    
    Returns:
        dict: event_name -> difficulty_score (higher = harder)
    """
    # Combine all training texts
    all_train_texts = []
    for event, rows in train_events_data.items():
        texts = [row['text'] for row in rows]
        all_train_texts.extend(texts)
    
    print(f"Total training samples: {len(all_train_texts)}")
    
    # Get training embeddings (do once)
    train_embeddings = get_text_embeddings(all_train_texts, model_name)
    train_center = np.mean(train_embeddings, axis=0)
    
    # Compute difficulty for each test event
    difficulty_scores = {}
    
    for event, rows in test_event_data.items():
        print(f"\nProcessing event: {event}")
        test_texts = [row['text'] for row in rows]
        test_embeddings = get_text_embeddings(test_texts, model_name)
        
        # Average similarity to training center
        similarities = cosine_similarity(test_embeddings, train_center.reshape(1, -1))
        avg_similarity = np.mean(similarities)
        
        # Difficulty = 1 - similarity (higher = harder)
        difficulty_scores[event] = 1 - avg_similarity
        
        print(f"  Avg similarity: {avg_similarity:.4f}")
        print(f"  Difficulty: {difficulty_scores[event]:.4f}")
    
    return difficulty_scores

def main():
    # Path to your JSONL
    data_path = Path(__file__).parent.parent.parent / "data" / "pheme.jsonl"
    
    # Check if file exists
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        return
    
    # Load data
    print(f"Loading data from: {data_path}")
    all_data = load_jsonl_data(data_path)
    
    events = list(all_data.keys())
    print(f"\nFound {len(events)} events: {events}")
    
    # Need at least 2 events for LOEO
    if len(events) < 2:
        print("Need at least 2 events for analysis")
        return
    
    # Compute difficulty for each event using leave-one-event-out
    results = {}
    
    for test_event in events:
        print(f"\n{'='*50}")
        print(f"Leaving out: {test_event}")
        
        train_events = {e: all_data[e] for e in events if e != test_event}
        test_events = {test_event: all_data[test_event]}
        
        difficulty = compute_event_difficulty(train_events, test_events)
        results[test_event] = difficulty[test_event]
    
    print("\n" + "="*50)
    print("Final Event Difficulty Scores:")
    print("="*50)
    for event, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{event:20s}: {score:.4f}")
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / "outputs" / "event_difficulty.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()