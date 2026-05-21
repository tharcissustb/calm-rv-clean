"""
LLM Augmentation for Hard Events using BART-base (Local, Stable)
Runs on CPU to avoid CUDA memory issues
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Hard events from your experiments
HARD_EVENTS = ['ferguson', 'gurlitt', 'prince-toronto', 'putinmissing']

def load_data(filepath):
    """Load JSONL and group by event"""
    data_by_event = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                event = row.get('event_id', 'unknown')
                if event not in data_by_event:
                    data_by_event[event] = []
                data_by_event[event].append(row)
            except:
                continue
    return data_by_event

def generate_paraphrases_bart(texts, num_variants=3, batch_size=2):
    """
    Generate paraphrases using BART-base on CPU (stable, no CUDA errors)
    """
    # Force CPU for stability
    device = torch.device('cpu')
    print(f"Using device: {device} (stable, may be slower)")
    
    # Use BART-base (smaller, faster, less memory)
    model_name = "facebook/bart-base"
    
    print(f"Loading {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    
    all_paraphrases = []
    all_original_texts = []  # Track which original each paraphrase came from
    
    # Simple paraphrase prompt
    prefix = "paraphrase: "
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating paraphrases"):
        batch = texts[i:i+batch_size]
        input_texts = [prefix + t for t in batch]
        
        # Tokenize
        inputs = tokenizer(input_texts, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        
        # Generate multiple paraphrases per input
        for variant in range(num_variants):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,  # Reduced from 5 for speed
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True
                )
            
            paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up paraphrases (remove the "paraphrase: " prefix if present)
            clean_paraphrases = []
            for p in paraphrases:
                p = p.replace("paraphrase: ", "").strip()
                if len(p) > 10:  # Only keep non-empty paraphrases
                    clean_paraphrases.append(p)
            
            all_paraphrases.extend(clean_paraphrases)
            all_original_texts.extend(batch[:len(clean_paraphrases)])
            
            # Clear cache
            del outputs
            torch.cuda.empty_cache()
    
    return all_paraphrases, all_original_texts

def main():
    # Load data
    data_path = Path("data/pheme.jsonl")
    data_by_event = load_data(data_path)
    
    # Collect hard event texts
    hard_texts = []
    hard_labels = []  # Store corresponding labels
    for event in HARD_EVENTS:
        if event in data_by_event:
            for row in data_by_event[event]:
                hard_texts.append(row['text'])
                hard_labels.append(row['label'])
            print(f"{event}: {len(data_by_event[event])} texts")
    
    print(f"\nTotal hard event texts: {len(hard_texts)}")
    
    # Generate paraphrases
    print("\nGenerating paraphrases with BART-base (this may take 20-30 minutes)...")
    paraphrases, original_texts = generate_paraphrases_bart(hard_texts, num_variants=3, batch_size=2)
    
    print(f"\nGenerated {len(paraphrases)} paraphrases")
    
    # Create augmented data with labels
    augmented_rows = []
    for i, (orig_text, para_text) in enumerate(zip(original_texts, paraphrases)):
        # Find the original label for this text
        try:
            orig_idx = hard_texts.index(orig_text)
            label = hard_labels[orig_idx]
        except ValueError:
            label = "unverified"
        
        row = {
            "id": f"augmented_{i}",
            "dataset": "pheme_augmented",
            "event_id": "augmented_hard",
            "text": para_text,
            "label": label
        }
        augmented_rows.append(row)
    
    # Save augmented data
    output_path = Path("data/augmented/hard_augmented.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in augmented_rows:
            f.write(json.dumps(row) + '\n')
    
    print(f"\n✅ Saved {len(augmented_rows)} augmented examples to: {output_path}")
    
    # Show sample
    print("\nSample augmented text:")
    if augmented_rows:
        sample = augmented_rows[0]
        print(f"Original label: {sample['label']}")
        print(f"Augmented text: {sample['text'][:200]}...")

if __name__ == "__main__":
    main()