"""
Generate LIAR-specific augmentations using BART
Creates synthetic examples for political claim verification
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def load_liar_data(filepath, max_samples=3000):
    """Load LIAR data and sample a subset for augmentation"""
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                texts.append(row['text'])
                labels.append(row['label'])
            except:
                continue
    
    # Sample balanced subset
    data = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(data)
    
    # Take up to max_samples
    sampled = data[:max_samples]
    sampled_texts = [t for t, l in sampled]
    sampled_labels = [l for t, l in sampled]
    
    print(f"Loaded {len(sampled_texts)} LIAR examples for augmentation")
    print(f"Label distribution: true={sampled_labels.count('true')}, "
          f"false={sampled_labels.count('false')}, "
          f"unverified={sampled_labels.count('unverified')}")
    
    return sampled_texts, sampled_labels

def generate_paraphrases_bart(texts, labels, num_variants=2, batch_size=4):
    """
    Generate paraphrases using BART-base on CPU
    """
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model_name = "facebook/bart-base"
    
    print(f"Loading {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    
    all_paraphrases = []
    all_labels = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating LIAR paraphrases"):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        
        # Generate multiple paraphrases per input
        for _ in range(num_variants):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True
                )
            
            paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_paraphrases.extend(paraphrases)
            all_labels.extend(batch_labels)
    
    return all_paraphrases, all_labels

def main():
    # Create output directory
    output_dir = Path("data/augmented")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LIAR data
    data_path = Path("data/liar.jsonl")
    liar_texts, liar_labels = load_liar_data(data_path, max_samples=3000)
    
    # Generate paraphrases
    print("\nGenerating LIAR-specific paraphrases (this will take 30-40 minutes)...")
    paraphrases, labels = generate_paraphrases_bart(liar_texts, liar_labels, num_variants=2, batch_size=4)
    
    print(f"\nGenerated {len(paraphrases)} LIAR paraphrases")
    
    # Save augmented data
    output_path = output_dir / "liar_augmented.jsonl"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (text, label) in enumerate(zip(paraphrases, labels)):
            row = {
                "id": f"liar_augmented_{i}",
                "dataset": "liar_augmented",
                "event_id": "liar_standalone",
                "text": text,
                "label": label
            }
            f.write(json.dumps(row) + '\n')
    
    print(f"\n✅ Saved {len(paraphrases)} LIAR-augmented examples to: {output_path}")
    
    # Show samples
    print("\n📝 Sample LIAR-augmented texts:")
    for i in range(min(3, len(paraphrases))):
        print(f"  Original label: {labels[i]}")
        print(f"  Augmented: {paraphrases[i][:150]}...")
        print()

if __name__ == "__main__":
    main()