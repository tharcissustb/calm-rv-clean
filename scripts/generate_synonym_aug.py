"""
Generate synonym-augmented examples for ferguson LOEO training set.
Uses WordNet (no LLM, no pretrained knowledge).
"""

import json
import random
import nltk
from nltk.corpus import wordnet
from pathlib import Path

# Download required NLTK data (run once if not already)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

ROOT = Path(__file__).resolve().parents[1]

def get_synonyms(word, pos=None):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replace(text, p=0.1):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    new_words = []
    for word, tag in tagged:
        if random.random() < p:
            # Map POS to WordNet POS
            pos_map = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'R': wordnet.ADV, 'V': wordnet.VERB}
            wn_pos = None
            for key, val in pos_map.items():
                if tag.startswith(key):
                    wn_pos = val
                    break
            synonyms = get_synonyms(word, pos=wn_pos)
            if synonyms:
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if 'text' in row and 'label' in row:
                rows.append(row)
    return rows

def main():
    # Load all PHEME data
    data_path = ROOT / 'data/pheme.jsonl'
    all_rows = load_jsonl(data_path)
    
    # Filter out ferguson (test event) – keep only training events
    train_rows = [r for r in all_rows if r.get('event_id') != 'ferguson']
    print(f"Training events: {len(train_rows)} examples (excluding ferguson)")
    
    # Generate augmented examples (2 variants per original)
    augmented = []
    for row in train_rows:
        for i in range(2):
            new_text = synonym_replace(row['text'], p=0.1)
            new_row = row.copy()
            new_row['id'] = f"{row['id']}_syn_{i}"
            new_row['text'] = new_text
            augmented.append(new_row)
    
    # Save to JSONL
    out_path = ROOT / 'data/augmented/synonym_augmented.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in augmented:
            f.write(json.dumps(row) + '\n')
    
    print(f"Generated {len(augmented)} synonym-augmented examples")
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()