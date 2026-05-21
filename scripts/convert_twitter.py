"""
Convert Twitter15/Twitter16 dataset to JSONL format
Based on Ma et al. (IJCAI 2016) dataset structure
"""

import json
import re
from pathlib import Path

def convert_twitter(input_path, output_path, dataset_name):
    """
    Convert Twitter15/16 to JSONL
    
    Expected format (per line):
    label \t source_text \t reply1 \t reply2 \t ... \t replyK
    
    label: 0 = false, 1 = true
    """
    rows = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            # First part is label (0 or 1)
            label_raw = parts[0].strip()
            label = "true" if label_raw == "1" else "false"
            
            # Second part is source text
            source_text = parts[1].strip() if len(parts) > 1 else ""
            
            # Remaining parts are replies
            replies = parts[2:] if len(parts) > 2 else []
            
            # Clean up replies (remove empty)
            replies = [r.strip() for r in replies if r.strip()]
            
            # Build the thread text (source + replies for context)
            thread_text = source_text
            for i, reply in enumerate(replies[:10]):  # Limit to first 10 replies
                thread_text += f" [REPLY{i+1}] {reply}"
            
            row = {
                "id": f"{dataset_name}_{line_idx}",
                "dataset": dataset_name,
                "event_id": dataset_name,  # No event structure in Twitter15/16
                "source_text": source_text,
                "replies": replies,
                "text": thread_text,
                "label": label
            }
            rows.append(row)
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(rows)} examples from {dataset_name}")
    print(f"   Label distribution: true={sum(1 for r in rows if r['label']=='true')}, "
          f"false={sum(1 for r in rows if r['label']=='false')}")
    
    return rows

def main():
    # Set paths based on where you extracted the files
    base_path = Path("data/raw/twitter")
    
    # Convert Twitter15
    twitter15_input = base_path / "twitter15.txt"  # adjust filename as needed
    if twitter15_input.exists():
        convert_twitter(twitter15_input, "data/twitter15.jsonl", "twitter15")
    else:
        print(f"⚠️ Twitter15 not found at {twitter15_input}")
        print("   Please download and extract to data/raw/twitter/")
    
    # Convert Twitter16
    twitter16_input = base_path / "twitter16.txt"
    if twitter16_input.exists():
        convert_twitter(twitter16_input, "data/twitter16.jsonl", "twitter16")
    else:
        print(f"⚠️ Twitter16 not found at {twitter16_input}")

if __name__ == "__main__":
    main()