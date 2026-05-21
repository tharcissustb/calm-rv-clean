"""
Convert Chinese_Rumor_Dataset to JSONL format
Binary classification: rumor (true) vs non-rumor (false)
"""

import json
import os
from pathlib import Path

def convert_chinese_rumor_dataset(base_path, output_path):
    """
    Convert Chinese_Rumor_Dataset to JSONL
    
    Structure:
    - rumor-repost/ contains rumor examples (label = true)
    - non-rumor-repost/ contains non-rumor examples (label = false)
    """
    rows = []
    
    # Process rumor examples (label = "true")
    rumor_folder = Path(base_path) / "rumor-repost"
    if rumor_folder.exists():
        for json_file in rumor_folder.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Data is a list of posts in the thread
                    if isinstance(data, list) and len(data) > 0:
                        # Use the first post as source
                        first_post = data[0]
                        text = first_post.get('text', '')
                        if text:
                            rows.append({
                                "id": f"chinese_rumor_{json_file.stem}",
                                "dataset": "chinese_rumor",
                                "text": text,
                                "label": "true"
                            })
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    # Process non-rumor examples (label = "false")
    non_rumor_folder = Path(base_path) / "non-rumor-repost"
    if non_rumor_folder.exists():
        for json_file in non_rumor_folder.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        first_post = data[0]
                        text = first_post.get('text', '')
                        if text:
                            rows.append({
                                "id": f"chinese_nonrumor_{json_file.stem}",
                                "dataset": "chinese_rumor",
                                "text": text,
                                "label": "false"
                            })
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(rows)} examples to {output_path}")
    true_count = sum(1 for r in rows if r['label'] == 'true')
    false_count = len(rows) - true_count
    print(f"   Label distribution: true={true_count}, false={false_count}")
    
    return rows

def main():
    base_path = Path("Additional_datasets/Chinese_Rumor_Dataset/Chinese_Rumor_Dataset-master/CED_Dataset")
    output_path = Path("data/chinese_rumor.jsonl")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = convert_chinese_rumor_dataset(base_path, output_path)

if __name__ == "__main__":
    main()