import json
from pathlib import Path

data_path = Path("data/pheme.jsonl")

print(f"Loading from: {data_path.absolute()}")

count = 0
events = set()

with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
                events.add(row.get('event_id', 'unknown'))
                count += 1
            except Exception as e:
                print(f"Error on line {count}: {e}")

print(f"Total rows: {count}")
print(f"Unique events: {len(events)}")
print(f"Events: {events}")