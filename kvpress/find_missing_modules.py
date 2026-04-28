import json
import re
from collections import Counter
from pathlib import Path

path = Path(r"C:\Users\Prodyut\Downloads\TurnKV\predictions_sample_shard0.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
missing = Counter()
for row in rows:
    for field in ("execution_feedback", "compilation_feedback"):
        text = str(row.get(field, ""))
        for m in re.findall(r"No module named '([^']+)'", text):
            missing[m.split(".")[0]] += 1
print("Missing modules:")
for mod, cnt in missing.most_common():
    print(f"  {mod}: {cnt} occurrences")
