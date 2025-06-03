import json
import random
import re

class DialogueCleaner:
    def clean(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("“", "\"").replace("”", "\"")
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("…", "...").replace("•", "-")
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"(User[12]):", r"\1 :", text)
        return text


def split_jsonl(input_path, train_path, val_path, val_ratio=0.1, seed=42):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - val_ratio))
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]

    with open(train_path, "w", encoding="utf-8") as f_train:
        for entry in train_data:
            f_train.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for entry in val_data:
            f_val.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dataset split: {len(train_data)} train / {len(val_data)} val")
