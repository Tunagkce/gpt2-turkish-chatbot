import json

input_path = "responsesGenAI_fixed.jsonl"
output_path = "responses_formatted-finalv2.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned_line = line.rstrip(",\n").strip()
        if not cleaned_line:
            continue
        try:
            data = json.loads(cleaned_line)
        except json.JSONDecodeError:
            print("Invalid JSON:", cleaned_line)
            continue
        data["dialogue"] = [msg for msg in data["dialogue"] if msg.startswith("User2:")]
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

