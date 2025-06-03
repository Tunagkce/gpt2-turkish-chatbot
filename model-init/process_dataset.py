import json
import nltk
nltk.download('punkt')

def preprocess_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()  
            if not line:
                continue
            try:
                data = json.loads(line) 
                last_msg = data.get("last_message", "")
                reply = data.get("reply", "")
                if last_msg and reply:
                    dialogue_text = f"{last_msg}\nUser1: {reply}"
                   
                    tokens = nltk.word_tokenize(dialogue_text, language="turkish")
                    tokenized_text = " ".join(tokens)
                    fout.write(json.dumps({"text": tokenized_text.strip()}, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_number}: {e}")
                print(f"Problematic line: {line}")

if __name__ == "__main__":
    preprocess_jsonl("responses_formatted-finalv2_cleaned.jsonl", "processed_responses-v2.jsonl")
