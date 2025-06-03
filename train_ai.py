import json
from sklearn.model_selection import train_test_split
from genai_model import TurkishChatBot

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    data = load_jsonl("processed_responses-v2.jsonl")
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_path = "temp_train.jsonl"
    val_path = "temp_val.jsonl"
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    chatbot = TurkishChatBot(model_dir="gpt2-turkish-chatbot-v3", model_name="redrussianarmy/gpt2-turkish-cased")
    chatbot.train(
        train_jsonl=train_path,
        val_jsonl=val_path,
        epochs=5,
        batch_size=2
    )
    
if __name__ == "__main__":
    main()
            

