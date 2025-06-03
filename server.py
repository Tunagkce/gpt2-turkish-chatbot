from flask import Flask, request, jsonify
from flask_cors import CORS
from genai_model import TurkishChatBot
import torch
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)


bot = TurkishChatBot(model_dir="gpt2-turkish-chatbot-v3")
bot.load_model()

last_dialogue = []
last_suggestions = []

@app.route("/")
def home():
    return "Chatbot backend is running!"

@app.route("/suggest", methods=["POST"])
def suggest():
    global last_dialogue, last_suggestions

    data = request.get_json()
    dialogue = data.get("dialogue", []) 
    
    if not dialogue or not isinstance(dialogue, list):
        return jsonify({"error": "Invalid dialogue format"}), 400

    last_dialogue = dialogue
    last_suggestions = bot.generate_multiple_replies(dialogue, num_replies=3)

    return jsonify({
        "suggestions": last_suggestions,
        "context": dialogue
    })

@app.route("/regenerate", methods=["GET"])
def regenerate():
    global last_dialogue, last_suggestions

    if not last_dialogue or not last_suggestions:
        return jsonify({"error": "No previous suggestions found. Call /suggest first."}), 400

    index_str = request.args.get("index", "0")
    try:
        index = int(index_str)
        if not (0 <= index < len(last_suggestions)):
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid index. Must be 0, 1, or 2."}), 400

    
    new_reply = bot.regenerate_reply(last_dialogue, index=index)
    last_suggestions[index] = new_reply

    return jsonify({
        "suggestions": last_suggestions,
        "regenerated": new_reply,
        "updated_index": index,
        "context": last_dialogue
    })

@app.route("/dialogue", methods=["GET"])
def get_last_dialogue():
    if not last_dialogue:
        return jsonify({"error": "No dialogue found."}), 404
    return jsonify({"dialogue": last_dialogue})

@app.route("/suggestions", methods=["GET"])
def get_last_suggestions():
    if not last_suggestions:
        return jsonify({"error": "No suggestions found."}), 404
    return jsonify({"suggestions": last_suggestions})


if __name__ == "__main__":
    app.run(debug=True, port=3002)
