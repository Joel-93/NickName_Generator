from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__, template_folder="templates")

# Load fine-tuned model
model_path = "./nickname-model-gptneo"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Helper function to clean nickname output
def clean_nickname(text):
    if "### Output:" in text:
        nickname = text.split("### Output:")[-1]
    else:
        nickname = text

    # Remove everything after a space or newline (nicknames should be 1–2 words)
    nickname = nickname.strip().split("\n")[0]
    nickname = nickname.split(" ")[0]  # keep only first token if it rambles

    # Remove weird characters
    nickname = "".join([c for c in nickname if c.isalnum()])

    return nickname


# Route to serve your frontend
@app.route("/", methods=["GET"])
def home():
    return render_template("index2.html")   # serves index.html


# API route to generate nickname
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    name = data.get("name", "")
    style = data.get("style", "trendy")
    temperature = float(data.get("temperature", 0.9))
    top_p = float(data.get("top_p", 0.95))
    max_new_tokens = int(data.get("max_new_tokens", 12))

    if not name:
        return jsonify({"error": "No name provided"}), 400

    prompt = (
      f"### Input:\n"
      f"Name: {name}\n"
      f"Style: {style}\n"
      f"Generate a short, creative nickname that matches the style. "
      f"Nickname should be 1–2 words, fun, and catchy.\n\n"
      f"### Output:\n"
   )
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
      **inputs,
      max_new_tokens=5,     # nicknames should be short
      temperature=0.6,      # less randomness
      top_p=0.8,            # tighter sampling
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    nickname = clean_nickname(raw)

    return jsonify({"nickname": nickname})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
