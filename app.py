from flask import Flask, request, jsonify, render_template
import random
import google.generativeai as genai  # Gemini API

app = Flask(__name__, template_folder="templates")


genai.configure(api_key="AIzaSyAv_wNeH0T4U7kF56vVye_OllVmmUHvPL4")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


nickname_dataset = {}
dataset_path = "./train_gpt_lstm.txt"

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read().split("### Input:")
        for block in content:
            if block.strip() == "":
                continue
            # Extract name, style, and output
            name_line = next((line for line in block.splitlines() if line.startswith("Name:")), None)
            style_line = next((line for line in block.splitlines() if line.startswith("Style:")), None)
            output_line = next((line for line in block.splitlines() if line.startswith("### Output")), None)

            if name_line and output_line:
                name = name_line.split(":", 1)[1].strip()
                style = style_line.split(":", 1)[1].strip() if style_line else "trendy"

                # Handle both formats:
                lines = block.splitlines()
                idx = lines.index(output_line)
                if output_line.strip() == "### Output:":
                    if idx + 1 < len(lines):
                        nickname = lines[idx + 1].strip()
                    else:
                        nickname = ""
                else:
                    nickname = output_line.split(":", 1)[1].strip()

                if nickname:
                    nickname_dataset[(name.lower(), style.lower())] = nickname
    print(f"✅ Loaded {len(nickname_dataset)} nickname entries from dataset.")
except FileNotFoundError:
    print(f"⚠️ Dataset file not found at {dataset_path}. Make sure it exists.")


@app.route("/", methods=["GET"])
def home():
    return render_template("index2.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        style = data.get("style", "trendy").strip()

        if not name:
            return jsonify({"error": "No name provided"}), 400

        # Step 1: Exact match (name + style)
        dataset_nickname = nickname_dataset.get((name.lower(), style.lower()))
        if dataset_nickname:
            print(f"🎯 Exact match found: ({name}, {style}) → {dataset_nickname}")
            return jsonify({"nickname": dataset_nickname})

        # Step 2: Default style ("trendy")
        dataset_nickname = nickname_dataset.get((name.lower(), "trendy"))
        if dataset_nickname:
            print(f"🎯 Default style match: ({name}, trendy) → {dataset_nickname}")
            return jsonify({"nickname": dataset_nickname})

        # Step 3: Match any style for the name
        for (n, s), nick in nickname_dataset.items():
            if n == name.lower():
                print(f"🎯 Any style match: ({name}, {s}) → {nick}")
                return jsonify({"nickname": nick})

        # Step 4: Fallback → Generate nickname using Gemini
        prompt = f"Generate a creative {style} nickname for the name '{name}'. Only return the nickname, no explanation."
        gemini_response = gemini_model.generate_content(prompt)
        nickname = gemini_response.text.strip()

        if not nickname:  # If Gemini fails, use a manual fallback
            fallback_nicknames = [
                f"{name}ster",
                f"Cool {name}",
                f"{name}y",
                f"The {style.capitalize()} {name}",
                f"{name}zilla",
                f"{name}_X",
                f"{name}tron",
            ]
            nickname = random.choice(fallback_nicknames)

        print(f"⚡ Gemini Fallback used → {nickname}")
        return jsonify({"nickname": nickname})

    except Exception as e:
        print("⚠️ ERROR in /generate:", str(e))
        return jsonify({"error": "Could not generate nickname"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
