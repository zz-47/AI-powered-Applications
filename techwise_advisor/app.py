from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Flask app setup
# -------------------------
app = Flask(__name__)

# -------------------------
# Model Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# -------------------------
# Example prompts for frontend
# -------------------------
ISSUES = [
    "Windows 11 battery drains very fast even when idle and the laptop overheats during light tasks",
    "Windows 11 Wi-Fi disconnects randomly after waking from sleep",
    "After updating Windows 11, some USB devices (keyboard and mouse) stop working intermittently",
    "Windows 11 laptop screen flickers randomly during light tasks",
    "Printer connected to Windows 11 network is not detected by multiple computers"
]

# -------------------------
# Generate reasoning-based response
# -------------------------
def generate_hint_response(issue: str) -> str:
    prompt = (
        f"You are an expert Windows technician. "
        f"Provide a clear, professional, reasoning-based analysis for the following issue:\n\n"
        f"Issue: {issue}\n\n"
        f"Output should:\n"
        f"- Explain likely causes\n"
        f"- Provide hints at the problem\n"
        f"- Avoid step-by-step instructions\n"
        f"- Focus on Windows-specific scenarios and hardware/software considerations\n"
        f"- Keep the response concise and readable"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_length=350,
        min_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Flask routes
# -------------------------
@app.route('/')
def index():
    # Pass example prompts to frontend
    return render_template('index.html', issues=ISSUES)

@app.route('/get_advice', methods=['POST'])
def get_advice():
    data = request.get_json()
    issue = data.get("query", "").strip()
    if not issue:
        return jsonify({"error": "No query provided."}), 400
    try:
        response_text = generate_hint_response(issue)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: streaming version (if you want to stream chunks to frontend)
@app.route('/get_advice_stream', methods=['POST'])
def get_advice_stream():
    data = request.get_json()
    issue = data.get("query", "").strip()
    if not issue:
        return jsonify({"error": "No query provided."}), 400

    def generate():
        # For simplicity, here we just yield the full response
        # Can be adapted to stream model output token by token
        response_text = generate_hint_response(issue)
        yield response_text

    return Response(stream_with_context(generate()), mimetype='text/plain')

# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
