# File: app.py
import logging
from flask import Flask, request, Response, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- Model Loading --------------------
device = torch.device("cpu")  # Change to "cuda" if GPU is available
logging.info("Loading FLAN-T5 model on CPU...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
logging.info("Model loaded successfully.")

# -------------------- Utilities --------------------
def deduplicate_text(text):
    """Remove repeated lines/sentences."""
    seen = set()
    result = []
    for line in text.split("\n"):
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            result.append(line_clean)
    return "\n".join(result)

def generate_prompt(query, style):
    """Create a very long and detailed prompt."""
    style_prompt = (
        "Use numbered steps with sub-points, provide long detailed explanations, examples, and preventive measures." 
        if style=="points" 
        else "Write in very long structured paragraphs with full explanations, examples, and preventive advice."
    )
    return f"""
You are TechWISE Advisor, an expert IT assistant. Your answers must be **as long and detailed as possible**, covering all scenarios and beginner-friendly.

Include:
- Introduction (explain the problem in depth)
- Causes (all possible reasons)
- Step-by-step Solutions (detailed, numbered, with sub-points if necessary)
- Extra Tips and Preventive Measures
- Examples if applicable

**Make sure your answer is at least 1000–1500 words, very thorough, and contains multiple examples. Do not stop early.**

Answer Style: {style_prompt}

Question: "{query}"
"""

# -------------------- Streaming Response --------------------
def stream_answer(prompt):
    """Generate answer in chunks for very long responses."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2500,    # allow very long answers
            num_beams=5,            # better quality
            no_repeat_ngram_size=3,
            early_stopping=False,   # allow full completion
            do_sample=True,         # sampling for diverse output
            top_p=0.95,
            temperature=0.75
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = deduplicate_text(decoded)
        
        # Yield in chunks of 200 characters
        chunk_size = 200
        for i in range(0, len(decoded), chunk_size):
            yield decoded[i:i+chunk_size]
            time.sleep(0.02)
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        yield "⚠️ Failed to generate advice."

# -------------------- Flask Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_advice_stream", methods=["POST"])
def get_advice_stream():
    data = request.get_json()
    query = data.get("query", "").strip()
    style = data.get("style", "paragraph")
    if not query:
        return Response("⚠️ Please provide a query.", mimetype="text/plain")
    logging.info(f"Received query: {query} | style: {style}")
    prompt = generate_prompt(query, style)
    return Response(stream_answer(prompt), mimetype="text/plain")

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
