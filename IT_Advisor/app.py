from flask import Flask, render_template, request, jsonify
from model_service import generate_answer
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("prompt", "").strip()
    style = data.get("style", "paragraph").strip().lower()

    if not question:
        return jsonify({"answer": "⚠️ Please enter a valid question."})

    logging.info(f"Received: {question}")
    answer = generate_answer(question, style)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
