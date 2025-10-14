from flask import Flask, render_template, request, jsonify
from cleaner.text_cleaner import clean_text
from crawler.web_crawler import scrape_web_summary
from requirement_planner import generate_advice

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"response": "Please enter your IT-related question."})

    # Step 1: Clean input
    cleaned = clean_text(query)

    # Step 2: Scrape contextual info
    web_context = scrape_web_summary(cleaned)

    # Step 3: Generate refined advice
    response = generate_advice(cleaned, web_context)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
