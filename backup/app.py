from flask import Flask, render_template, request, redirect, url_for
import spacy
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)

# Load spaCy model (assuming English language)
nlp = spacy.load('en_core_web_sm')

# Initialize Sentence-BERT model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define more specific labels for the zero-shot classification model
labels = ["Factual", "False but Plausible", "Completely Fake", "Unverifiable"]

class SemanticAgent:
    def __init__(self):
        pass

    def analyze_text(self, text):
        t = text.lower().strip()
        
        # Handle short or empty text inputs
        if len(t) < 10:
            return "UNKNOWN", 50.0, "Text is too short to determine accuracy."

        doc = nlp(t)

        # Internal analysis using NLP to check for harmful terms and basic factual structure
        harmful_terms = self._detect_harmful_terms(t)
        if harmful_terms:
            return self._fake_news_result(harmful_terms)

        # Factual claim check based on patterns like "X is Y"
        factual_claim = self._check_for_factual_claims(t)
        if factual_claim:
            return self._real_news_result()

        # Classify with Zero-shot classification
        label, confidence = self.classify_with_zero_shot(text)
        
        return label, confidence, "Text classified using Zero-shot model for accurate labeling."

    def _detect_harmful_terms(self, text):
        harmful_items = ["poison", "virus", "malware", "war", "explosion", "attack"]
        return [item for item in harmful_items if item in text]

    def _check_for_factual_claims(self, text):
        # Check for simple factual patterns like "X is Y"
        pattern = r"^\s*[\w\s]+ (is|are|stands for|equals|prevents|helps prevent) [\w\s]+$"
        return bool(re.match(pattern, text))

    def classify_with_zero_shot(self, text):
        # Perform zero-shot classification using pre-defined labels
        result = classifier(text, candidate_labels=labels)
        
        # Choose the label with the highest confidence
        max_score_index = result['scores'].index(max(result['scores']))
        label = result['labels'][max_score_index]
        confidence = result['scores'][max_score_index] * 100  # Convert to percentage

        # Return the result formatted to two decimal places
        return label, round(confidence, 2)

    def _fake_news_result(self, harmful_terms):
        return "FAKE", 95.0, f"Detected harmful content such as: {', '.join(harmful_terms)}."

    def _real_news_result(self):
        return "REAL", 90.0, "This news seems factual based on analysis."

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        article_text = request.form.get("article_text", "").strip()

        # If no input provided, show error message
        if not article_text:
            return render_template("index.html", error="Please enter some text to detect.")

        # Create semantic agent instance and analyze text
        agent = SemanticAgent()
        
        # Classify text and get reasoning
        label, confidence, reasoning = agent.analyze_text(article_text)

        # Render results with submitted text and detection output
        return render_template(
            "index.html",
            article_text=article_text,
            label=label,
            confidence=confidence,
            reasoning=reasoning
        )

    return render_template("index.html")

@app.route("/restart")
def restart():
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
