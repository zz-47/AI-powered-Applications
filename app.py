from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define more specific labels for the zero-shot classification model
labels = ["Factual", "False but Plausible", "Completely Fake", "Unverifiable"]

class SemanticAgent:
    def __init__(self):
        pass

    def analyze_text(self, text):
        t = text.lower().strip()
        
        if len(t) < 10:
            return "UNKNOWN", 50.0, "Text is too short to determine accuracy."
        
        # Classify with Zero-shot classification
        label, confidence = self.classify_with_zero_shot(text)
        
        return label, confidence, "Text classified using Zero-shot model for accurate labeling."

    def classify_with_zero_shot(self, text):
        result = classifier(text, candidate_labels=labels)
        max_score_index = result['scores'].index(max(result['scores']))
        label = result['labels'][max_score_index]
        confidence = result['scores'][max_score_index] * 100  # Convert to percentage
        return label, round(confidence, 2)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        article_text = request.form.get("article_text", "").strip()

        # Create semantic agent instance and analyze text
        agent = SemanticAgent()
        
        # Classify text and get reasoning
        label, confidence, reasoning = agent.analyze_text(article_text)

        # Return result as JSON
        response = {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning
        }

        return jsonify(response)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
