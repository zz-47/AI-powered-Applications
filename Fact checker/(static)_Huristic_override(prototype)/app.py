from flask import Flask, render_template, request, redirect, url_for
import re

app = Flask(__name__)

def heuristic_override(text):
    """
    Heuristic fake news detector logic.
    - Detects if text contains mentions of safe or harmful items.
    - Applies special logic for war/political/scientific topics.
    - Handles both short claims and longer paragraphs with nuanced reasoning.
    Returns:
        label (str): 'REAL', 'FAKE', or 'UNKNOWN'
        confidence (float): confidence percentage (0-100)
        reasoning (str): explanation for the decision
    """
    t = text.lower().strip()

    # Define safe items (food, science, technology, politics - neutral terms)
    safe_items = set([
        # Food & everyday
        "milk", "water", "juice", "vegetables", "fruits", "coffee", "tea", "bread", "rice", "eggs",
        "cheese", "yogurt", "nuts", "seeds", "fish", "chicken", "beef", "turkey", "tofu", "lentils",
        "beans", "spinach", "carrots", "apples", "bananas", "oranges", "potato", "corn", "oats", "honey",
        "olive oil", "butter", "sugar", "salt", "pepper", "herbs", "spices", "cucumber", "tomato", "onion",
        "garlic", "broccoli", "cauliflower", "mushrooms", "watermelon", "grapes", "strawberries",

        # Physics & Space
        "gravity", "mass", "energy", "velocity", "acceleration", "force", "momentum", "quantum", "photon",
        "electron", "proton", "neutron", "atom", "molecule", "galaxy", "planet", "star", "black hole",
        "supernova", "nebula", "universe", "cosmos", "light year", "astronomy", "space", "solar system",
        "orbit", "telescope", "comet", "asteroid", "eclipse",

        # Technology & Computing
        "computer", "software", "hardware", "internet", "algorithm", "database", "programming", "python",
        "java", "javascript", "network", "server", "cloud", "ai", "machine learning", "deep learning",
        "robotics", "encryption", "cybersecurity", "processor", "chip", "sensor", "mobile", "smartphone",
        "blockchain", "data", "quantum computing",

        # General Science & Study
        "biology", "chemistry", "physics", "mathematics", "geology", "ecology", "evolution", "cell",
        "enzyme", "dna", "genetics", "virus", "bacteria", "vaccine", "experiment", "theory", "hypothesis",

        # Politics & War - neutral/academic terms
        "treaty", "armistice", "alliance", "diplomacy", "negotiation", "sanction", "sovereignty",
        "government", "democracy", "republic", "parliament", "constitution", "election",
        "world war", "wwi", "wwii", "cold war", "united nations", "league of nations"
    ])

    # Define harmful items (poisons, toxins, harmful tech, war-related violence)
    harmful_items = set([
        # Poisons & toxins
        "poison", "bleach", "acid", "cyanide", "toxin", "arsenic", "rat poison", "methanol", "formaldehyde",
        "lead", "mercury", "ethanol", "pesticide", "herbicide", "fungicide", "radioactive", "uranium",
        "plutonium", "benzene", "chloroform", "ammonia", "hydrogen sulfide", "carbon monoxide", "nicotine",
        "morphine", "cocaine", "heroin", "methamphetamine", "sarin", "ricin", "aflatoxin", "cyanogen",
        "botulinum", "chloroquine", "arsenate", "arsenite", "chlorine gas",

        # Harmful tech/misuse
        "malware", "virus", "phishing", "ransomware", "hacker", "spyware", "trojan", "exploit", "ddos",

        # Dangerous physics/space phenomena
        "black hole", "supernova", "radiation", "solar flare", "gamma ray burst", "neutron star",

        # Misuse or dangerous materials
        "explosive", "bomb", "gunpowder", "napalm", "mustard gas",

        # War/Violence related
        "war", "battle", "conflict", "bombing", "invasion", "attack", "soldier", "army", "weapon",
        "casualty", "massacre", "siege", "genocide", "nuclear", "military",
    ])

    # --- Check harmful mentions first ---

    # If harmful words exist in text
    if any(h in t for h in harmful_items):

        # For short claims (under 200 chars), treat harmful claims suspiciously
        if len(t) < 200:
            if "drink" in t or "drinking" in t:
                return (
                    "FAKE",
                    99.9,
                    "Claim promotes harmful ingestion, likely false."
                )
            return (
                "FAKE",
                90.0,
                "Mentions harmful substances, probably false."
            )

        # For longer texts, check if war-related terms appear
        war_related = any(
            w in t for w in ["war", "battle", "conflict", "attack", "army", "soldier", "bomb"]
        )
        if war_related:
            # Check for historical context words, e.g. dates or treaties
            if any(
                date_word in t for date_word in [
                    "1914", "1918", "treaty", "armistice", "versailles", "world war"
                ]
            ):
                return (
                    "REAL",
                    85.0,
                    "Historical war-related content with factual context."
                )
            else:
                return (
                    "UNKNOWN",
                    60.0,
                    "Mentions war/violence but lacks sufficient details for confident classification."
                )

        # Default for harmful mentions in longer text
        return (
            "FAKE",
            85.0,
            "Mentions harmful substances or violence; caution advised."
        )

    # --- Check simple factual statements ---

    # Patterns for simple factual/definitional statements (usually short)
    simple_patterns = [
        r"^\s*[\w\s]+ (is|are|stands for|equals|equals to) [\w\s\d\.\-]+\.?\s*$",
        r"^\s*[\w\s]+ (prevents|helps prevent) [\w\s]+$"
    ]

    for pat in simple_patterns:
        if re.match(pat, t) and len(t) < 120:
            return (
                "REAL",
                100.0,
                "Simple factual or definitional statement."
            )

    # --- Keyword density analysis for longer text ---

    safe_count = sum(1 for word in safe_items if word in t)
    harmful_count = sum(1 for word in harmful_items if word in t)

    # For longer text (paragraphs), use keyword counts to decide
    if len(t) > 200:
        if safe_count > harmful_count:
            return (
                "REAL",
                90.0,
                f"Contains many scientific or safe terms ({safe_count}), likely truthful."
            )
        elif harmful_count > safe_count:
            return (
                "UNKNOWN",
                55.0,
                "Mentions harmful terms but mixed context; cannot confidently classify."
            )
        else:
            return (
                "UNKNOWN",
                50.0,
                "Long text with mixed or neutral content."
            )

    # --- Default fallback ---
    return (
        "UNKNOWN",
        50.0,
        "Insufficient info for confident classification."
    )


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Main route for the web app.
    Handles:
    - GET: renders empty detection page.
    - POST: processes submitted article text and returns detection results.
    """
    if request.method == "POST":
        article_text = request.form.get("article_text", "").strip()

        # If no input provided, show error message
        if article_text == "":
            return render_template("index.html", error="Please enter some text to detect.")

        # Run heuristic detection
        label, confidence, reasoning = heuristic_override(article_text)

        # Render results with submitted text and detection output
        return render_template(
            "index.html",
            article_text=article_text,
            label=label,
            confidence=confidence,
            reasoning=reasoning
        )

    # For GET request, just render the empty page
    return render_template("index.html")


@app.route("/restart")
def restart():
    """
    Route to reset/restart the detection page,
    redirects user to the main page with empty form.
    """
    return redirect(url_for("home"))


if __name__ == "__main__":
    # Run Flask app in debug mode for easier development
    app.run(debug=True)
