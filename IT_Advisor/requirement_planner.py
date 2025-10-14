# requirement_planner.py
# Drop-in replacement for your old planner. Uses transformers model like before.

import re
import psutil
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
from difflib import SequenceMatcher

# Select model automatically by RAM
_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
_MODEL_NAME = "google/flan-t5-base" if _ram_gb < 12 else "google/flan-t5-large"

# Load tokenizer & model once at import
_tokenizer = T5Tokenizer.from_pretrained(_MODEL_NAME)
_model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)

# --- utility functions -----------------------------------------------------

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip()).lower()

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# Basic rule-based fallback planner (guaranteed reasonable output)
def rule_based_plan(query: str) -> str:
    q = query.lower()
    domain = "General IT project"
    if "chatbot" in q or "bot" in q:
        domain = "Chatbot"
    elif "web" in q or "website" in q:
        domain = "Web application"
    plan = [
        f"**Goal:** Build a {domain} to satisfy: {query}",
        "**Recommended Tech Stack:** Python, Flask, SQLite (start), simple frontend (HTML/CSS/JS).",
        "**Step-by-Step Plan:**",
        "1) Clarify requirements: users, example dialogues, must-have features.",
        "2) Design minimal data model and endpoints.",
        "3) Implement MVP: core functionality only.",
        "4) Add tests and logging.",
        "5) Deploy locally or on small host.",
        "**Testing & Deployment:** Unit tests + manual end-to-end checks. Deploy to a small VM or PaaS.",
        "**Notes & Reasoning:** Start simple; iterate once you have real user interactions."
    ]
    return "\n".join(plan)

# Validate model output: checks length, relevance, and novelty from template
def is_output_good(output: str, query: str, example_text: str) -> bool:
    if not output or len(output.strip()) < 120:
        return False
    # must mention at least one important token from query (heuristic)
    qtokens = set(re.findall(r'\w+', query.lower()))
    common = sum(1 for t in qtokens if t in output.lower())
    if common < max(1, min(3, len(qtokens)//4)):
        return False
    # must not be too similar to the example template
    if similar(normalize_text(output), normalize_text(example_text)) > 0.75:
        return False
    return True

# Create robust prompt (example is style-only and explicitly banned from repetition)
def build_prompt(query: str, context_summary: str) -> str:
    # compact example — clearly marked as "style only"
    example_q = "How can I automate file backups on Windows?"
    example_a = ("Goal: Automate file backups.\n"
                 "Tech Stack: Python (shutil), Task Scheduler.\n"
                 "Plan:\n1. Identify folders.\n2. Write copy script.\n3. Test locally.\n4. Schedule job.\nNotes: Use logging.")
    prompt = f"""
You are TechWISE Advisor, an expert IT project planner.

INSTRUCTIONS:
- Produce a fresh, original plan tailored to the user's question.
- Output must include these sections (in this order): Goal, Recommended Tech Stack, Step-by-Step Plan, Testing & Deployment, Notes & Reasoning.
- Do NOT copy the example below; it is provided for style only and must NOT be repeated verbatim.

=== EXAMPLE (STYLE ONLY) ===
Question: {example_q}
Answer:
{example_a}
=== END EXAMPLE ===

User Question: {query}
Web Context Summary: {context_summary}

Now produce a unique, detailed answer that follows the section order exactly.
"""
    return prompt

# Perform generation with configurable parameters
def _generate_text(prompt: str, max_length=600, num_beams=4, do_sample=False, repetition_penalty=1.2):
    tok = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # if prompt tokenizer length equals max_length it may be truncated; we still try
    gen = _model.generate(
        **tok,
        max_length=max_length,
        num_beams=num_beams if not do_sample else 1,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    return out

# Top-level API used by Flask
def generate_advice(query: str, context_summary: str = "") -> str:
    # 1) quick baseline
    baseline = rule_based_plan(query)

    # 2) build prompt
    prompt = build_prompt(query, context_summary)

    # 3) generation attempts with validator: beam then sampling
    example_text = ("Goal: Automate file backups.\nTech Stack: Python (shutil), Task Scheduler.\nPlan: 1. Identify folders. 2. Write script. 3. Test. 4. Schedule.")
    # try deterministic beams first
    try:
        out1 = _generate_text(prompt, max_length=650, num_beams=5, do_sample=False, repetition_penalty=1.2)
        if is_output_good(out1, query, example_text):
            return out1
    except Exception:
        out1 = ""

    # if not good, try sampled generation with temperature by toggling do_sample True (sampling)
    try:
        out2 = _generate_text(prompt, max_length=750, num_beams=1, do_sample=True, repetition_penalty=1.3)
        if is_output_good(out2, query, example_text):
            return out2
    except Exception:
        out2 = ""

    # As a last resort return the robust baseline but expanded with query tokens to be more specific
    fallback = baseline + "\n\n(Automatic fallback: model output failed quality checks; here's a deterministic plan.)"
    return fallback
