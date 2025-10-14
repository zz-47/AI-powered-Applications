# requirement_planner.py
import json
import re
import logging
import textwrap
from typing import Dict, Any, List

# import the model objects from your existing model_service
from model_service import tokenizer, model, device

import torch
logging.basicConfig(level=logging.INFO)

# Config
MAX_INPUT_TOKENS = 1024
MAX_SUMMARY_TOKENS = 256
MAX_STEP_TOKENS = 400


def run_model(prompt: str, max_new_tokens: int = 256, num_beams: int = 4, temperature: float = 0.7) -> str:
    """Run the local model for a single completion (synchronous)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.6,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def extract_json_from_text(text: str) -> Any:
    """Try to find a JSON object in text and parse it."""
    # greedy find first { ... } balanced braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        block = match.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # try to fix simple single quotes and trailing commas
            try:
                fixed = block.replace("'", '"')
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except Exception:
                return None
    return None


def summarize_if_long(text: str) -> str:
    """If input tokenized length is large, create a short summary using the model."""
    tokens = tokenizer.tokenize(text)
    if len(tokens) > 900:
        prompt = (
            "Summarize the following technical user problem into one concise paragraph suitable for troubleshooting context:\n\n"
            f"{text}\n\nSummary:"
        )
        return run_model(prompt, max_new_tokens=MAX_SUMMARY_TOKENS, num_beams=4, temperature=0.3)
    return text


def extract_requirements(user_text: str) -> Dict[str, Any]:
    """
    Use the model to extract structured fields from the user's question.
    Returns dict with keys (may be None): problem_summary, os, hardware, symptoms, frequency, triggers,
    recent_changes, constraints, desired_outcome, urgency.
    """
    text = summarize_if_long(user_text)
    prompt = textwrap.dedent(f"""
    You are a requirements extractor. Read the user problem below and produce a JSON object with EXACT keys:
    {{
      "problem_summary": string,
      "os": string or null,
      "hardware": string or null,
      "symptoms": string or null,
      "frequency": string or null,
      "triggers": string or null,
      "recent_changes": string or null,
      "constraints": string or null,
      "desired_outcome": string or null,
      "urgency": string or null
    }}

    Don’t add any extra commentary, only output the JSON object.

    User problem:
    {text}

    JSON:
    """).strip()

    raw = run_model(prompt, max_new_tokens=250, num_beams=4, temperature=0.0)
    parsed = extract_json_from_text(raw)
    if parsed is None:
        # fallback: heuristic extraction
        parsed = {
            "problem_summary": (user_text[:1000] + "...") if len(user_text) > 1000 else user_text,
            "os": None, "hardware": None, "symptoms": None,
            "frequency": None, "triggers": None, "recent_changes": None,
            "constraints": None, "desired_outcome": None, "urgency": None
        }
    return parsed


def generate_plan(requirements: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """
    Given the extracted requirements, ask the model to generate a prioritized troubleshooting plan.
    Returns a dict: { "confidence": int, "steps": [ {id,title,description,prechecks,commands,estimated_minutes,risk} ] }
    """
    req_json = json.dumps(requirements, indent=2)
    prompt = textwrap.dedent(f"""
    You are a senior systems engineer. Based on this extracted requirements JSON, produce a prioritized troubleshooting plan.
    Output valid JSON with keys:
    {{
      "confidence": integer (1-10),
      "steps": [
        {{
          "id": integer,
          "title": string,
          "description": string,
          "prechecks": [string],
          "commands": [string],
          "estimated_minutes": integer,
          "risk": "low"|"medium"|"high"
        }}
      ]
    }}

    Requirements:
    {req_json}

    Provide steps from safest/lowest-risk to more invasive. Aim for 3-8 steps.
    """).strip()

    raw = run_model(prompt, max_new_tokens=512, num_beams=5, temperature=0.45)
    parsed = extract_json_from_text(raw)
    if parsed is None:
        # If parsing fails, build a minimal plan
        parsed = {"confidence": 6, "steps": [
            {"id":1,"title":"Check drivers","description":"Update or reinstall Wi-Fi drivers from vendor.",
             "prechecks":["Identify adapter model","Backup current driver"], "commands":[],
             "estimated_minutes":15, "risk":"low"},
            {"id":2,"title":"Power settings","description":"Disable power saving for adapter", "prechecks":[], "commands":[],
             "estimated_minutes":5, "risk":"low"}
        ]}
    return parsed


def expand_steps(steps: List[Dict[str, Any]], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For each step generated above, ask the model to produce a detailed human-friendly execution block including:
    - short intro,
    - exact commands if available,
    - how to verify success,
    - rollback steps.
    """
    expanded = []
    for step in steps:
        title = step.get("title")
        description = step.get("description", "")
        prompt = textwrap.dedent(f"""
        You are TechWISE Advisor. Expand this troubleshooting step into a full actionable guide.

        Step title: {title}
        Short description: {description}
        Context / Requirements: {json.dumps(requirements)}

        Output a JSON object with keys:
        {{
          "id": {step.get('id')},
          "title": "{title}",
          "guide": string,           # user-friendly step-by-step guide (2-6 paragraphs)
          "commands": [string],      # zero or more exact commands (Windows shell) to run as examples
          "verification": string,    # how to verify success
          "rollback": string         # how to revert the step if it makes things worse (short)
        }}
        """).strip()

        raw = run_model(prompt, max_new_tokens=400, num_beams=4, temperature=0.7)
        parsed = extract_json_from_text(raw)
        if parsed is None:
            # fallback: use the plain description packaged as guide
            parsed = {
                "id": step.get("id"),
                "title": title,
                "guide": description,
                "commands": step.get("commands", []),
                "verification": "Check connectivity and logs.",
                "rollback": "Reinstall previous driver or undo changes."
            }
        expanded.append(parsed)
    return expanded


def plan_and_generate(user_text: str) -> Dict[str, Any]:
    """
    Main orchestrator: run requirement extraction, plan generation, step expansion,
    and return a single structured result.
    """
    reqs = extract_requirements(user_text)
    plan = generate_plan(reqs)
    steps = plan.get("steps", [])
    expanded = expand_steps(steps, reqs)

    result = {
        "requirements": reqs,
        "plan": plan,
        "expanded_steps": expanded
    }
    return result


# ---------- quick helper: pretty human text ----------
def to_human_report(struct: Dict[str, Any]) -> str:
    """Convert the planner output into a readable multi-section text."""
    lines = []
    reqs = struct["requirements"]
    lines.append("=== Extracted Requirements ===")
    for k, v in reqs.items():
        lines.append(f"- {k}: {v if v else '—'}")
    lines.append("\n=== Plan (summary) ===")
    for s in struct["plan"]["steps"]:
        lines.append(f"{s['id']}. {s['title']}  (risk: {s.get('risk','?')}, ~{s.get('estimated_minutes','?')}m)")
    lines.append("\n=== Detailed Steps ===")
    for s in struct["expanded_steps"]:
        lines.append(f"\n{ s['id'] }. { s['title'] }\n")
        lines.append(s.get("guide",""))
        cmds = s.get("commands",[])
        if cmds:
            lines.append("\nCommands / Examples:")
            for c in cmds:
                lines.append(f"  {c}")
        lines.append("\nVerification: " + s.get("verification",""))
        lines.append("\nRollback: " + s.get("rollback",""))
    return "\n".join(lines)
