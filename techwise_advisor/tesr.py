import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Model Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"  # larger model for better reasoning
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# -------------------------
# Generate Reasoning-Based Response
# -------------------------
def generate_hint_response(issue: str) -> str:
    """
    Generate a Windows-specific reasoning-based response from the model.
    The output highlights possible causes and hints at the problem without giving step-by-step instructions.
    """
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
# Test Multiple Windows Issues
# -------------------------
issues = [
    "Windows 11 battery drains very fast even when idle and the laptop overheats during light tasks",
    "Windows 11 Wi-Fi disconnects randomly after waking from sleep",
    "After updating Windows 11, some USB devices (keyboard and mouse) stop working intermittently",
    "Windows 11 laptop screen flickers randomly during light tasks",
    "Printer connected to Windows 11 network is not detected by multiple computers"
]

for issue in issues:
    response = generate_hint_response(issue)
    print("\n" + "="*100 + "\n")
    print(f"Issue: {issue}\n")
    print(response)
    print("\n" + "="*100 + "\n")
