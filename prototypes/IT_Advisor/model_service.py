import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import textwrap

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "google/flan-t5-large"

logging.info(f"Loading {MODEL_NAME} on CPU (this may take a minute)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 512


# ---------------------------------------------------
# Chunk long text safely
# ---------------------------------------------------
def chunk_text(text: str, max_tokens: int = 900):
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    count = 0

    for token in tokens:
        current_chunk.append(token)
        count += 1
        if count >= max_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk, count = [], 0
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    return chunks


# ---------------------------------------------------
# Generate structured, complete answers
# ---------------------------------------------------
def generate_answer(question: str, style: str = "paragraph") -> str:
    question = question.strip()
    if not question:
        return "⚠️ Please enter a valid IT-related question."

    try:
        logging.info(f"Received: {question}")

        # Chunking for long inputs
        chunks = chunk_text(question)
        if len(chunks) > 1:
            summaries = []
            for i, chunk in enumerate(chunks):
                sub_prompt = f"Summarize part {i+1}/{len(chunks)}:\n{chunk}"
                inputs = tokenizer(sub_prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
                ids = model.generate(**inputs, max_new_tokens=256)
                summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
            question = " ".join(summaries)
            logging.info("Summarized long input into shorter context.")

        # Reinforced prompt with structure and length guidance
        prompt = textwrap.dedent(f"""
        You are **TechWISE Advisor**, an advanced IT troubleshooting system.

        Task:
        Analyze the user's IT problem deeply and generate a complete explanation with **three detailed sections**:
        1️⃣ **Likely Root Causes** — explain 2–3 possible technical reasons for the issue.
        2️⃣ **Step-by-Step Troubleshooting** — give at least 3 detailed actions users can take to fix it.
        3️⃣ **Preventive Recommendations** — provide tips to avoid this issue in the future.

        Each section must be clear, structured, and at least 2–3 sentences long.
        Use Markdown-style formatting with bullet points and code snippets if relevant.
        Avoid one-line summaries.

        User question:
        {question}
        """).strip()

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            num_beams=6,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.8,
            length_penalty=1.5,
            do_sample=True,
            no_repeat_ngram_size=3,
        )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Retry if the model produced too little
        if len(answer.split()) < 40:
            logging.warning("Short answer detected. Retrying with reformulated prompt...")
            reprompt = prompt + "\n\nPlease expand your response into a detailed professional guide."
            inputs = tokenizer(reprompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                num_beams=5,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.6,
            )
            answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return answer

    except torch.cuda.OutOfMemoryError:
        return "❌ Model ran out of memory. Try using a smaller model like `flan-t5-base`."
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return f"❌ Internal model error: {e}"
