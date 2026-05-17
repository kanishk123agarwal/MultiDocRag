import os
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

CONFLICT_PROMPT = """You are an expert fact-checker. Below are answers to the same question, each from a different document.

Question: {question}

{answers}

Your task:
1. Determine if any two answers CONTRADICT each other on a specific factual claim.
2. If there is a conflict, explain exactly what the conflict is and which documents disagree.
3. If there is no conflict (answers agree or cover different aspects), say "No conflict detected."

Respond in this exact format:
CONFLICT_FOUND: yes/no
EXPLANATION: <your explanation>
DOCUMENT_A: <name of first conflicting doc, if any>
DOCUMENT_B: <name of second conflicting doc, if any>
CLAIM_A: <what document A claims>
CLAIM_B: <what document B claims>"""

def detect_conflicts(question: str, doc_answers: dict) -> dict:
    """
    Takes per-document answers and detects contradictions.
    Returns structured conflict result.
    """
    answers_text = "\n\n".join(
        [f"[{doc}]:\n{answer}" for doc, answer in doc_answers.items()]
    )

    prompt = CONFLICT_PROMPT.format(question=question, answers=answers_text)

    llm = Gemini(model="models/gemini-2.5-flash")
    response = llm.complete(prompt)
    raw = response.text

    result = {
        "raw": raw,
        "conflict_found": False,
        "explanation": "",
        "doc_a": "",
        "doc_b": "",
        "claim_a": "",
        "claim_b": ""
    }

    # Robust parser for colon-separated key-value lines
    def parse_field(line, prefix):
        if line.upper().startswith(prefix):
            return line[len(prefix):].strip()
        return None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.upper().startswith("CONFLICT_FOUND:"):
            val = line[len("CONFLICT_FOUND:"):].strip().lower()
            result["conflict_found"] = "yes" in val
            continue

        for field, key in [
            ("EXPLANATION:", "explanation"),
            ("DOCUMENT_A:", "doc_a"),
            ("DOCUMENT_B:", "doc_b"),
            ("CLAIM_A:", "claim_a"),
            ("CLAIM_B:", "claim_b")
        ]:
            val = parse_field(line, field)
            if val is not None:
                result[key] = val
                break

    return result
