import os
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

CONFLICT_PROMPT = """You are an expert research analyst comparing findings from multiple documents.

Question asked: {question}

Below are the answers from each document:

{answers}

Your task is to find CONFLICTS between these documents. A conflict exists when documents lead to DIFFERENT or INCOMPATIBLE conclusions about the same topic. Conflicts can be:

1. DIRECT: One says "X is true", another says "X is false"
2. CAPABILITY: One says the system CAN do something (e.g. works in real-time), the other says it CANNOT or is "not yet ready"
3. SENSOR/METHOD: One requires specific hardware (e.g. EEG headset/brain sensor), another requires different hardware (e.g. webcam/camera) — implying incompatible technical requirements
4. QUANTITATIVE: They report significantly different numbers for the same metric (e.g. different accuracy figures)
5. SCOPE/APPLICABILITY: One says the approach is ready for real-world use, another says it only works in controlled/lab conditions

IMPORTANT RULES:
- Be AGGRESSIVE in finding conflicts. Even if not explicitly stated, if a reader would reach DIFFERENT conclusions from each document about the same question, that IS a conflict.
- If Document A describes a system that requires X, and Document B describes a system that does NOT require X and uses Y instead — this IS a conflict about what is needed.
- If one system is described as real-time and another as offline-only — this IS a conflict.
- Do NOT say "no conflict" just because papers use polite academic language. Look at what the systems actually DO and REQUIRE.

Respond in this EXACT format (do not add extra lines or change the keys):
CONFLICT_FOUND: yes/no
EXPLANATION: <clear explanation of what exactly conflicts and why>
DOCUMENT_A: <filename of first conflicting document>
DOCUMENT_B: <filename of second conflicting document>
CLAIM_A: <the specific claim or finding from Document A>
CLAIM_B: <the specific claim or finding from Document B that contradicts Claim A>"""


def detect_conflicts(question: str, doc_answers: dict) -> dict:
    """
    Takes per-document answers and detects contradictions.
    Returns structured conflict result.
    """
    answers_text = "\n\n".join(
        [f"[{doc}]:\n{answer}" for doc, answer in doc_answers.items()]
    )

    prompt = CONFLICT_PROMPT.format(question=question, answers=answers_text)

    llm = Gemini(model="models/gemini-3.5-flash")
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
