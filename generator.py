"""Grounded answer generation module.
Takes a question and retrieved context chunks, builds a grounded prompt,
calls Gemini, and returns an answer with source citations.
Answers are grounded: Gemini is instructed to use only the provided context."""

import os
import sys
from urllib import response
from xmlrpc import client
from google import genai
from dotenv import load_dotenv

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not _api_key:
    print(
        "❌ Error: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.",
        file=sys.stderr,
    )
    sys.exit(1)


GEN_MODEL_FALLBACKS = [
    "gemini-3.1-flash-lite",  # confirmed working on your account
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


GENERATION_PROMPT = """You are a document Q&A assistant. Answer the user's question using ONLY the context passages provided below. Do not use any knowledge outside these passages.

If the passages do not contain enough information to answer the question, say exactly: "I don't have enough information in the provided documents to answer this."

For each fact in your answer, cite the source document in parentheses like (Source: filename.pdf). If multiple passages support your answer, cite all relevant sources.
Keep your answer concise and direct.

---CONTEXT PASSAGES---
{context_block}
---END CONTEXT---

User question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from retrieved context chunks using Gemini.

    Args:
        question: The user's question as a plain string.
        chunks: List of dicts, each with keys "text" (str), "source" (str),
            and "distance" (float). This is the format returned by
            chroma_store.py's query function.

    Returns:
        The generated answer string from Gemini. Each model in
        GEN_MODEL_FALLBACKS is tried in order; the first successful response
        is returned. If every model fails, returns an error message string
        (does not raise).

    Raises:
        Does not raise. All exceptions are caught and returned as error
        message strings.
    """
    if not chunks:
        return "No context was retrieved. Cannot generate an answer."

    context_block = "\n".join(
        f"[CHUNK {i + 1} — Source: {chunk['source']}]\n{chunk['text']}\n"
        for i, chunk in enumerate(chunks)
    )

    full_prompt = GENERATION_PROMPT.format(
        context_block=context_block, question=question
    )

    last_error = None
    for model_name in GEN_MODEL_FALLBACKS:
        try:
            client = genai.Client(api_key=_api_key)
            response = client.models.generate_content(
                model=model_name, contents=full_prompt
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            print(
                f"⚠️  Model '{model_name}' failed: {e} — trying next fallback...",
                file=sys.stderr,
            )
            continue

    print(
        f"❌ Generation error: all models failed. Last error: {last_error}",
        file=sys.stderr,
    )
    return f"Generation failed: {last_error}"


def main() -> None:
    """CLI test runner for grounded answer generation.

    Usage:
        python3 generator.py --question "your question here"

    Parses the ``--question`` argument, builds hardcoded mock chunks as test
    fixtures, calls generate_answer(), and prints the result to stdout.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test answer generation with mock chunks."
    )
    parser.add_argument("--question", required=True, help="Question to answer")
    args = parser.parse_args()

    # Use 2 mock chunks for the standalone test — these are hardcoded test fixtures
    mock_chunks = [
        {
            "text": "Refund requests must be submitted within 30 days of purchase. Contact billing@support.com with your invoice number.",
            "source": "sample.pdf",
            "distance": 0.15,
        },
        {
            "text": "All accounts include a 14-day free trial. No credit card required to start.",
            "source": "sample.pdf",
            "distance": 0.31,
        },
    ]

    try:
        print(f"\n📋 Mock chunks provided: {len(mock_chunks)}")
        print(f"❓ Question: {args.question}")
        print("\n⚙️  Calling Gemini for grounded answer...\n")
        answer = generate_answer(args.question, mock_chunks)
        print("─" * 60)
        print(answer)
        print("─" * 60)
    except KeyboardInterrupt:
        print("\n👋 Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
