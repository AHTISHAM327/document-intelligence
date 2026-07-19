import os
import sys
import math
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()
EMBED_MODEL = "text-embedding-004"

CORPUS = [
    "I can't log into my account, it keeps rejecting me.",
    "The billing page shows a charge I don't recognize.",
    "How do I export my dashboard as a PDF report?",
    "Authentication fails every time I enter my credentials.",
    "The pasta at that new Italian restaurant was incredible.",
    "Our quarterly revenue exceeded projections by 12 percent.",
    "Password reset emails are not arriving in my inbox.",
    "The hiking trail offers stunning views of the valley.",
]


def get_embedding(client: object, text: str) -> list[float] | None:
    """Embed a single piece of text using the Gemini embeddings API.

    Args:
        client: An initialized genai.Client instance.
        text: The text to embed.

    Returns:
        The embedding vector as a list of floats, or None if the API
        call failed.

    Raises:
        Nothing: all expected API and network errors are caught and
        reported to stderr.
    """
    try:
        result = client.models.embed_content(model=EMBED_MODEL, contents=text)
        return result.embeddings[0].values
    except genai_errors.ClientError as e:
        if e.code == 429:
            print("❌ Rate limit hit. Wait a minute.", file=sys.stderr)
        else:
            print(f"❌ API error: {e.message}", file=sys.stderr)
        return None
    except genai_errors.ServerError as e:
        print(f"❌ Server error: {e.message}", file=sys.stderr)
        return None
    except httpx.RequestError:
        print("❌ Network error. Check your connection.", file=sys.stderr)
        return None


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        vec_a: The first vector.
        vec_b: The second vector.

    Returns:
        The cosine similarity in [-1.0, 1.0], or 0.0 if either vector
        has zero magnitude.

    Raises:
        Nothing.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_corpus(client: object, query: str) -> list[tuple[float, str]] | None:
    """Rank every corpus sentence by similarity to the query.

    Args:
        client: An initialized genai.Client instance.
        query: The search query to compare against the corpus.

    Returns:
        A list of (score, sentence) tuples sorted descending by score,
        or None if the query itself could not be embedded. Sentences
        that fail to embed are skipped with a warning.

    Raises:
        Nothing: embedding failures are handled by get_embedding.
    """
    query_embedding = get_embedding(client, query)
    if query_embedding is None:
        return None

    results = []
    for i, sentence in enumerate(CORPUS):
        print(f"⚙️ Embedding {i + 1}/{len(CORPUS)}...", file=sys.stderr)
        sentence_embedding = get_embedding(client, sentence)
        if sentence_embedding is None:
            print(f"⚠️ Skipping sentence {i + 1}: embedding failed.", file=sys.stderr)
            continue
        score = cosine_similarity(query_embedding, sentence_embedding)
        results.append((score, sentence))

    return sorted(results, key=lambda pair: pair[0], reverse=True)


def main() -> None:
    """Run the semantic search demo against the built-in corpus.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If GEMINI_API_KEY is missing or ranking fails.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "❌ Error: GEMINI_API_KEY not set. Add it to your .env file.",
                file=sys.stderr,
            )
            sys.exit(1)

        client = genai.Client(api_key=api_key)
        query = "my password doesn't work"
        print(f"🔍 Query: {query}\n")

        results = rank_corpus(client, query)
        if results is None:
            sys.exit(1)

        print("Rank | Score  | Sentence")
        for i, (score, sentence) in enumerate(results):
            print(f"{i + 1:>4} | {score:.4f} | {sentence}")
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
