"""Two-mode Chroma demo. --mode ingest: embed text corpus and store in local Chroma DB.
--mode query: search stored embeddings by semantic similarity using Gemini embeddings.
"""

import os
import sys
import argparse
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import chromadb

load_dotenv()
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "support-docs"
CHROMA_PATH = "./chroma_db"

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


def get_chroma_collection() -> chromadb.Collection:
    """Open the persistent Chroma database and return the demo collection.

    PersistentClient writes to CHROMA_PATH on disk. get_or_create_collection
    creates the collection on first run and loads the existing one on
    subsequent runs.

    Args:
        None

    Returns:
        Chroma Collection object.

    Raises:
        Does not raise.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedding(genai_client: object, text: str) -> list[float] | None:
    """Embed a single piece of text with the Gemini embeddings API.

    Args:
        genai_client: Initialized genai.Client instance.
        text: Text to embed.

    Returns:
        List of embedding floats, or None if the API call failed.

    Raises:
        Does not raise; API and network errors are printed to stderr.
    """
    try:
        result = genai_client.models.embed_content(model=EMBED_MODEL, contents=text)
        return result.embeddings[0].values
    except genai_errors.ClientError as e:
        if e.code == 429:
            print("❌ Rate limit hit. Wait a minute and retry.", file=sys.stderr)
        else:
            print(f"❌ API error: {e.message}", file=sys.stderr)
        return None
    except genai_errors.ServerError as e:
        print(f"❌ Server error: {e.message}", file=sys.stderr)
        return None
    except httpx.RequestError:
        print("❌ Network error. Check your connection.", file=sys.stderr)
        return None


def ingest_corpus(genai_client: object, collection: chromadb.Collection) -> None:
    """Embed each corpus sentence and store it in the Chroma collection.

    Skips ingestion entirely if the collection already contains entries.

    Args:
        genai_client: Initialized genai.Client instance.
        collection: Chroma collection to write into.

    Returns:
        None

    Raises:
        Does not raise; per-entry embedding failures are skipped with a warning.
    """
    if collection.count() > 0:
        print(
            f"⚠️ Collection already has {collection.count()} entries. "
            "Skipping ingest. Delete chroma_db/ to re-ingest."
        )
        return
    for i, sentence in enumerate(CORPUS):
        print(f"⚙️ Ingesting {i+1}/{len(CORPUS)}: {sentence[:40]}...", file=sys.stderr)
        embedding = get_embedding(genai_client, sentence)
        if embedding is None:
            print(f"❌ Skipping entry {i} — embedding failed.", file=sys.stderr)
            continue
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[sentence],
            metadatas=[{"source": "corpus", "chunk_id": i}],
        )
    print(
        f"✅ Ingested {collection.count()} entries into '{COLLECTION_NAME}' collection."
    )


def query_collection(
    genai_client: object,
    collection: chromadb.Collection,
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Embed a question, print the most similar stored documents, and return them.

    Args:
        genai_client: Initialized genai.Client instance.
        collection: Chroma collection to search.
        question: Natural-language question to search for.
        top_k: Number of results to return.

    Returns:
        List of result dicts, one per hit, each with keys: rank (int, 1-based),
        similarity (float), document (str), metadata (dict). Empty list if the
        collection is empty or the query could not be embedded. Callers can
        iterate the result safely without a None check.

    Raises:
        Does not raise; failures are printed to stderr.
    """
    if collection.count() == 0:
        print(
            "❌ No data in collection. Run with --mode ingest first.", file=sys.stderr
        )
        return []
    print(f"\n🔍 Query: {question}\n")
    embedding = get_embedding(genai_client, question)
    if embedding is None:
        print("❌ Could not embed query.", file=sys.stderr)
        return []
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    print("Rank | Similarity | Result")
    print("-" * 60)
    hits = []
    for rank, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
        # Chroma returns L2 distance, not similarity — convert here
        similarity = 1 - dist
        print(f"{rank+1:>4} | {similarity:.4f}     | {doc}")
        hits.append(
            {
                "rank": rank + 1,
                "similarity": similarity,
                "document": doc,
                "metadata": meta,
            }
        )
    print(f"\nSource metadata: {metadatas}")
    return hits


def main() -> None:
    """Parse CLI arguments and dispatch to ingest or query mode.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: If GEMINI_API_KEY is missing or arguments are invalid.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "❌ Error: GEMINI_API_KEY not set. Add it to your .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
        parser = argparse.ArgumentParser(description="Chroma semantic search demo")
        parser.add_argument(
            "--mode",
            choices=["ingest", "query"],
            required=True,
            help="ingest: store corpus in Chroma. query: search by semantic similarity.",
        )
        parser.add_argument(
            "--question",
            type=str,
            help="Question to search for (required when --mode query)",
        )
        args = parser.parse_args()
        if args.mode == "query" and not args.question:
            print("❌ --question is required when using --mode query", file=sys.stderr)
            sys.exit(1)
        genai_client = genai.Client(api_key=api_key)
        collection = get_chroma_collection()
        if args.mode == "ingest":
            ingest_corpus(genai_client, collection)
        if args.mode == "query":
            query_collection(genai_client, collection, args.question)
    except KeyboardInterrupt:
        print("\n\n👋 Caught Ctrl+C — stopping cleanly. Your Chroma data is safe on disk.")
        print("   Thanks for using the semantic search demo. Goodbye! ✨")
        # 130 = conventional exit code for termination by SIGINT (128 + 2)
        sys.exit(130)


if __name__ == "__main__":
    main()
