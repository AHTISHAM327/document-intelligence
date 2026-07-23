"""Single-command Q&A CLI.
Wraps PDF ingestion, semantic retrieval, and grounded answer generation
behind one command: python3 qa.py --pdf FILE --question QUESTION"""

import os
import sys
import shutil
from dotenv import load_dotenv
from google import genai
from ingest_text import ingest_file
from chroma_store import get_chroma_collection, query_collection
from generator import generate_answer

load_dotenv()

# Defaults mirror ingest_text.py's CLI (size=500, overlap=100).
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


def run_qa(
    pdf_path: str,
    question: str,
    top_k: int = 3,
    force_reingest: bool = False,
) -> str:
    """Run the full RAG pipeline for one PDF and one question.

    Ingests the PDF into Chroma if no matching index exists, retrieves the
    most relevant chunks, and generates a grounded answer with citations.

    Args:
        pdf_path: Path to the PDF file to answer questions about.
        question: The user's question.
        top_k: Number of chunks to retrieve (default 3).
        force_reingest: If True, delete any existing chroma_db and re-ingest
            even if data already exists.

    Returns:
        The grounded answer string. Returns a clear error message string (not a
        raised exception) if the PDF file doesn't exist or ingestion fails.

    Raises:
        Does not raise for expected failure cases (missing file, ingestion
        failure) — returns an error string instead.
    """
    # Step 1 — Validate the PDF exists.
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return f"Error: PDF file not found: {pdf_path}"

    # Step 2 — Set up the genai client and Chroma collection.
    genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Step 3 — Decide whether to ingest.
    chroma_db_exists = os.path.isdir("chroma_db")
    requested_source = os.path.basename(pdf_path)

    if force_reingest and chroma_db_exists:
        shutil.rmtree("chroma_db")
        print(
            "🔄 Force re-ingest requested — cleared existing chroma_db",
            file=sys.stderr,
        )
        chroma_db_exists = False
    elif chroma_db_exists:
        # CRITICAL: an existing index may hold a DIFFERENT document. Detect that.
        # Open collection here only to read one chunk and compare source_file.
        try:
            _check_collection = get_chroma_collection()
            sample = _check_collection.get(limit=1)
            metadatas = sample.get("metadatas") or []
            indexed_source = metadatas[0].get("source_file") if metadatas else None
        except Exception:
            indexed_source = None
        if indexed_source is None:
            # Index exists but is empty or unreadable — treat as no index.
            shutil.rmtree("chroma_db")
            chroma_db_exists = False
        elif indexed_source != requested_source:
            print(
                f"🔄 Existing index holds '{indexed_source}' but you asked about "
                f"'{requested_source}' — re-ingesting",
                file=sys.stderr,
            )
            shutil.rmtree("chroma_db")
            chroma_db_exists = False

    if not chroma_db_exists:
        print(f"📥 No existing index found — ingesting {pdf_path}...", file=sys.stderr)
        # ingest_file signals failure by raising SystemExit (sys.exit(1) on
        # error, sys.exit(0) when the collection is already populated).
        try:
            ingest_file(
                genai_client,
                pdf_path,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_OVERLAP,
                is_pdf=True,
            )
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"❌ Error: ingestion failed for {pdf_path}", file=sys.stderr)
                return f"Error: ingestion failed for {pdf_path}"
    else:
        print(
            f"✅ Using existing index for '{requested_source}' "
            "(pass --reingest to force a fresh ingest)",
            file=sys.stderr,
        )

    # Step 4 — Open collection after ingestion is complete, then query.
    collection = get_chroma_collection()
    chunks = query_collection(genai_client, collection, question, top_k=top_k)
    # Step 5 — Generate.
    answer = generate_answer(question, chunks)
    return answer


def main() -> None:
    """CLI entry point for the single-command Q&A pipeline.

    Usage:
        python3 qa.py --pdf document.pdf --question "your question"
        python3 qa.py --pdf document.pdf --question "your question" --reingest
        python3 qa.py --pdf document.pdf --question "your question" --top-k 5

    Returns:
        None.
    """
    import argparse

    try:
        parser = argparse.ArgumentParser(
            description="Ask questions about a PDF document."
        )
        parser.add_argument("--pdf", required=True, help="Path to the PDF file")
        parser.add_argument(
            "--question",
            required=True,
            help="Question to ask about the document",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=3,
            dest="top_k",
            help="Number of chunks to retrieve (default: 3)",
        )
        parser.add_argument(
            "--reingest",
            action="store_true",
            help="Force re-ingestion even if an index already exists",
        )
        args = parser.parse_args()

        print(f"\n📄 Document: {args.pdf}")
        print(f"❓ Question: {args.question}\n")

        answer = run_qa(
            args.pdf,
            args.question,
            top_k=args.top_k,
            force_reingest=args.reingest,
        )

        print("─" * 60)
        print(answer)
        print("─" * 60)
    except KeyboardInterrupt:
        print("\n👋 Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
