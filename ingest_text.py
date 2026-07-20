"""Orchestrator — chunks a text file and ingests all chunks into the Chroma vector database.

Uses chunker.split_into_chunks() for splitting and Gemini embeddings for storage.
Run this before querying with chroma_store.py --mode query.
"""

import os
import sys
import argparse
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import chromadb
from chunker import split_into_chunks, load_text
from document_loader import load_pdf

load_dotenv()
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "support-docs"
CHROMA_PATH = "./chroma_db"


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


def ingest_file(
    genai_client: object,
    file_path: str,
    chunk_size: int,
    overlap: int,
    is_pdf: bool = False,
) -> None:
    """Chunk a text file and store each chunk's embedding in Chroma.

    Args:
        genai_client: Initialized genai.Client instance.
        file_path: Path to the .txt file to ingest.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.
        is_pdf: If True, use load_pdf() for extraction. If False, use load_text(). Default: False.

    Returns:
        None.

    Raises:
        SystemExit: If the file cannot be loaded, chunking parameters are
            invalid, no chunks are produced, or the collection already
            contains entries.
    """
    if is_pdf:
        text = load_pdf(file_path)
    else:
        text = load_text(file_path)
    if text is None:
        sys.exit(1)
    try:
        chunks = split_into_chunks(
            text,
            source_file=os.path.basename(file_path),
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not chunks:
        print("❌ No chunks produced.", file=sys.stderr)
        sys.exit(1)

    print(
        f"📄 Loaded '{os.path.basename(file_path)}' → {len(chunks)} chunks (size={chunk_size}, overlap={overlap})"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        print(
            f"⚠️  Collection already has {collection.count()} entries. Delete chroma_db/ to re-ingest.",
            file=sys.stderr,
        )
        sys.exit(0)

    ingested = 0
    for i, chunk in enumerate(chunks):
        print(f"⚙️  Embedding chunk {i+1}/{len(chunks)}...", file=sys.stderr, end="\r")
        embedding = get_embedding(genai_client, chunk["text"])
        if embedding is None:
            print(f"\n⚠️  Skipping chunk {i} — embedding failed.", file=sys.stderr)
            continue
        collection.add(
            ids=[f"{os.path.basename(file_path)}_chunk_{chunk['chunk_id']}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "source_file": chunk["source_file"],
                    "chunk_id": chunk["chunk_id"],
                    "char_start": chunk["char_start"],
                }
            ],
        )
        ingested += 1

    print(
        f"\n✅ Ingested {ingested}/{len(chunks)} chunks into '{COLLECTION_NAME}' collection."
    )
    print(
        f'   Now run: python3 chroma_store.py --mode query --question "your question here"'
    )


def main() -> None:
    """Parse command-line arguments and run the ingestion pipeline.

    Returns:
        None.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "❌ Error: GEMINI_API_KEY not set. Add it to your .env file.",
                file=sys.stderr,
            )
            sys.exit(1)

        parser = argparse.ArgumentParser(
            description="Chunk a text file and store embeddings in Chroma."
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--file", type=str, help="Path to a .txt file to ingest")
        group.add_argument("--pdf", type=str, help="Path to a .pdf file to ingest")
        parser.add_argument(
            "--size", type=int, default=500, help="Characters per chunk (default: 500)"
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=100,
            help="Overlap between chunks (default: 100)",
        )
        args = parser.parse_args()

        genai_client = genai.Client(api_key=api_key)
        if args.pdf:
            ingest_file(genai_client, args.pdf, args.size, args.overlap, is_pdf=True)
        else:
            ingest_file(genai_client, args.file, args.size, args.overlap, is_pdf=False)
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
