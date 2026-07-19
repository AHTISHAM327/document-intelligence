"""Text chunker — splits long documents into overlapping chunks for embedding and vector storage.
Each chunk is a dict with 'text', 'source_file', 'chunk_id', and 'char_start' fields.
"""

import os
import sys
import json
import argparse


def load_text(file_path: str) -> str | None:
    """Read a text file and return its stripped contents.

    Args:
        file_path: Path to the text file to read.

    Returns:
        The file contents with leading/trailing whitespace stripped, or
        None if the file is missing, empty, or unreadable.

    Raises:
        Does not raise; errors are printed to stderr.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}", file=sys.stderr)
        return None
    try:
        with open(file_path, "r") as f:
            text = f.read().strip()
    except PermissionError:
        print(
            f"❌ Error: Cannot read file (permission denied): {file_path}",
            file=sys.stderr,
        )
        return None
    if not text:
        print(f"❌ Error: File is empty: {file_path}", file=sys.stderr)
        return None
    return text


def split_into_chunks(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Split text into overlapping chunks with positional metadata.

    Args:
        text: The full document text to split.
        source_file: The original filename — stored in metadata so we know
            where each chunk came from.
        chunk_size: Number of characters per chunk (default 500).
        overlap: Number of characters shared between consecutive chunks
            (default 100).

    Returns:
        List of dicts, each with keys: "text", "source_file", "chunk_id",
        and "char_start".

    Raises:
        ValueError: If chunk_size <= overlap (overlap must be smaller than
            chunk size).
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
        )
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source_file": source_file,
                    "chunk_id": chunk_id,
                    "char_start": start,
                }
            )
            chunk_id += 1
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def print_chunks(chunks: list[dict], verbose: bool = False) -> None:
    """Print a summary or full JSON dump of the produced chunks.

    Args:
        chunks: Non-empty list of chunk dicts from split_into_chunks.
        verbose: If True, print each chunk as formatted JSON; otherwise
            print a one-line summary per chunk.

    Returns:
        None

    Raises:
        Does not raise.
    """
    print(f"✅ Created {len(chunks)} chunks from '{chunks[0]['source_file']}'")
    print(
        f"   Chunk size: up to {max(len(c['text']) for c in chunks)} chars "
        "| Overlap: configured at call time"
    )
    print("─" * 60)
    for chunk in chunks:
        if verbose:
            print(json.dumps(chunk, indent=2))
        else:
            print(
                f"Chunk {chunk['chunk_id']:>3} | "
                f"chars {chunk['char_start']:>5}–"
                f"{chunk['char_start'] + len(chunk['text']):<5} | "
                f"{chunk['text'][:80]}..."
            )


def main() -> None:
    """Parse CLI arguments, chunk the input file, and print the results.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: If the file cannot be loaded, arguments are invalid,
            or no chunks are produced.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Split a text file into overlapping chunks for RAG embedding."
        )
        parser.add_argument(
            "--file", required=True, help="Path to the .txt file to chunk"
        )
        parser.add_argument(
            "--size",
            type=int,
            default=500,
            help="Characters per chunk (default: 500)",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=100,
            help="Overlap between chunks in characters (default: 100)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print full chunk JSON instead of summary",
        )
        args = parser.parse_args()
        text = load_text(args.file)
        if text is None:
            sys.exit(1)
        try:
            chunks = split_into_chunks(
                text,
                source_file=os.path.basename(args.file),
                chunk_size=args.size,
                overlap=args.overlap,
            )
        except ValueError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
        if not chunks:
            print("❌ No chunks produced — file may be too short.", file=sys.stderr)
            sys.exit(1)
        print_chunks(chunks, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
