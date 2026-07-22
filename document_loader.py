"""PDF text extraction module.

Provides load_pdf() for extracting clean text from standard PDF files.
Does not support scanned/image PDFs (those require OCR).
"""

import os
import sys
import PyPDF2


def load_pdf(file_path: str) -> str | None:
    """Extract clean text from a PDF file.

    Args:
        file_path: Absolute or relative path to a .pdf file.

    Returns:
        Extracted text as a single string with pages joined by double
        newlines. Returns None if extraction fails for any reason.

    Raises:
        Does not raise. All exceptions are caught and printed to stderr.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}", file=sys.stderr)
        return None
    if not file_path.lower().endswith(".pdf"):
        print(f"❌ Error: File is not a PDF: {file_path}", file=sys.stderr)
        return None

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            if total_pages == 0:
                print(f"❌ Error: PDF has no pages: {file_path}", file=sys.stderr)
                return None
            print(
                f"📄 Reading PDF: {os.path.basename(file_path)} ({total_pages} pages)",
                file=sys.stderr,
            )
            pages = []
            for i, page in enumerate(reader.pages):
                print(
                    f"   Extracting page {i+1}/{total_pages}...",
                    end="\r",
                    file=sys.stderr,
                )
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text.strip())
                except Exception as e:
                    print(
                        f"\n⚠️  Warning: Could not extract page {i+1}: {e}",
                        file=sys.stderr,
                    )
                    continue
    except PyPDF2.errors.PdfReadError as e:
        print(
            f"❌ Error: Cannot read PDF (file may be corrupted or encrypted): {e}",
            file=sys.stderr,
        )
        return None
    except PermissionError:
        print(
            f"❌ Error: Cannot read file (permission denied): {file_path}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"❌ Unexpected error reading PDF: {e}", file=sys.stderr)
        return None

    if not pages:
        print(
            f"\n❌ Error: Could not extract any text from PDF. It may be a scanned/image PDF.",
            file=sys.stderr,
        )
        print(
            "   PyPDF2 only works with text-based PDFs, not scanned images.",
            file=sys.stderr,
        )
        return None

    full_text = "\n\n".join(pages)
    full_text = full_text.strip()
    print(
        f"\n✅ Extracted {len(full_text)} characters from {total_pages} pages",
        file=sys.stderr,
    )
    return full_text


def main() -> None:
    """Quick test runner for PDF extraction.

    Lets you call "python3 document_loader.py path/to/file.pdf" to verify
    that text extraction works, printing a preview of the extracted text.
    """
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Test PDF text extraction.")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--file", help="Path to a .txt file to read")
        group.add_argument("--pdf", help="Path to a .pdf file to extract")
        parser.add_argument(
            "--chars",
            type=int,
            default=500,
            help="How many characters of output to print (default: 500)",
        )
        args = parser.parse_args()

        if args.pdf:
            text = load_pdf(args.pdf)
        else:
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                print(f"❌ Error: Cannot read file: {e}", file=sys.stderr)
                sys.exit(1)

        if text is None:
            sys.exit(1)

        print(f"\n--- Extracted text preview (first {args.chars} chars) ---")
        print(text[: args.chars])
        print(f"\n--- Total: {len(text)} chars, roughly {len(text.split())} words ---")
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
