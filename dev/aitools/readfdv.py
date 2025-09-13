import sys
import io
import re
import argparse
from pathlib import Path

def extract_text(pdf_path):
    # Try pdfplumber first, fall back to PyPDF2
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as e:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
            return "\n".join(text_parts)
        except Exception as e2:
            raise RuntimeError("Failed to extract text. Install pdfplumber or PyPDF2.") from e2

def find_run_syntax_lines(text, ctx=2):
    lines = text.splitlines()
    matches = []

    # Common patterns that document syntax or usage of 'run' operator
    # Adjust or add as needed for your PDF formatting
    patterns = [
        # e.g., "run=<count>", "run=<N>", "run=<iterations>"
        re.compile(r'^\s*(?:- )?\s*run\s*=\s*[^,\s;]+.*$', flags=re.IGNORECASE),
        # e.g., "run <count>", "run <N>"
        re.compile(r'^\s*(?:- )?\s*run\s+[^\s,;]+.*$', flags=re.IGNORECASE),
        # "syntax: run=..." or "operator: run ..."
        re.compile(r'.*\b(syntax|operator)\b.*\brun\b.*', flags=re.IGNORECASE),
        # general lines that likely include run usage
        re.compile(r'^\s*run\b.*$', flags=re.IGNORECASE),
    ]

    # Also capture surrounding context when a line contains keywords
    keyword = re.compile(r'\brun\b', flags=re.IGNORECASE)

    for i, line in enumerate(lines):
        hit = any(p.search(line) for p in patterns)
        if not hit and not keyword.search(line):
            continue

        # Collect context
        start = max(0, i - ctx)
        end = min(len(lines), i + ctx + 1)
        block = lines[start:end]
        # Deduplicate blocks by content
        block_text = "\n".join(block).strip()
        if block_text and all(block_text != m['text'] for m in matches):
            matches.append({
                'line': i + 1,
                'text': block_text
            })

    return matches

def main():
    ap = argparse.ArgumentParser(description="Extract FDV 'run' operator syntax from PDF.")
    ap.add_argument("-i", "--input", default=str(Path.cwd() / "fdvfsm_rev33.pdf"),
                    help="Path to fdvfsm_rev33.pdf (default: ./fdvfsm_rev33.pdf)")
    ap.add_argument("-o", "--output", default=None,
                    help="Optional path to save extracted matches as a text file. Default: <pdf_dir>/fdv_run_syntax.txt")
    ap.add_argument("--context", type=int, default=2, help="Context lines before/after each hit (default: 2)")
    args = ap.parse_args()

    pdf_path = Path(args.input).resolve()
    if not pdf_path.exists():
        print("ERROR: PDF not found:", pdf_path)
        sys.exit(1)

    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print("ERROR:", e)
        print("Tip: pip install pdfplumber or PyPDF2")
        sys.exit(2)

    matches = find_run_syntax_lines(text, ctx=args.context)
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = pdf_path.with_name("fdv_run_syntax.txt")

    # Build report
    buf = io.StringIO()
    buf.write("FDV 'run' operator syntax/context extracted from: {}\n".format(pdf_path))
    buf.write("=" * 80 + "\n\n")
    if not matches:
        buf.write("No lines containing 'run' were found. Try increasing --context or adjusting patterns.\n")
    else:
        for idx, m in enumerate(matches, 1):
            buf.write("[Match {} at around line {}]\n".format(idx, m['line']))
            buf.write(m['text'] + "\n")
            buf.write("-" * 40 + "\n")

    report = buf.getvalue()
    print(report)

    try:
        with io.open(str(out_path), "w", encoding="utf-8") as f:
            f.write(report)
        print("Saved:", out_path)
    except Exception as e:
        print("WARN: Could not save output:", e)

if __name__ == "__main__":
    main()