"""
Pre-train a 32K WordPiece tokenizer on the full book corpus + reasoning data.

Extracts text from all 145 books (EPUB/PDF) and 50K JSONL reasoning entries,
then trains an MLXTokenizer and saves it for use in the production training run.

Docs: https://huggingface.co/docs/tokenizers/
      https://ml-explore.github.io/mlx/

Sample input:
    python train_tokenizer.py

Expected output:
    runs/tokenizer_32k.json  # Trained tokenizer with ~32K vocab
"""

import json
import os
import time
from typing import List

from loguru import logger

from mlx_data import discover_books, extract_epub_text, extract_pdf_text
from mlx_tokenizer import MLXTokenizer


BOOK_DIR = "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books"
JSONL_PATH = "data/reasoning_50k/reasoning_data.jsonl"
OUTPUT_PATH = "runs/tokenizer_32k.json"
VOCAB_SIZE = 32000


def collect_book_texts(book_dir: str) -> List[str]:
    """Extract text from all books and return as list of text chunks."""
    paths = discover_books(book_dir)
    texts: List[str] = []
    for i, path in enumerate(paths):
        ext = path.lower().rsplit(".", 1)[-1]
        if ext == "epub":
            text = extract_epub_text(path)
        elif ext == "pdf":
            text = extract_pdf_text(path)
        else:
            continue

        if not text or len(text) < 100:
            logger.warning(f"Skipped {path}: too short ({len(text)} chars)")
            continue

        # Split into ~5K char chunks for tokenizer training
        for start in range(0, len(text), 5000):
            chunk = text[start : start + 5000]
            if len(chunk) > 50:
                texts.append(chunk)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(paths)} books, {len(texts)} chunks so far")

    logger.info(f"Total book chunks: {len(texts)}")
    return texts


def collect_reasoning_texts(jsonl_path: str) -> List[str]:
    """Load reasoning examples as formatted text strings."""
    texts: List[str] = []
    with open(os.path.expanduser(jsonl_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            parts = []
            if "instruction" in entry:
                parts.append(entry["instruction"])
            if "reasoning" in entry:
                parts.append(entry["reasoning"])
            if "response" in entry:
                parts.append(entry["response"])
            if parts:
                texts.append(" ".join(parts))

    logger.info(f"Total reasoning examples: {len(texts)}")
    return texts


if __name__ == "__main__":
    import sys

    start_time = time.time()

    logger.info("=== Tokenizer Pre-Training ===")
    logger.info(f"Target vocab size: {VOCAB_SIZE}")
    logger.info(f"Book dir: {BOOK_DIR}")
    logger.info(f"JSONL path: {JSONL_PATH}")

    # Collect all training texts
    logger.info("Extracting book texts...")
    book_texts = collect_book_texts(BOOK_DIR)

    logger.info("Loading reasoning data...")
    reasoning_texts = collect_reasoning_texts(JSONL_PATH)

    all_texts = book_texts + reasoning_texts
    total_chars = sum(len(t) for t in all_texts)
    logger.info(
        f"Total corpus: {len(all_texts)} chunks, {total_chars:,} chars "
        f"({len(book_texts)} book + {len(reasoning_texts)} reasoning)"
    )

    # Train tokenizer
    logger.info(f"Training tokenizer with vocab_size={VOCAB_SIZE}...")
    tokenizer = MLXTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(all_texts, vocab_size=VOCAB_SIZE, output_path=OUTPUT_PATH)

    elapsed = time.time() - start_time

    # Validation
    all_validation_failures: List[str] = []
    total_tests = 0

    # Test 1: Vocab size is near target
    total_tests += 1
    if tokenizer.vocab_size < VOCAB_SIZE * 0.8:
        all_validation_failures.append(
            f"Vocab size {tokenizer.vocab_size} is <80% of target {VOCAB_SIZE}"
        )

    # Test 2: Encode/decode roundtrip
    total_tests += 1
    test_text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    if "quick" not in decoded.lower() or "fox" not in decoded.lower():
        all_validation_failures.append(f"Roundtrip failed: '{decoded}'")

    # Test 3: Special tokens work
    total_tests += 1
    think_ids = tokenizer.encode("<think>reasoning here</think>")
    think_decoded = tokenizer.decode(think_ids)
    if "think" not in think_decoded.lower():
        all_validation_failures.append(f"Special tokens lost: '{think_decoded}'")

    # Test 4: File saved
    total_tests += 1
    if not os.path.isfile(OUTPUT_PATH):
        all_validation_failures.append(f"Output file not found: {OUTPUT_PATH}")

    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for f in all_validation_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\n✅ Tokenizer trained successfully in {elapsed:.1f}s")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Corpus: {len(all_texts)} chunks ({total_chars:,} chars)")
        print(f"  Saved to: {OUTPUT_PATH}")
        sys.exit(0)
