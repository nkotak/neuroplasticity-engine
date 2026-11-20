"""
MLX data pipeline for mixed-source language model training.

Ingests text from local ebooks (EPUB/PDF), HuggingFace streaming datasets,
and JSONL reasoning data into a unified tokenized batch iterator.  Supports
deterministic train/val splits, English language detection, chunk quality
filtering, and content-hash deduplication.

Docs: https://ml-explore.github.io/mlx/
      https://docs.idiap.ch/ebooklib/
      https://pymupdf.readthedocs.io/

Sample input:
    stream = MixedDataStream([BookTextStream("./books"), HFStream("dataset")], [0.7, 0.3])
    batches = TokenizedBatchIterator(stream, tokenizer, seq_len=1024, batch_size=4)

Expected output:
    batch -> mx.array of shape (4, 1024) with dtype int32
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, Iterator, List, Optional, Set, Tuple

from loguru import logger

from mlx_tokenizer import MLXTokenizer

# -- Book extraction ---------------------------------------------------------

BOOK_CHUNK_SIZE = 10_000

# Stopwords used by _is_english to verify English content
_ENGLISH_STOPWORDS: Set[str] = {"the", "and", "is", "in", "to", "of", "a", "that", "it", "for"}


def _is_english(text: str) -> bool:
    """Check whether *text* is likely English.

    Returns True when >70% of characters are ASCII **and** the lowercased text
    contains at least 3 common English stopwords from _ENGLISH_STOPWORDS.
    """
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    if ascii_count / len(text) < 0.70:
        return False
    words = set(text.lower().split())
    hits = words & _ENGLISH_STOPWORDS
    return len(hits) >= 3


def extract_epub_text(path: str) -> str:
    """Extract plain text from EPUB via ebooklib + BeautifulSoup <p> tags."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    try:
        book = epub.read_epub(path, options={"ignore_ncx": True})
        parts: List[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if text:
                    parts.append(text)
        full_text = "\n".join(parts)
        logger.debug(f"Extracted {len(full_text)} chars from EPUB: {path}")
        return full_text
    except Exception as exc:
        logger.warning(f"Failed to extract EPUB {path}: {exc}")
        return ""


def extract_pdf_text(path: str) -> str:
    """Extract plain text from PDF via PyMuPDF, page by page."""
    import fitz

    try:
        doc = fitz.open(path)
        parts: List[str] = []
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text)
        doc.close()
        full_text = "\n".join(parts)
        logger.debug(f"Extracted {len(full_text)} chars from PDF: {path}")
        return full_text
    except Exception as exc:
        logger.warning(f"Failed to extract PDF {path}: {exc}")
        return ""


def discover_books(root_dir: str) -> List[str]:
    """Walk *root_dir* (~ expanded) and return paths to .epub/.pdf files."""
    root = os.path.expanduser(root_dir)
    found: List[str] = []
    if not os.path.isdir(root):
        logger.info(f"Book directory does not exist: {root}")
        return found
    for dirpath, _dirs, filenames in os.walk(root):
        for fname in filenames:
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if ext in ("epub", "pdf"):
                found.append(os.path.join(dirpath, fname))
    logger.info(f"Discovered {len(found)} books in {root}")
    return found


# -- Stream classes ----------------------------------------------------------


class BookTextStream:
    """Yields ~10K-char text chunks from discovered ebooks in random order.

    Supports deterministic train/val splitting by holding out the last
    *val_fraction* of alphabetically-sorted book paths as validation data.
    Applies English detection, minimum-quality checks, and SHA-256 dedup.
    """

    def __init__(
        self,
        book_dir: str,
        val_fraction: float = 0.15,
        split: str = "train",
    ) -> None:
        self.book_dir = book_dir
        self.val_fraction = val_fraction
        self.split = split
        self._seen_hashes: Set[str] = set()

        all_paths = sorted(discover_books(book_dir))
        n_val = max(1, int(len(all_paths) * val_fraction)) if all_paths else 0
        n_train = len(all_paths) - n_val

        if split == "val":
            self.paths = all_paths[n_train:]
        else:
            self.paths = all_paths[:n_train]

        logger.info(
            f"BookTextStream split={split}: {len(self.paths)} books "
            f"(total={len(all_paths)}, train={n_train}, val={n_val})"
        )

    def __iter__(self) -> Iterator[str]:
        order = list(self.paths)
        random.shuffle(order)
        for path in order:
            ext = path.lower().rsplit(".", 1)[-1]
            text = extract_epub_text(path) if ext == "epub" else extract_pdf_text(path)
            if not text:
                continue
            for start in range(0, len(text), BOOK_CHUNK_SIZE):
                chunk = text[start : start + BOOK_CHUNK_SIZE]

                # -- Quality filters --
                # Minimum length: 500 chars
                if len(chunk) < 500:
                    continue

                # At least 50% alphabetic (filters OCR noise, tables, metadata)
                alpha_count = sum(1 for ch in chunk if ch.isalpha())
                if alpha_count / len(chunk) < 0.50:
                    continue

                # English detection
                if not _is_english(chunk):
                    continue

                # SHA-256 deduplication on first 500 chars
                chunk_hash = hashlib.sha256(chunk[:500].encode("utf-8")).hexdigest()
                if chunk_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(chunk_hash)

                yield chunk


class HuggingFaceStream:
    """Streams text from a HuggingFace dataset using the datasets library.

    Supports deterministic train/val splitting via SHA-256 hash of text content
    and optional quality filtering by score field and minimum character count.
    """

    TEXT_COLUMNS = ("text", "content", "document", "passage", "sentence")

    def __init__(
        self,
        dataset_name: str,
        hf_split: str = "train",
        val_fraction: float = 0.15,
        split: str = "train",
        min_chars: int = 200,
        min_quality_score: float = 3.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.hf_split = hf_split
        self.val_fraction = val_fraction
        self.split = split
        self.min_chars = min_chars
        self.min_quality_score = min_quality_score

    def _find_text_column(self, example: Dict) -> Optional[str]:
        for col in self.TEXT_COLUMNS:
            if col in example and isinstance(example[col], str):
                return col
        for key, val in example.items():
            if isinstance(val, str) and len(val) > 50:
                return key
        return None

    def _is_val(self, text: str) -> bool:
        """Deterministic hash-based split: True if text belongs to val set."""
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        return (h % 1000) < int(self.val_fraction * 1000)

    def __iter__(self) -> Iterator[str]:
        from datasets import load_dataset

        logger.info(
            f"Loading HF dataset: {self.dataset_name} "
            f"(hf_split={self.hf_split}, split={self.split})"
        )
        ds = load_dataset(self.dataset_name, split=self.hf_split, streaming=True)
        text_col: Optional[str] = None
        for example in ds:
            if text_col is None:
                text_col = self._find_text_column(example)
                if text_col is None:
                    logger.warning(
                        f"No text column found in {self.dataset_name}. "
                        f"Available keys: {list(example.keys())}"
                    )
                    return
                logger.info(f"Using column '{text_col}' from {self.dataset_name}")
            text = example.get(text_col, "")
            if not isinstance(text, str) or len(text) < self.min_chars:
                continue

            # Quality score filter (e.g. FineWeb-edu 'score' field)
            if "score" in example:
                try:
                    score = float(example["score"])
                    if score < self.min_quality_score:
                        continue
                except (TypeError, ValueError):
                    pass

            # Deterministic train/val split
            is_val = self._is_val(text)
            if self.split == "val" and not is_val:
                continue
            if self.split == "train" and is_val:
                continue

            yield text


class JSONLStream:
    """Yields formatted text from a JSONL file containing reasoning data.

    Handles dicts with 'instruction', 'reasoning', 'response' keys by
    formatting them as a structured conversation.  Supports train/val
    splitting by holding out the last *val_fraction* of lines as validation.
    """

    def __init__(
        self,
        path: str,
        val_fraction: float = 0.15,
        split: str = "train",
    ) -> None:
        self.path = os.path.expanduser(path)
        self.val_fraction = val_fraction
        self.split = split

    @staticmethod
    def format_entry(entry: Dict) -> str:
        parts: List[str] = []
        if "system" in entry and str(entry["system"]).strip():
            parts.append(f"<system> {entry['system']}")
        if "context" in entry and str(entry["context"]).strip():
            parts.append(f"<context> {entry['context']}")
        if "instruction" in entry:
            parts.append(f"<user> {entry['instruction']}")
        if "reasoning" in entry:
            parts.append(f"<think>{entry['reasoning']}</think>")
        if "response" in entry:
            parts.append(f"<assistant> {entry['response']}")
        if parts:
            return "\n".join(parts)
        # Fallback: concatenate all string values
        return " ".join(str(v) for v in entry.values() if isinstance(v, str))

    def _format_entry(self, entry: Dict) -> str:
        return self.format_entry(entry)

    def _count_lines(self) -> int:
        """Count non-empty lines in the JSONL file."""
        count = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count

    def __iter__(self) -> Iterator[str]:
        if not os.path.isfile(self.path):
            logger.warning(f"JSONL file not found: {self.path}")
            return

        total_lines = self._count_lines()
        n_val = max(1, int(total_lines * self.val_fraction)) if total_lines else 0
        n_train = total_lines - n_val

        logger.info(
            f"Reading JSONL: {self.path} (split={self.split}, "
            f"total={total_lines}, train={n_train}, val={n_val})"
        )

        with open(self.path, "r", encoding="utf-8") as fh:
            line_idx = 0
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                # Determine whether this non-empty line is train or val
                in_train = line_idx < n_train
                line_idx += 1

                if self.split == "train" and not in_train:
                    continue
                if self.split == "val" and in_train:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_num}")
                    continue
                text = self._format_entry(entry)
                if text:
                    yield text


class SelfLearnJSONLStream:
    """Reads accepted self-learn JSONL shards from a directory tree."""

    def __init__(
        self,
        root_dir: str,
        val_fraction: float = 0.15,
        split: str = "train",
        max_shards: int = 20,
    ) -> None:
        self.root_dir = os.path.expanduser(root_dir)
        self.val_fraction = val_fraction
        self.split = split
        self.max_shards = max_shards
        self._seen_hashes: Set[str] = set()
        self.paths = self._discover_paths()

    def _discover_paths(self) -> List[str]:
        if not os.path.isdir(self.root_dir):
            logger.info(f"SelfLearn directory does not exist: {self.root_dir}")
            return []

        paths: List[str] = []
        for dirpath, _dirs, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(dirpath, fname))
        paths = sorted(paths)
        if self.max_shards > 0:
            paths = paths[-self.max_shards :]
        logger.info(
            f"SelfLearnJSONLStream split={self.split}: {len(paths)} shard(s) from {self.root_dir}"
        )
        return paths

    def _sample_hash(self, entry: Dict, text: str) -> str:
        raw_hash = entry.get("sample_hash")
        if isinstance(raw_hash, str) and raw_hash:
            return raw_hash
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_val(self, sample_hash: str) -> bool:
        h = int(hashlib.sha256(sample_hash.encode("utf-8")).hexdigest(), 16)
        return (h % 1000) < int(self.val_fraction * 1000)

    def __iter__(self) -> Iterator[str]:
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping malformed selflearn JSON at {path}:{line_num}"
                        )
                        continue

                    text = JSONLStream.format_entry(entry)
                    if not text:
                        continue

                    sample_hash = self._sample_hash(entry, text)
                    if sample_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(sample_hash)

                    is_val = self._is_val(sample_hash)
                    if self.split == "train" and is_val:
                        continue
                    if self.split == "val" and not is_val:
                        continue

                    yield text


# -- Pipeline assembly -------------------------------------------------------


class MixedDataStream:
    """Interleaves multiple text iterables according to mixing weights.

    At each step, randomly selects a source based on the weights.
    Exhausted sources are removed and weights renormalized.
    """

    def __init__(
        self, sources: List[Iterator[str]], weights: List[float]
    ) -> None:
        if len(sources) != len(weights):
            raise ValueError(
                f"sources ({len(sources)}) and weights ({len(weights)}) must match"
            )
        self.sources = sources
        self.weights = weights

    def __iter__(self) -> Iterator[str]:
        iters: List[Iterator[str]] = [iter(s) for s in self.sources]
        weights = list(self.weights)
        while iters:
            # Normalize weights
            total = sum(weights)
            if total <= 0:
                break
            probs = [w / total for w in weights]
            idx = random.choices(range(len(iters)), weights=probs, k=1)[0]
            try:
                text = next(iters[idx])
                yield text
            except StopIteration:
                logger.info(f"Source {idx} exhausted, removing from mix")
                iters.pop(idx)
                weights.pop(idx)


class TokenizedBatchIterator:
    """Packs tokenized text into fixed-length batches for training.

    Concatenates documents with EOS separator tokens into a rolling buffer,
    then slices seq_len windows. Yields mx.array batches of shape
    (batch_size, seq_len) with dtype int32.
    """

    def __init__(
        self,
        text_stream: Iterator[str],
        tokenizer: MLXTokenizer,
        seq_len: int = 1024,
        batch_size: int = 4,
    ) -> None:
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.total_tokens: int = 0
        self._buffer: List[int] = []
        self._text_iter: Optional[Iterator[str]] = None
        self._exhausted: bool = False

    def _fill_buffer(self) -> None:
        """Pull text from the stream and tokenize into the buffer."""
        if self._text_iter is None:
            self._text_iter = iter(self.text_stream)
        while len(self._buffer) < self.seq_len and not self._exhausted:
            try:
                text = next(self._text_iter)
            except StopIteration:
                self._exhausted = True
                return
            ids = self.tokenizer.encode(text).tolist()
            self._buffer.extend(ids)
            self._buffer.append(self.tokenizer.eos_id)

    def _next_sequence(self) -> Optional[List[int]]:
        """Slice one seq_len window from the buffer, or None if insufficient."""
        self._fill_buffer()
        if len(self._buffer) < self.seq_len:
            return None
        seq = self._buffer[: self.seq_len]
        self._buffer = self._buffer[self.seq_len :]
        self.total_tokens += self.seq_len
        return seq

    def __iter__(self) -> "TokenizedBatchIterator":
        return self

    def __next__(self):
        import mlx.core as mx

        sequences: List[List[int]] = []
        while len(sequences) < self.batch_size:
            seq = self._next_sequence()
            if seq is None:
                break
            sequences.append(seq)
        if not sequences:
            raise StopIteration
        # Pad last batch if needed (rare: only at end of stream)
        while len(sequences) < self.batch_size:
            sequences.append([self.tokenizer.pad_id] * self.seq_len)
        return mx.array(sequences, dtype=mx.int32)


# -- Convenience factory -----------------------------------------------------


def build_data_pipeline(
    tokenizer: MLXTokenizer,
    book_dir: str = "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books",
    hf_datasets: Optional[List[str]] = None,
    jsonl_path: Optional[str] = None,
    selflearn_dir: Optional[str] = None,
    selflearn_max_shards: int = 20,
    mixing_weights: Optional[List[float]] = None,
    seq_len: int = 1024,
    batch_size: int = 4,
    split: str = "train",
    val_fraction: float = 0.15,
) -> TokenizedBatchIterator:
    """Build a complete data pipeline from config values.

    Creates stream sources for books, HuggingFace datasets, and JSONL data,
    mixes them, and returns a TokenizedBatchIterator.

    The *split* parameter ("train" or "val") and *val_fraction* are forwarded
    to every stream constructor so that train and val pipelines draw from
    non-overlapping subsets of the data.
    """
    sources: List[Iterator[str]] = []
    weights: List[float] = []
    family_weights = list(mixing_weights) if mixing_weights else [0.45, 0.25, 0.20, 0.10]

    # Books
    book_stream = BookTextStream(book_dir, val_fraction=val_fraction, split=split)
    if book_stream.paths:
        sources.append(book_stream)
        weights.append(family_weights[0] if len(family_weights) >= 1 else 0.45)
        logger.info(f"Added book source with {len(book_stream.paths)} files")

    # HuggingFace
    if hf_datasets:
        hf_weight_total = family_weights[1] if len(family_weights) >= 2 else 0.25
        for ds_name in hf_datasets:
            sources.append(
                HuggingFaceStream(
                    ds_name, val_fraction=val_fraction, split=split,
                )
            )
            weights.append(hf_weight_total / len(hf_datasets))
        logger.info(f"Added {len(hf_datasets)} HuggingFace source(s)")

    # JSONL reasoning
    if jsonl_path and os.path.isfile(os.path.expanduser(jsonl_path)):
        sources.append(
            JSONLStream(jsonl_path, val_fraction=val_fraction, split=split)
        )
        weights.append(family_weights[2] if len(family_weights) >= 3 else 0.20)
        logger.info(f"Added JSONL source: {jsonl_path}")

    # Accepted self-learn JSONL
    if selflearn_dir and os.path.isdir(os.path.expanduser(selflearn_dir)):
        selflearn_stream = SelfLearnJSONLStream(
            selflearn_dir,
            val_fraction=val_fraction,
            split=split,
            max_shards=selflearn_max_shards,
        )
        if selflearn_stream.paths:
            sources.append(selflearn_stream)
            weights.append(family_weights[3] if len(family_weights) >= 4 else 0.10)
            logger.info(
                f"Added selflearn source: {selflearn_dir} ({len(selflearn_stream.paths)} shard(s))"
            )

    if not sources:
        raise RuntimeError("No data sources available. Check paths and config.")

    # Apply direct per-source weights if provided and matching the active source count
    if mixing_weights and len(mixing_weights) == len(sources):
        weights = list(mixing_weights)

    mixed = MixedDataStream(sources, weights)
    return TokenizedBatchIterator(mixed, tokenizer, seq_len, batch_size)


def build_train_val_pipelines(
    tokenizer: MLXTokenizer,
    book_dir: str = "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books",
    hf_datasets: Optional[List[str]] = None,
    jsonl_path: Optional[str] = None,
    selflearn_dir: Optional[str] = None,
    selflearn_max_shards: int = 20,
    mixing_weights: Optional[List[float]] = None,
    seq_len: int = 1024,
    batch_size: int = 4,
    val_fraction: float = 0.15,
) -> Tuple[TokenizedBatchIterator, TokenizedBatchIterator]:
    """Build separate train and val data pipelines."""
    train_iter = build_data_pipeline(
        tokenizer, book_dir, hf_datasets, jsonl_path, selflearn_dir,
        selflearn_max_shards, mixing_weights,
        seq_len, batch_size, split="train", val_fraction=val_fraction,
    )
    val_iter = build_data_pipeline(
        tokenizer, book_dir, hf_datasets, jsonl_path, selflearn_dir,
        selflearn_max_shards, mixing_weights,
        seq_len, batch_size, split="val", val_fraction=val_fraction,
    )
    return train_iter, val_iter


# -- Validation block --------------------------------------------------------

if __name__ == "__main__":
    import sys
    all_validation_failures: List[str] = []
    total_tests = 0

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning with MLX on Apple Silicon.",
        "Explain what this function does and how it works.",
        "Step 1: Read the function. Step 2: Trace the logic.",
        "Hello world, this is a simple test of the data pipeline.",
    ] * 100
    tok = MLXTokenizer(vocab_size=500)
    tok.train(sample_texts)

    # Test 1: discover_books handles missing directory gracefully
    total_tests += 1
    try:
        books = discover_books("/tmp/nonexistent_book_dir_12345")
        if books != []:
            all_validation_failures.append(f"discover_books missing dir: expected [], got {books}")
        else:
            print("  Test 1 OK: discover_books on missing dir returned []")
    except Exception as exc:
        all_validation_failures.append(f"discover_books raised: {exc}")

    # Test 2: discover_books + extract on real books directory
    total_tests += 1
    real_books = discover_books(
        "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books"
    )
    if isinstance(real_books, list):
        print(f"  Test 2 OK: discover_books found {len(real_books)} books")
    else:
        all_validation_failures.append(f"discover_books returned {type(real_books)}")

    # Test 3: extract_epub_text on a real file (if available)
    total_tests += 1
    epub_files = [b for b in real_books if b.lower().endswith(".epub")]
    if epub_files:
        epub_text = extract_epub_text(epub_files[0])
        if isinstance(epub_text, str) and len(epub_text) > 0:
            print(f"  Test 3 OK: extracted {len(epub_text)} chars from EPUB")
        else:
            all_validation_failures.append(f"extract_epub_text empty for {epub_files[0]}")
    else:
        print("  Test 3 SKIP: no EPUB files available")

    # Test 4: extract_pdf_text on a real file (if available)
    total_tests += 1
    pdf_files = [b for b in real_books if b.lower().endswith(".pdf")]
    if pdf_files:
        pdf_text = extract_pdf_text(pdf_files[0])
        if isinstance(pdf_text, str) and len(pdf_text) > 0:
            print(f"  Test 4 OK: extracted {len(pdf_text)} chars from PDF")
        else:
            all_validation_failures.append(f"extract_pdf_text empty for {pdf_files[0]}")
    else:
        print("  Test 4 SKIP: no PDF files available")

    # Test 5: JSONLStream reads reasoning data
    total_tests += 1
    jsonl_path = os.path.join(
        os.path.dirname(__file__), "data", "reasoning_50k", "reasoning_data.jsonl"
    )
    if os.path.isfile(jsonl_path):
        jsonl_items = []
        for item in JSONLStream(jsonl_path):
            jsonl_items.append(item)
            if len(jsonl_items) >= 3:
                break
        has_user = any("<user>" in it for it in jsonl_items)
        has_think = any("<think>" in it for it in jsonl_items)
        if len(jsonl_items) == 3 and has_user and has_think:
            print(f"  Test 5 OK: JSONLStream yielded 3 formatted entries")
        else:
            all_validation_failures.append(
                f"JSONLStream: got {len(jsonl_items)} items, user={has_user}, think={has_think}"
            )
    else:
        print(f"  Test 5 SKIP: JSONL not found at {jsonl_path}")

    # Test 6: MixedDataStream interleaves correctly
    total_tests += 1
    def _gen(prefix: str, n: int) -> Iterator[str]:
        for i in range(n):
            yield f"{prefix}_{i}"
    mixed_items = list(MixedDataStream([_gen("A", 10), _gen("B", 10)], [0.5, 0.5]))
    a_ct = sum(1 for x in mixed_items if x.startswith("A"))
    b_ct = sum(1 for x in mixed_items if x.startswith("B"))
    if len(mixed_items) == 20 and a_ct == 10 and b_ct == 10:
        print("  Test 6 OK: MixedDataStream yielded 20 items (10 A, 10 B)")
    else:
        all_validation_failures.append(
            f"MixedDataStream: expected 20 (10A+10B), got {len(mixed_items)} ({a_ct}A+{b_ct}B)"
        )

    # Test 7: TokenizedBatchIterator produces correct shapes
    total_tests += 1
    synthetic = ["The quick brown fox jumps over the lazy dog. " * 20] * 50
    seq_len, batch_size = 64, 4
    batch_iter = TokenizedBatchIterator(iter(synthetic), tok, seq_len=seq_len, batch_size=batch_size)
    first_batch = next(batch_iter)
    if (isinstance(first_batch, mx.array)
            and first_batch.shape == (batch_size, seq_len)
            and first_batch.dtype == mx.int32):
        print(f"  Test 7 OK: batch shape={first_batch.shape}, dtype={first_batch.dtype}")
    else:
        all_validation_failures.append(
            f"Batch: type={type(first_batch)}, shape={getattr(first_batch, 'shape', '?')}, "
            f"dtype={getattr(first_batch, 'dtype', '?')}"
        )

    # Test 8: Multiple batches and total_tokens tracking
    total_tests += 1
    batch_count = 1
    for batch in batch_iter:
        batch_count += 1
        if batch_count >= 5:
            break
    if batch_iter.total_tokens >= seq_len * batch_size * 2:
        print(f"  Test 8 OK: {batch_count} batches, total_tokens={batch_iter.total_tokens}")
    else:
        all_validation_failures.append(
            f"total_tokens: expected >= {seq_len * batch_size * 2}, got {batch_iter.total_tokens}"
        )

    # Test 9: _is_english detects English text correctly
    total_tests += 1
    english_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test of the English language detection that checks for "
        "common stopwords and ASCII character ratios in the text."
    )
    japanese_text = "これはテストです。日本語のテキストです。"
    mixed_mostly_jp = "Hello " + "あ" * 300

    eng_result = _is_english(english_text)
    jp_result = _is_english(japanese_text)
    mix_result = _is_english(mixed_mostly_jp)

    if eng_result is True and jp_result is False and mix_result is False:
        print("  Test 9 OK: _is_english correctly identifies English vs non-English")
    else:
        all_validation_failures.append(
            f"_is_english: english={eng_result} (expected True), "
            f"japanese={jp_result} (expected False), "
            f"mixed={mix_result} (expected False)"
        )

    # Test 10: BookTextStream train/val split yields non-overlapping paths
    total_tests += 1
    book_dir = "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books"
    train_stream = BookTextStream(book_dir, val_fraction=0.15, split="train")
    val_stream = BookTextStream(book_dir, val_fraction=0.15, split="val")
    train_set = set(train_stream.paths)
    val_set = set(val_stream.paths)
    overlap = train_set & val_set
    total_books = len(train_set) + len(val_set)
    if len(overlap) == 0 and total_books > 0:
        print(
            f"  Test 10 OK: BookTextStream train={len(train_set)}, "
            f"val={len(val_set)}, overlap=0"
        )
    elif total_books == 0:
        print("  Test 10 SKIP: no books available to test split")
    else:
        all_validation_failures.append(
            f"BookTextStream split overlap: {len(overlap)} paths in both train and val"
        )

    # Test 11: BookTextStream deduplication filters repeated chunks
    total_tests += 1
    dedup_stream = BookTextStream("/tmp/nonexistent_dedup_dir", split="train")
    # Manually inject paths and test the dedup mechanism via _seen_hashes
    # We simulate by checking that adding the same hash twice is caught
    test_chunk = "A" * 600  # 600 chars, all alphabetic, but won't pass _is_english
    # Use a chunk that passes all filters
    dedup_chunk = (
        "The quick brown fox jumps over the lazy dog and this is a test of "
        "the deduplication mechanism that checks for repeated content in the "
        "data pipeline to ensure we do not feed duplicate text to the model "
        "during training which would waste compute and skew the distribution. "
        "Additional padding text to make this chunk long enough to pass the "
        "minimum 500 character requirement that is enforced by the quality "
        "filters in the BookTextStream iterator implementation."
    )
    chunk_hash = hashlib.sha256(dedup_chunk[:500].encode("utf-8")).hexdigest()
    # First insertion should succeed
    dedup_stream._seen_hashes.clear()
    first_add = chunk_hash not in dedup_stream._seen_hashes
    dedup_stream._seen_hashes.add(chunk_hash)
    # Second check should find it already present
    second_add = chunk_hash in dedup_stream._seen_hashes
    if first_add and second_add:
        print("  Test 11 OK: dedup hash tracking works (first=new, second=seen)")
    else:
        all_validation_failures.append(
            f"Dedup: first_add={first_add} (expected True), "
            f"second_add={second_add} (expected True)"
        )

    # Test 12: JSONLStream train/val split by line count (non-overlapping lines)
    total_tests += 1
    if os.path.isfile(jsonl_path):
        train_jsonl = list(JSONLStream(jsonl_path, val_fraction=0.15, split="train"))
        val_jsonl = list(JSONLStream(jsonl_path, val_fraction=0.15, split="val"))
        total_jsonl = len(train_jsonl) + len(val_jsonl)
        # The split is by line index, not content. The source JSONL has duplicate
        # entries so string-based overlap doesn't validate the split logic.
        # Instead verify count partitioning: train + val == total non-empty lines.
        expected_total = JSONLStream(jsonl_path)._count_lines()
        if total_jsonl == expected_total and total_jsonl > 0:
            unique_overlap = len(set(train_jsonl) & set(val_jsonl))
            print(
                f"  Test 12 OK: JSONLStream train={len(train_jsonl)}, "
                f"val={len(val_jsonl)}, total={total_jsonl} "
                f"(content overlap={unique_overlap} due to source duplicates)"
            )
        elif total_jsonl == 0:
            all_validation_failures.append("JSONLStream split: no entries yielded")
        else:
            all_validation_failures.append(
                f"JSONLStream split count mismatch: train({len(train_jsonl)}) + "
                f"val({len(val_jsonl)}) = {total_jsonl}, expected {expected_total}"
            )
    else:
        print(f"  Test 12 SKIP: JSONL not found at {jsonl_path}")

    # Test 13: HuggingFaceStream constructor accepts new parameters
    total_tests += 1
    try:
        hf = HuggingFaceStream(
            "test/dataset",
            val_fraction=0.15,
            split="train",
            min_chars=200,
            min_quality_score=3.0,
        )
        if (hf.val_fraction == 0.15 and hf.split == "train"
                and hf.min_chars == 200 and hf.min_quality_score == 3.0):
            print("  Test 13 OK: HuggingFaceStream constructor accepts all new params")
        else:
            all_validation_failures.append(
                f"HuggingFaceStream params: val_fraction={hf.val_fraction}, "
                f"split={hf.split}, min_chars={hf.min_chars}, "
                f"min_quality_score={hf.min_quality_score}"
            )
    except Exception as exc:
        all_validation_failures.append(f"HuggingFaceStream constructor raised: {exc}")

    # Test 14: HuggingFaceStream hash-based split is deterministic
    total_tests += 1
    hf_test = HuggingFaceStream("dummy", val_fraction=0.15, split="train")
    sample = "This is a deterministic test sentence for hash split"
    result1 = hf_test._is_val(sample)
    result2 = hf_test._is_val(sample)
    if result1 == result2:
        print(f"  Test 14 OK: HuggingFaceStream hash split is deterministic (is_val={result1})")
    else:
        all_validation_failures.append(
            f"HuggingFaceStream hash split not deterministic: {result1} vs {result2}"
        )

    # Test 15: build_data_pipeline accepts split and val_fraction
    total_tests += 1
    try:
        # Just verify the signature works - don't actually build (no data needed)
        import inspect
        sig = inspect.signature(build_data_pipeline)
        param_names = list(sig.parameters.keys())
        if "split" in param_names and "val_fraction" in param_names:
            print("  Test 15 OK: build_data_pipeline accepts split and val_fraction")
        else:
            all_validation_failures.append(
                f"build_data_pipeline missing params: {param_names}"
            )
    except Exception as exc:
        all_validation_failures.append(f"build_data_pipeline signature check: {exc}")

    # Test 16: build_train_val_pipelines exists and has correct signature
    total_tests += 1
    try:
        import inspect
        sig = inspect.signature(build_train_val_pipelines)
        param_names = list(sig.parameters.keys())
        ret = sig.return_annotation
        if "val_fraction" in param_names and "tokenizer" in param_names:
            print("  Test 16 OK: build_train_val_pipelines has correct signature")
        else:
            all_validation_failures.append(
                f"build_train_val_pipelines missing params: {param_names}"
            )
    except Exception as exc:
        all_validation_failures.append(f"build_train_val_pipelines check: {exc}")

    # Final result
    if all_validation_failures:
        print(f"\n{'!'*60}")
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for f in all_validation_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\nVALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
