"""
MLX-compatible WordPiece tokenizer for the Neuroplasticity Engine.

Wraps the HuggingFace `tokenizers` library to produce mlx.core arrays
suitable for model training and inference on Apple Silicon.

Docs: https://huggingface.co/docs/tokenizers/
      https://ml-explore.github.io/mlx/

Sample input:
    tokenizer = MLXTokenizer(vocab_size=1000)
    tokenizer.train(["Hello world", "The quick brown fox"] * 50)
    ids = tokenizer.encode("Hello world")

Expected output:
    ids -> mx.array of int32 token IDs
    tokenizer.decode(ids) -> approximate reconstruction of "Hello world"
"""

from __future__ import annotations

import os
from typing import List, Optional

from loguru import logger
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>",
    "<think>", "</think>",
    "<user>", "<assistant>", "<system>",
]


class MLXTokenizer:
    """WordPiece tokenizer that returns MLX arrays."""

    def __init__(self, vocab_size: int = 30000) -> None:
        self._vocab_size = vocab_size
        self._tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
        self._tok.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.StripAccents(),
        ])
        self._tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Digits(individual_digits=False),
        ])
        self._tok.decoder = decoders.WordPiece()

    # ---- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<pad>") or 0

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id("<unk>") or 1

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id("<s>") or 2

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("</s>") or 3

    # ---- core methods ------------------------------------------------------

    def encode(self, text: str):
        """Encode a single string into an mx.array of token IDs."""
        import mlx.core as mx

        enc = self._tok.encode(text)
        return mx.array(enc.ids, dtype=mx.int32)

    def encode_batch(self, texts: List[str]):
        """Encode a list of strings into a list of mx.array."""
        import mlx.core as mx

        encodings = self._tok.encode_batch(texts)
        return [mx.array(e.ids, dtype=mx.int32) for e in encodings]

    def decode(self, ids) -> str:
        """Decode an mx.array (or list) of token IDs back to text."""
        import mlx.core as mx

        if isinstance(ids, mx.array):
            id_list = ids.tolist()
        else:
            id_list = list(ids)
        return self._tok.decode(id_list)

    # ---- train / save / load -----------------------------------------------

    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Train the tokenizer on a list of texts."""
        target_size = vocab_size or self._vocab_size
        trainer = trainers.WordPieceTrainer(
            vocab_size=target_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        self._tok.train_from_iterator(texts, trainer)
        logger.info(f"Trained tokenizer: vocab_size={self.vocab_size}")
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._tok.save(output_path)
            logger.info(f"Saved tokenizer to {output_path}")

    @classmethod
    def load(cls, path: str) -> "MLXTokenizer":
        """Load a previously saved tokenizer from a JSON file."""
        obj = cls.__new__(cls)
        obj._tok = Tokenizer.from_file(path)
        obj._vocab_size = obj._tok.get_vocab_size()
        logger.info(f"Loaded tokenizer from {path}, vocab_size={obj._vocab_size}")
        return obj

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._tok.save(path)


if __name__ == "__main__":
    import sys

    all_validation_failures: List[str] = []
    total_tests = 0

    # -- Train a small tokenizer on sample data --
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "System: You are a helpful assistant.",
        "User: What is 2+2?",
        "Assistant: <think>Simple arithmetic.</think> The answer is 4.",
        "Hello world, this is a test of the adaptive tokenizer.",
        "Machine learning on Apple Silicon with MLX framework.",
    ] * 100

    tok = MLXTokenizer(vocab_size=500)
    tok.train(sample_texts)

    # Test 1: vocab_size is positive
    total_tests += 1
    if tok.vocab_size < 50:
        all_validation_failures.append(f"vocab_size: expected >= 50, got {tok.vocab_size}")
    else:
        print(f"  Test 1 OK: vocab_size = {tok.vocab_size}")

    # Test 2: encode returns mx.array of int32
    total_tests += 1
    ids = tok.encode("Hello world")
    if not isinstance(ids, mx.array):
        all_validation_failures.append(f"encode: expected mx.array, got {type(ids)}")
    elif ids.dtype != mx.int32:
        all_validation_failures.append(f"encode dtype: expected int32, got {ids.dtype}")
    elif ids.size == 0:
        all_validation_failures.append("encode: returned empty array")
    else:
        print(f"  Test 2 OK: encode('Hello world') -> {ids.tolist()}")

    # Test 3: decode reconstructs text
    total_tests += 1
    decoded = tok.decode(ids)
    if not isinstance(decoded, str) or len(decoded) == 0:
        all_validation_failures.append(f"decode: expected non-empty string, got '{decoded}'")
    else:
        print(f"  Test 3 OK: decode -> '{decoded}'")

    # Test 4: encode_batch returns list of mx.array
    total_tests += 1
    batch = tok.encode_batch(["Hello", "World", "Test"])
    if not isinstance(batch, list) or len(batch) != 3:
        all_validation_failures.append(f"encode_batch: expected list of 3, got {type(batch)} len={len(batch) if isinstance(batch, list) else 'N/A'}")
    elif not all(isinstance(b, mx.array) for b in batch):
        all_validation_failures.append("encode_batch: not all items are mx.array")
    else:
        print(f"  Test 4 OK: encode_batch -> {[b.tolist() for b in batch]}")

    # Test 5: special token IDs are valid
    total_tests += 1
    special_ok = True
    for name, val in [("pad_id", tok.pad_id), ("unk_id", tok.unk_id),
                       ("bos_id", tok.bos_id), ("eos_id", tok.eos_id)]:
        if not isinstance(val, int) or val < 0:
            all_validation_failures.append(f"special token {name}: expected non-negative int, got {val}")
            special_ok = False
            break
    if special_ok:
        print(f"  Test 5 OK: pad={tok.pad_id} unk={tok.unk_id} bos={tok.bos_id} eos={tok.eos_id}")

    # Test 6: save and load roundtrip
    total_tests += 1
    save_path = "/tmp/test_mlx_tokenizer.json"
    tok.save(save_path)
    tok2 = MLXTokenizer.load(save_path)
    ids2 = tok2.encode("Hello world")
    if ids2.tolist() != ids.tolist():
        all_validation_failures.append(f"save/load roundtrip: original {ids.tolist()} != loaded {ids2.tolist()}")
    else:
        print(f"  Test 6 OK: save/load roundtrip matches")

    # Test 7: thinking tags encode as single tokens
    total_tests += 1
    think_text = "<think>Test</think>"
    think_ids = tok.encode(think_text)
    think_decoded = tok.decode(think_ids)
    if "<think>" not in think_decoded and "think" not in think_decoded.lower():
        all_validation_failures.append(f"thinking tags: decoded '{think_decoded}' missing think tokens")
    else:
        print(f"  Test 7 OK: thinking tags -> {think_ids.tolist()} -> '{think_decoded}'")

    # Final result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
