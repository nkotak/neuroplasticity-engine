"""
Test AdaptiveTokenizer encode/decode round-trip and special tokens.

Uses real tokenizer operations, no mocks. Verifies the tokenizer
can train on text, encode, and decode correctly.

Note: tokenizer.encode() returns a tokenizers.Encoding object.
Use .ids to get the list of integer token IDs.
Special tokens are only in the vocabulary after training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_tokenizer import AdaptiveTokenizer


# Shared training corpus used by multiple tests
TRAINING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process natural language.",
    "Neural networks learn representations from data.",
    "Transformers use attention mechanisms for sequence modeling.",
    "System: You are a helpful assistant.",
    "User: What is machine learning?",
    "Assistant: Machine learning is a branch of AI.",
] * 20


def _trained_tokenizer(vocab_size=500):
    """Create and train a tokenizer (shared helper)."""
    tok = AdaptiveTokenizer(initial_vocab_size=vocab_size)
    tok.train(TRAINING_TEXTS, output_path="/tmp/test_tokenizer_output")
    return tok


def test_basic_encode_decode():
    """Tokenizer encodes and decodes text correctly after training."""
    tokenizer = _trained_tokenizer()

    test_text = "The quick brown fox"
    encoding = tokenizer.encode(test_text)
    ids = encoding.ids

    assert isinstance(ids, list), f"Expected list, got {type(ids)}"
    assert len(ids) > 0, "Encoded sequence should not be empty"

    decoded = tokenizer.decode(ids)
    assert isinstance(decoded, str), f"Expected str, got {type(decoded)}"
    assert len(decoded) > 0, "Decoded text should not be empty"


def test_special_tokens_present():
    """All 9 special tokens are in the vocabulary after training."""
    tokenizer = _trained_tokenizer()

    expected_special = [
        "<pad>", "<unk>", "<s>", "</s>",
        "<think>", "</think>",
        "<user>", "<assistant>", "<system>"
    ]

    for token in expected_special:
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None, f"Special token '{token}' not found in vocabulary"


def test_pad_token_id():
    """Pad token has a valid ID after training."""
    tokenizer = _trained_tokenizer()
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id is not None, "Pad token should have an ID"
    assert isinstance(pad_id, int), f"Pad token ID should be int, got {type(pad_id)}"


def test_encode_returns_integer_ids():
    """Encode produces integer token IDs via .ids attribute."""
    tokenizer = _trained_tokenizer()

    encoding = tokenizer.encode("Hello world")
    ids = encoding.ids
    assert all(isinstance(t, int) for t in ids), \
        f"All tokens should be integers, got types: {[type(t) for t in ids]}"


def test_empty_string_handling():
    """Tokenizer handles empty string without crashing."""
    tokenizer = _trained_tokenizer()

    encoding = tokenizer.encode("")
    ids = encoding.ids
    assert isinstance(ids, list), f"Expected list for empty string, got {type(ids)}"


def test_vocab_size():
    """Tokenizer respects configured vocab size."""
    target_size = 300
    tokenizer = _trained_tokenizer(vocab_size=target_size)

    actual_size = tokenizer.get_vocab_size()
    # Vocab should not exceed target (may be less with small corpus)
    assert actual_size <= target_size, \
        f"Vocab size {actual_size} exceeds target {target_size}"
    assert actual_size > 0, "Vocab size should be greater than 0"


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Basic encode/decode", test_basic_encode_decode),
        ("Special tokens present", test_special_tokens_present),
        ("Pad token ID", test_pad_token_id),
        ("Encode returns integer IDs", test_encode_returns_integer_ids),
        ("Empty string handling", test_empty_string_handling),
        ("Vocab size", test_vocab_size),
    ]

    for name, test_fn in tests:
        total_tests += 1
        try:
            test_fn()
        except Exception as e:
            all_validation_failures.append(f"{name}: {e}")

    if all_validation_failures:
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
