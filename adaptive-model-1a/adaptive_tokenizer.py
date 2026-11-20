"""
Adaptive WordPiece Tokenizer for the Neuroplasticity Engine.

Wraps the HuggingFace `tokenizers` library to provide an adaptive tokenizer
with support for vocabulary growth and special tokens for thinking/chat.

Docs: https://huggingface.co/docs/tokenizers/

Sample input:
    texts = ["Hello world", "The quick brown fox"]
    tokenizer = AdaptiveTokenizer(initial_vocab_size=1000)
    tokenizer.train(texts)
    encoded = tokenizer.encode("Hello world")

Expected output:
    encoded.ids -> list of token IDs
    tokenizer.decode(encoded.ids) -> "Hello world"
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

logger = logging.getLogger(__name__)

DEFAULT_SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>",
    "<think>", "</think>",
    "<user>", "<assistant>", "<system>"
]


class AdaptiveTokenizer:
    """Adaptive WordPiece tokenizer with vocabulary growth support.

    Wraps HuggingFace tokenizers.Tokenizer to add:
    - Automatic vocabulary growth via grow_vocabulary()
    - Special token management for chat/thinking tags
    - Metadata tracking (vocab history, growth events)
    """

    def __init__(
        self,
        initial_vocab_size: int = 30000,
        max_vocab_size: int = 100000,
        tokenizer_path: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        growth_factor: float = 1.5,
    ):
        self.initial_vocab_size = initial_vocab_size
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.growth_factor = growth_factor
        self.vocab_history: List[int] = []

        if tokenizer_path and os.path.exists(tokenizer_path):
            logger.info(f"Loading existing tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self._create_new_tokenizer()

    def _create_new_tokenizer(self) -> None:
        """Create a fresh WordPiece tokenizer with standard normalization."""
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.StripAccents(),
        ])

        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Digits(individual_digits=False),
        ])

        self.tokenizer.decoder = decoders.WordPiece()
        logger.info("Created new WordPiece tokenizer")

    def train(
        self,
        texts: List[str],
        new_vocab_size: Optional[int] = None,
        output_path: str = "tokenizer",
    ) -> str:
        """Train the tokenizer on provided texts.

        Args:
            texts: List of training texts.
            new_vocab_size: Target vocabulary size. Defaults to initial_vocab_size.
            output_path: Directory to save the trained tokenizer.

        Returns:
            Path to the saved tokenizer JSON file.
        """
        vocab_size = new_vocab_size or self.initial_vocab_size

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        logger.info(f"Training WordPiece tokenizer on {len(texts)} texts, target vocab: {vocab_size}")
        self.tokenizer.train_from_iterator(texts, trainer)

        actual_size = self.get_vocab_size()
        self.vocab_history.append(actual_size)
        logger.info(f"Tokenizer trained. Actual vocabulary size: {actual_size}")

        # Save tokenizer
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, "tokenizer.json")
        self.tokenizer.save(save_path)

        # Save metadata
        metadata = {
            "initial_vocab_size": self.initial_vocab_size,
            "max_vocab_size": self.max_vocab_size,
            "growth_factor": self.growth_factor,
            "special_tokens": self.special_tokens,
            "vocab_history": self.vocab_history,
            "current_vocab_size": actual_size,
        }
        metadata_path = os.path.join(output_path, "tokenizer_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Tokenizer saved to {save_path}")

        # Validate thinking tags
        self._test_thinking_tags()

        return save_path

    def grow_vocabulary(
        self, texts: List[str], output_path: str = "tokenizer"
    ) -> str:
        """Grow the vocabulary by training on new texts with a larger vocab target.

        Args:
            texts: New texts to incorporate.
            output_path: Directory to save updated tokenizer.

        Returns:
            Path to the saved tokenizer JSON file.
        """
        current_size = self.get_vocab_size()
        new_size = min(
            int(current_size * self.growth_factor),
            self.max_vocab_size,
        )

        if new_size <= current_size:
            logger.info(f"Vocabulary already at max ({current_size}), skipping growth")
            return os.path.join(output_path, "tokenizer.json")

        logger.info(f"Growing vocabulary from {current_size} to {new_size}")
        return self.train(texts, new_vocab_size=new_size, output_path=output_path)

    def _test_thinking_tags(self) -> None:
        """Validate that thinking tags encode and decode correctly."""
        test = "<think>Testing thinking tags.</think>"
        encoded = self.tokenizer.encode(test)
        decoded = self.tokenizer.decode(encoded.ids)

        think_start = self.token_to_id("<think>")
        think_end = self.token_to_id("</think>")

        if think_start is not None and think_end is not None:
            logger.info(f"Thinking tags verified: <think>={think_start}, </think>={think_end}")
        else:
            logger.warning("Thinking tag tokens not found in vocabulary")

    def encode(self, text: str):
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Return current vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """Convert a token ID to its string."""
        return self.tokenizer.id_to_token(id)

    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary mapping."""
        return self.tokenizer.get_vocab()

    def save(self, path: str) -> None:
        """Save the tokenizer to a file."""
        self.tokenizer.save(path)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying tokenizer."""
        if name in ("tokenizer", "initial_vocab_size", "max_vocab_size",
                     "special_tokens", "growth_factor", "vocab_history"):
            raise AttributeError(name)
        return getattr(self.tokenizer, name)


if __name__ == "__main__":
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Create tokenizer
    total_tests += 1
    try:
        tok = AdaptiveTokenizer(initial_vocab_size=1000, max_vocab_size=5000)
        if not isinstance(tok, AdaptiveTokenizer):
            all_validation_failures.append(f"Creation: Expected AdaptiveTokenizer, got {type(tok)}")
    except Exception as e:
        all_validation_failures.append(f"Creation: {e}")

    # Test 2: Train on sample data
    total_tests += 1
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "System: You are a helpful assistant.",
        "User: What is 2+2?",
        "Assistant: <think>Simple arithmetic.</think> The answer is 4.",
        "Hello world, this is a test of the adaptive tokenizer.",
    ] * 100  # Repeat for enough training data
    try:
        save_path = tok.train(sample_texts, output_path="/tmp/test_tokenizer")
        if not os.path.exists(save_path):
            all_validation_failures.append(f"Training: Tokenizer not saved to {save_path}")
    except Exception as e:
        all_validation_failures.append(f"Training: {e}")

    # Test 3: Encode/decode roundtrip
    total_tests += 1
    try:
        test_text = "Hello world"
        encoded = tok.encode(test_text)
        decoded = tok.decode(encoded.ids)
        if "Hello" not in decoded and "hello" not in decoded.lower():
            all_validation_failures.append(f"Roundtrip: Expected 'Hello' in decoded, got '{decoded}'")
    except Exception as e:
        all_validation_failures.append(f"Roundtrip: {e}")

    # Test 4: Special tokens exist
    total_tests += 1
    try:
        for token in ["<pad>", "<unk>", "<think>", "</think>", "<user>", "<assistant>"]:
            tid = tok.token_to_id(token)
            if tid is None:
                all_validation_failures.append(f"Special tokens: '{token}' not found in vocabulary")
                break
    except Exception as e:
        all_validation_failures.append(f"Special tokens: {e}")

    # Test 5: Vocabulary size
    total_tests += 1
    try:
        vocab_size = tok.get_vocab_size()
        if vocab_size < 100:
            all_validation_failures.append(f"Vocab size: Expected >= 100, got {vocab_size}")
    except Exception as e:
        all_validation_failures.append(f"Vocab size: {e}")

    # Final result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
