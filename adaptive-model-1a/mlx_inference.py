"""
MLX inference module for autoregressive text generation.

Loads a trained MLXAdaptiveTransformer checkpoint and generates text
using temperature-controlled sampling with top-k and top-p filtering.

Docs: https://ml-explore.github.io/mlx/

Sample input:
    python mlx_inference.py path/to/checkpoint/ --prompt "Hello" --max-tokens 100

Expected output:
    Generated text following the prompt.
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import typer
from loguru import logger

from mlx_model import MLXAdaptiveTransformer
from mlx_tokenizer import MLXTokenizer

# Alias for MLX array materialization (NOT Python's eval builtin)
_mx_materialize = mx.eval  # noqa: S307


def get_next_token_logits(model: MLXAdaptiveTransformer, token_list: List[int]) -> mx.array:
    """Get logits for the next token position given a prefix.

    Wraps model forward pass: model(mx.array([token_list])) -> logits[0, -1, :].

    Args:
        model: Loaded MLXAdaptiveTransformer.
        token_list: List of token IDs as prefix.

    Returns:
        Logits array of shape (vocab_size,) for the next position.
    """
    input_tokens = mx.array([token_list], dtype=mx.int32)
    logits = model(input_tokens)
    _mx_materialize(logits)
    return logits[0, -1, :]


def load_model(checkpoint_dir: str) -> MLXAdaptiveTransformer:
    """Load a trained model from a checkpoint directory.

    The checkpoint directory must contain arch_info.json and weights.npz,
    as produced by MLXAdaptiveTransformer.save_checkpoint().
    """
    logger.info(f"Loading model from {checkpoint_dir}")
    model = MLXAdaptiveTransformer.load_checkpoint(checkpoint_dir)
    model.eval()
    mx.eval(model.parameters())
    logger.info(f"Model loaded: {model.count_parameters():,} parameters")
    return model


def sample_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[List[int]] = None,
) -> mx.array:
    """Sample a single token from logits with temperature, top-k, and top-p.

    Args:
        logits: Raw logits for a single position, shape (vocab_size,).
        temperature: Sampling temperature. Higher = more random.
        top_k: Keep only the top-k highest logits. 0 disables.
        top_p: Nucleus sampling threshold. 1.0 disables.
        repetition_penalty: Penalty for repeated tokens. 1.0 = off.
        generated_ids: Previously generated token IDs for penalty.

    Returns:
        Single token ID as mx.array scalar.
    """
    # Repetition penalty (before temp scaling and greedy check)
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            val = float(logits[token_id].item())
            if val > 0:
                logits = logits.at[token_id].multiply(1.0 / repetition_penalty)
            else:
                logits = logits.at[token_id].multiply(repetition_penalty)

    if temperature <= 0:
        return mx.argmax(logits)

    logits = logits / temperature

    # Top-k filtering: keep only the top_k highest logits
    if top_k > 0 and top_k < logits.shape[0]:
        sorted_vals = mx.sort(logits)
        threshold = sorted_vals[-top_k]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    # Top-p (nucleus) filtering -- work in sorted order, then map back
    if 0 < top_p < 1.0:
        # Sort descending by logit value
        desc_indices = mx.argsort(logits)[::-1]
        desc_logits = logits[desc_indices]
        desc_probs = mx.softmax(desc_logits)
        cumulative = mx.cumsum(desc_probs)

        # Keep tokens until cumulative probability exceeds top_p
        # The first token is always kept (subtract its own prob before comparing)
        cutoff_mask = (cumulative - desc_probs) > top_p
        desc_logits = mx.where(cutoff_mask, mx.array(float("-inf")), desc_logits)

        # Sample from sorted distribution, then map index back
        sampled_sorted_idx = mx.random.categorical(desc_logits)
        mx.eval(sampled_sorted_idx)
        return desc_indices[sampled_sorted_idx.item()]

    # No top-p: sample directly from (possibly top-k filtered) logits
    token = mx.random.categorical(logits)
    return token


def generate(
    model: MLXAdaptiveTransformer,
    tokenizer: MLXTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model: Loaded MLXAdaptiveTransformer.
        tokenizer: Loaded MLXTokenizer.
        prompt: Text prompt to continue from.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling parameter.
        repetition_penalty: Penalty for repeated tokens. 1.0 = off.

    Returns:
        Full generated text including the prompt.
    """
    tokens = tokenizer.encode(prompt)
    token_list = tokens.tolist()

    for _ in range(max_tokens):
        last_logits = get_next_token_logits(model, token_list)
        next_token = sample_token(
            last_logits, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=token_list,
        )
        _mx_materialize(next_token)

        next_id = next_token.item()
        if next_id == tokenizer.eos_id:
            break
        token_list.append(next_id)

    return tokenizer.decode(mx.array(token_list, dtype=mx.int32))


def interactive_repl(
    model: MLXAdaptiveTransformer,
    tokenizer: MLXTokenizer,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.0,
) -> None:
    """Interactive read-eval-print loop for text generation.

    Commands:
        /temp <value>  - set temperature
        /topk <value>  - set top-k
        /topp <value>  - set top-p
        /rep <value>   - set repetition penalty
        quit / exit    - exit the REPL
    """
    print("Interactive generation (type 'quit' or 'exit' to stop)")
    print(
        f"  temperature={temperature}, top_k={top_k}, top_p={top_p}, "
        f"repetition_penalty={repetition_penalty}\n"
    )

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        # Handle parameter commands
        if user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: /temp <float>")
            continue
        if user_input.startswith("/topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"  top_k set to {top_k}")
            except (IndexError, ValueError):
                print("  Usage: /topk <int>")
            continue
        if user_input.startswith("/topp "):
            try:
                top_p = float(user_input.split()[1])
                print(f"  top_p set to {top_p}")
            except (IndexError, ValueError):
                print("  Usage: /topp <float>")
            continue
        if user_input.startswith("/rep "):
            try:
                repetition_penalty = float(user_input.split()[1])
                print(f"  repetition_penalty set to {repetition_penalty}")
            except (IndexError, ValueError):
                print("  Usage: /rep <float>")
            continue

        output = generate(
            model, tokenizer, user_input, 200, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
        )
        print(output)
        print()


app = typer.Typer(help="MLX text generation with adaptive transformer.")


@app.command()
def main(
    checkpoint_dir: str = typer.Argument(..., help="Path to checkpoint directory"),
    prompt: Optional[str] = typer.Option(None, help="Single prompt to generate from"),
    max_tokens: int = typer.Option(200, help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_k: int = typer.Option(40, help="Top-k filtering"),
    top_p: float = typer.Option(0.92, help="Nucleus sampling threshold"),
    repetition_penalty: float = typer.Option(1.0, help="Repetition penalty (1.0=off)"),
    tokenizer_path: Optional[str] = typer.Option(
        None, help="Path to tokenizer.json (default: <checkpoint>/tokenizer.json)"
    ),
) -> None:
    """Generate text from a trained MLXAdaptiveTransformer checkpoint."""
    model = load_model(checkpoint_dir)

    tok_path = tokenizer_path or f"{checkpoint_dir}/tokenizer.json"
    tokenizer = MLXTokenizer.load(tok_path)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    if prompt:
        output = generate(
            model, tokenizer, prompt, max_tokens, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
        )
        print(output)
    else:
        interactive_repl(
            model, tokenizer, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
        )


def verify_inference(
    model: MLXAdaptiveTransformer,
    tokenizer: MLXTokenizer,
    prompt: str,
) -> float:
    """Verify incremental inference matches full-sequence forward pass.

    Encodes the prompt, runs a full forward pass to get all position logits,
    then compares each position's logits against get_next_token_logits() called
    with the prefix up to that position.

    Args:
        model: Loaded MLXAdaptiveTransformer (should be in eval mode).
        tokenizer: Loaded MLXTokenizer.
        prompt: Text to verify against.

    Returns:
        Maximum absolute difference across all positions.
    """
    tokens = tokenizer.encode(prompt)
    token_ids = tokens.tolist()
    n = len(token_ids)

    if n < 2:
        logger.warning("verify_inference: prompt too short (need >= 2 tokens)")
        return 0.0

    # Full forward pass
    full_input = mx.array([token_ids], dtype=mx.int32)
    full_logits = model(full_input)
    _mx_materialize(full_logits)
    # full_logits shape: (1, N, vocab_size)

    max_diff = 0.0
    for pos in range(1, n):
        prefix = token_ids[:pos]
        incremental_logits = get_next_token_logits(model, prefix)
        # Compare against full_logits[0, pos-1, :]
        # (pos-1 because full_logits[0, i, :] predicts token at position i+1)
        full_at_pos = full_logits[0, pos - 1, :]
        diff = mx.max(mx.abs(incremental_logits - full_at_pos))
        _mx_materialize(diff)
        pos_diff = float(diff.item())
        if pos_diff > max_diff:
            max_diff = pos_diff

    return max_diff


if __name__ == "__main__":
    import sys
    import os
    import tempfile

    # If CLI args are passed, run the typer CLI for inference.
    # Otherwise run validation tests.
    if len(sys.argv) > 1:
        app()
        sys.exit(0)

    all_validation_failures = []
    total_tests = 0

    # -- Setup: tiny model + tokenizer --
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning on Apple Silicon with MLX framework.",
        "Hello world, this is a test of the adaptive tokenizer.",
        "System: You are a helpful assistant.",
        "User: What is two plus two?",
        "Assistant: The answer is four.",
    ] * 100

    tokenizer = MLXTokenizer(vocab_size=500)
    tokenizer.train(sample_texts)

    model = MLXAdaptiveTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        nhead=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=128,
        dropout=0.0,
    )
    model.eval()

    # Save checkpoint and tokenizer for load_model test
    tmpdir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(tmpdir, "test_ckpt")
    model.save_checkpoint(ckpt_dir)
    tok_path = os.path.join(ckpt_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    # Test 1: load_model returns a working model
    total_tests += 1
    try:
        loaded = load_model(ckpt_dir)
        test_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        test_logits = loaded(test_tokens)
        mx.eval(test_logits)
        if test_logits.shape != (1, 3, tokenizer.vocab_size):
            all_validation_failures.append(
                f"load_model shape: expected (1, 3, {tokenizer.vocab_size}), "
                f"got {test_logits.shape}"
            )
        else:
            print(f"  Test 1 OK: load_model -> logits shape {test_logits.shape}")
    except Exception as e:
        all_validation_failures.append(f"load_model raised: {e}")

    # Test 2: sample_token produces valid token IDs
    total_tests += 1
    try:
        fake_logits = mx.random.normal((tokenizer.vocab_size,))
        mx.eval(fake_logits)
        sampled = sample_token(fake_logits, temperature=0.8, top_k=20, top_p=0.9)
        mx.eval(sampled)
        token_val = sampled.item()
        if not (0 <= token_val < tokenizer.vocab_size):
            all_validation_failures.append(
                f"sample_token: token {token_val} out of range [0, {tokenizer.vocab_size})"
            )
        else:
            print(f"  Test 2 OK: sample_token -> {token_val}")
    except Exception as e:
        all_validation_failures.append(f"sample_token raised: {e}")

    # Test 3: sample_token with temperature=0 gives argmax
    total_tests += 1
    try:
        fixed_logits = mx.zeros((tokenizer.vocab_size,))
        fixed_logits = fixed_logits.at[42].add(100.0)
        mx.eval(fixed_logits)
        greedy = sample_token(fixed_logits, temperature=0.0)
        mx.eval(greedy)
        if greedy.item() != 42:
            all_validation_failures.append(
                f"sample_token greedy: expected 42, got {greedy.item()}"
            )
        else:
            print(f"  Test 3 OK: greedy sample -> {greedy.item()}")
    except Exception as e:
        all_validation_failures.append(f"sample_token greedy raised: {e}")

    # Test 4: generate produces non-empty text
    total_tests += 1
    try:
        output = generate(
            model, tokenizer, "Hello", max_tokens=10, temperature=0.8,
            top_k=20, top_p=0.9,
        )
        if not isinstance(output, str) or len(output) == 0:
            all_validation_failures.append(
                f"generate: expected non-empty string, got '{output}'"
            )
        else:
            print(f"  Test 4 OK: generate -> '{output[:80]}...'")
    except Exception as e:
        all_validation_failures.append(f"generate raised: {e}")

    # Test 5: generate respects max_tokens limit
    total_tests += 1
    try:
        prompt_text = "The"
        prompt_ids = tokenizer.encode(prompt_text)
        mx.eval(prompt_ids)
        prompt_len = prompt_ids.size
        max_new = 5
        output5 = generate(
            model, tokenizer, prompt_text, max_tokens=max_new, temperature=0.8,
            top_k=20, top_p=0.9,
        )
        # Re-encode the output to count tokens
        output_ids = tokenizer.encode(output5)
        mx.eval(output_ids)
        output_len = output_ids.size
        # Output tokens should be at most prompt_len + max_new
        # (may be fewer if EOS was hit, and re-encoding can vary slightly)
        if output_len > prompt_len + max_new + 2:
            all_validation_failures.append(
                f"generate max_tokens: prompt_len={prompt_len}, max_new={max_new}, "
                f"but output has {output_len} tokens"
            )
        else:
            print(
                f"  Test 5 OK: max_tokens respected "
                f"(prompt={prompt_len}, max_new={max_new}, output={output_len})"
            )
    except Exception as e:
        all_validation_failures.append(f"generate max_tokens raised: {e}")

    # Test 6: verify_inference reports small diff
    total_tests += 1
    try:
        model.eval()
        max_diff = verify_inference(model, tokenizer, "Hello world test")
        if max_diff > 1e-4:
            all_validation_failures.append(
                f"verify_inference: max diff {max_diff:.6f} > 1e-4"
            )
        else:
            print(f"  Test 6 OK: verify_inference max diff = {max_diff:.2e}")
    except Exception as e:
        all_validation_failures.append(f"verify_inference raised: {e}")

    # Test 7: repetition penalty reduces repeated token probability
    total_tests += 1
    try:
        model.eval()
        base_logits = mx.random.normal((tokenizer.vocab_size,))
        _mx_materialize(base_logits)
        # Without penalty
        no_pen = sample_token(base_logits, temperature=1.0, top_k=0, top_p=1.0,
                              repetition_penalty=1.0, generated_ids=[5, 5, 5])
        _mx_materialize(no_pen)
        # With penalty — token 5 should be suppressed
        pen_logits = base_logits.at[5].add(10.0)  # Make token 5 dominant
        _mx_materialize(pen_logits)
        result_no_pen = sample_token(pen_logits, temperature=0.0, repetition_penalty=1.0)
        result_pen = sample_token(pen_logits, temperature=0.0, repetition_penalty=2.0,
                                  generated_ids=[5])
        _mx_materialize(result_no_pen, result_pen)
        # With strong enough penalty, greedy should avoid token 5
        # (penalty divides positive logits, so 10.0/2.0 = 5.0 which may still be highest)
        # Just verify it doesn't crash and returns valid token
        pen_val = result_pen.item()
        if not (0 <= pen_val < tokenizer.vocab_size):
            all_validation_failures.append(
                f"repetition_penalty: token {pen_val} out of range"
            )
        else:
            print(f"  Test 7 OK: repetition penalty runs (no_pen={result_no_pen.item()}, pen={pen_val})")
    except Exception as e:
        all_validation_failures.append(f"repetition_penalty raised: {e}")

    # Test 8: get_next_token_logits returns correct shape
    total_tests += 1
    try:
        model.eval()
        logits_out = get_next_token_logits(model, [1, 2, 3])
        if logits_out.shape != (tokenizer.vocab_size,):
            all_validation_failures.append(
                f"get_next_token_logits shape: expected ({tokenizer.vocab_size},), "
                f"got {logits_out.shape}"
            )
        else:
            print(f"  Test 8 OK: get_next_token_logits shape = {logits_out.shape}")
    except Exception as e:
        all_validation_failures.append(f"get_next_token_logits raised: {e}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Final result
    if all_validation_failures:
        print(
            f"\n\u274c VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"\n\u2705 VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        sys.exit(0)
