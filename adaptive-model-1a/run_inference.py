"""
Inference script for the neuroplasticity engine's trained model.

Dynamically loads any checkpoint regardless of how the architecture evolved
during training (merges, splits, pruning all change tensor dimensions).
Reads the state dict to infer the actual model shape.

Usage:
    python run_inference.py runs/20260225_215555/
    python run_inference.py runs/20260225_215555/ --prompt "Write a function to sort a list"
    python run_inference.py runs/20260225_215555/ --interactive

Sample input:
    --prompt "Explain how binary search works"
Expected output:
    Generated text continuing from the prompt (quality depends on training)
"""
import os
import sys
import json
import logging
from typing import Optional, List, Dict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import AdaptiveTransformer
from adaptive_tokenizer import AdaptiveTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _set_inference_mode(model: nn.Module) -> None:
    """Switch model to inference mode (disables dropout, etc.)."""
    model.training = False
    for m in model.modules():
        m.training = False


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Dynamically infer model configuration from saved weights.

    This is the key function that makes inference work regardless of how
    the architecture evolved during training. Merges reduce d_ff, splits
    increase it, and each layer can end up with a different size.

    Args:
        state_dict: The loaded model state dict

    Returns:
        Dict with all config needed to reconstruct the model
    """
    config = {}

    # d_model and vocab_size from embedding
    config["vocab_size"] = state_dict["embedding.weight"].shape[0]
    config["d_model"] = state_dict["embedding.weight"].shape[1]

    # max_seq_len from positional encoding
    config["max_seq_len"] = state_dict["pos_encoder"].shape[1]

    # Discover layers and their evolved d_ff
    layer_indices = set()
    for key in state_dict:
        if key.startswith("layer_"):
            idx = int(key.split(".")[0].split("_")[1])
            layer_indices.add(idx)
    config["num_layers"] = len(layer_indices)

    # Per-layer d_ff (the evolved dimension after merges/splits)
    d_ff_per_layer = {}
    for i in sorted(layer_indices):
        key = f"layer_{i}.feed_forward.input_layer.weight"
        if key in state_dict:
            d_ff_per_layer[i] = state_dict[key].shape[0]
    config["d_ff_per_layer"] = d_ff_per_layer
    config["d_ff_max"] = max(d_ff_per_layer.values()) if d_ff_per_layer else 1024

    # nhead from attention projection
    for key in state_dict:
        if "self_attn.in_proj_weight" in key:
            proj_size = state_dict[key].shape[0]
            # in_proj packs Q, K, V: shape is [3 * d_model, d_model]
            # nhead = d_model / head_dim, but we can get it from num_heads attr
            # For now infer: proj_size = 3 * d_model, nhead is usually 4 or 8
            config["nhead"] = 4  # Default, overridden below if detectable
            break
    else:
        config["nhead"] = 4

    # Try to detect nhead from out_proj shape patterns
    for key in state_dict:
        if "self_attn.out_proj.weight" in key:
            # out_proj is [d_model, d_model], doesn't directly give nhead
            # Use common values: d_model=256 -> nhead=4, d_model=512 -> nhead=8
            d = config["d_model"]
            if d >= 512:
                config["nhead"] = 8
            elif d >= 256:
                config["nhead"] = 4
            else:
                config["nhead"] = max(1, d // 64)
            break

    # Gating parameter IDs (UUIDs from training)
    saved_gating_ids = []
    for key in sorted(state_dict.keys()):
        if key.startswith("gating_params."):
            saved_gating_ids.append(key.split(".")[1])
    config["saved_gating_ids"] = saved_gating_ids

    return config


def load_trained_model(
    run_dir: str,
    checkpoint: str = "best",
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a trained model from an E2E run directory.

    Dynamically handles any architecture changes from plasticity — the model
    could have any combination of d_ff sizes across layers due to merges/splits.

    Args:
        run_dir: Path to the run directory (e.g. runs/20260225_215555/)
        checkpoint: "best", "final", or "epoch_N" to pick a specific checkpoint
        device: Device to load to. Auto-detected if None.

    Returns:
        (model, tokenizer, snapshot_info)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Locate model weights
    model_path = None
    snapshot_path = None

    if checkpoint == "best":
        candidate = os.path.join(run_dir, "best_model.pt")
        if os.path.exists(candidate):
            model_path = candidate
        else:
            checkpoint = "final"

    if model_path is None and checkpoint == "final":
        candidate = os.path.join(run_dir, "final_model.pt")
        if os.path.exists(candidate):
            model_path = candidate
        else:
            # Fall back to highest epoch checkpoint
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                epochs = sorted(
                    [d for d in os.listdir(ckpt_dir) if d.startswith("epoch_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                if epochs:
                    checkpoint = epochs[-1]

    if model_path is None and checkpoint.startswith("epoch_"):
        model_path = os.path.join(run_dir, "checkpoints", checkpoint, "model.pt")
        snapshot_path = os.path.join(run_dir, "checkpoints", checkpoint, "snapshot.json")

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found in {run_dir}")

    logger.info(f"Loading weights from: {model_path}")

    # Load snapshot metadata if available
    snapshot = {}
    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path) as f:
            snapshot = json.load(f)

    # Load state dict and infer architecture dynamically
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = infer_config_from_state_dict(state_dict)

    logger.info(
        f"Inferred architecture: vocab={config['vocab_size']}, d_model={config['d_model']}, "
        f"nhead={config['nhead']}, layers={config['num_layers']}, "
        f"max_seq_len={config['max_seq_len']}"
    )
    logger.info(f"Evolved d_ff per layer: {config['d_ff_per_layer']}")

    # Load tokenizer
    tokenizer_path = os.path.join(run_dir, "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = AdaptiveTokenizer(
        initial_vocab_size=config["vocab_size"],
        max_vocab_size=100000,
        tokenizer_path=tokenizer_path,
    )
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # Create model shell with max d_ff (we'll resize layers next)
    model = AdaptiveTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff_max"],
        max_seq_len=config["max_seq_len"],
        device=torch.device("cpu"),
    )

    # Warm up all layers
    for _ in range(200):
        model.increment_warmup()

    # Resize each feedforward layer to match the evolved dimensions
    from transformer_model.adaptive_feedforward import AdaptiveFeedForward

    for layer_idx, target_d_ff in config["d_ff_per_layer"].items():
        layer = getattr(model, f"layer_{layer_idx}")
        current_d_ff = layer.feed_forward.input_layer.weight.shape[0]

        if current_d_ff != target_d_ff:
            logger.info(f"  layer_{layer_idx}: resizing d_ff {current_d_ff} -> {target_d_ff}")
            layer.feed_forward = AdaptiveFeedForward(config["d_model"], target_d_ff)

    # Remap gating parameter UUIDs (saved IDs -> current model IDs)
    current_gating_ids = list(model.gating_params.keys())
    saved_gating_ids = config["saved_gating_ids"]

    remapped_state_dict = {}
    id_map = {}
    if len(saved_gating_ids) == len(current_gating_ids):
        for saved_id, current_id in zip(saved_gating_ids, current_gating_ids):
            id_map[saved_id] = current_id

    for key, value in state_dict.items():
        if key.startswith("gating_params."):
            saved_id = key.split(".")[1]
            if saved_id in id_map:
                remapped_state_dict[f"gating_params.{id_map[saved_id]}"] = value
        else:
            remapped_state_dict[key] = value

    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (skip_connection internals, etc.)
        real_missing = [k for k in missing if not k.startswith("skip_connection")]
        if real_missing:
            logger.warning(f"Missing keys: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device)
    _set_inference_mode(model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ready: {total_params:,} parameters on {device}")

    return model, tokenizer, snapshot


def generate(
    model: nn.Module,
    tokenizer: AdaptiveTokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate text from a prompt using autoregressive decoding.

    Args:
        model: The loaded AdaptiveTransformer
        tokenizer: AdaptiveTokenizer
        prompt: User prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-K filtering
        top_p: Nucleus sampling threshold
        system_prompt: Optional system instruction

    Returns:
        Generated text string
    """
    device = next(model.parameters()).device
    max_seq_len = model.max_seq_len if hasattr(model, "max_seq_len") else 256

    # Format as instruction
    if system_prompt:
        formatted = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    else:
        formatted = f"User: {prompt}\n\nAssistant:"

    encoded = tokenizer.encode(formatted)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    # Ensure room for generation
    max_prompt_len = max(10, max_seq_len - max_new_tokens)
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]

    generated_ids: List[int] = []
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)

            if outputs.dim() < 3 or outputs.shape[1] == 0:
                break

            logits = outputs[:, -1, :] / temperature

            # Mask out-of-vocab tokens
            if logits.shape[-1] > vocab_size:
                logits[:, vocab_size:] = float("-inf")

            # Top-K filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, topk_idx, topk_vals)
                logits = mask

            # Top-P (nucleus) filtering
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits[remove_mask] = float("-inf")
            logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            if token_id >= vocab_size:
                break
            if token_id == eos_id or token_id == pad_id:
                break

            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.shape[1] >= max_seq_len:
                break

    if generated_ids:
        return tokenizer.decode(generated_ids)
    return "(no output generated)"


def run_demo_prompts(model: nn.Module, tokenizer: AdaptiveTokenizer) -> None:
    """Run demo prompts to showcase the model."""
    prompts = [
        "Write a Python function that checks if a number is prime",
        "Explain how a binary search algorithm works step by step",
        "What is the difference between a list and a tuple in Python?",
        "Write a function to reverse a linked list",
        "Explain the concept of recursion with an example",
    ]

    print("\n" + "=" * 70)
    print("  NEUROPLASTICITY ENGINE - INFERENCE DEMO")
    print("  Model architecture evolved through structural plasticity")
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'~' * 70}")
        print(f"  Prompt {i}: {prompt}")
        print(f"{'~' * 70}")

        response = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7)
        print(f"\n  Response:\n")
        for line in response.split("\n"):
            print(f"    {line}")

    print(f"\n{'=' * 70}\n")


def interactive_mode(model: nn.Module, tokenizer: AdaptiveTokenizer) -> None:
    """Interactive prompt-response loop."""
    print("\n" + "=" * 70)
    print("  NEUROPLASTICITY ENGINE - INTERACTIVE MODE")
    print("  Type your prompt and press Enter. Type 'quit' to exit.")
    print("=" * 70)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not prompt:
            continue

        response = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    import sys as _sys

    all_validation_failures: List[str] = []
    total_tests = 0

    # Parse arguments
    run_dir = None
    interactive = "--interactive" in _sys.argv or "-i" in _sys.argv
    custom_prompt = None

    for j, arg in enumerate(_sys.argv[1:], 1):
        if arg == "--prompt" and j + 1 < len(_sys.argv):
            custom_prompt = _sys.argv[j + 1]
        elif not arg.startswith("-") and run_dir is None:
            run_dir = arg

    if run_dir is None:
        # Auto-select most recent run
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            runs = sorted(
                [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            )
            if runs:
                run_dir = os.path.join(runs_dir, runs[-1])
                logger.info(f"Auto-selected latest run: {run_dir}")

    if run_dir is None:
        print("Usage: python run_inference.py <run_dir> [--prompt 'text'] [--interactive]")
        _sys.exit(1)

    # Test 1: Load model
    total_tests += 1
    try:
        model, tokenizer, snapshot = load_trained_model(run_dir)
    except Exception as e:
        all_validation_failures.append(f"Model loading failed: {e}")
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for f in all_validation_failures:
            print(f"  - {f}")
        _sys.exit(1)

    # Test 2: Generate from a simple prompt
    total_tests += 1
    test_output = generate(model, tokenizer, "What is 2 + 2?", max_new_tokens=50, temperature=0.5)
    if not test_output or test_output == "(no output generated)":
        all_validation_failures.append("Generation produced no output for simple prompt")
    else:
        logger.info(f"Test generation ({len(test_output)} chars): {test_output[:100]}...")

    # Test 3: Output is non-empty for second prompt
    total_tests += 1
    test_output2 = generate(model, tokenizer, "Hello", max_new_tokens=20, temperature=0.5)
    if not test_output2 or len(test_output2.strip()) == 0:
        all_validation_failures.append("Generation produced empty output for 'Hello'")

    if all_validation_failures:
        print(f"\nVALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for f in all_validation_failures:
            print(f"  - {f}")
        _sys.exit(1)
    else:
        print(f"\nVALIDATION PASSED - All {total_tests} tests produced expected results")

    # Run the requested mode
    if interactive:
        interactive_mode(model, tokenizer)
    elif custom_prompt:
        response = generate(model, tokenizer, custom_prompt, max_new_tokens=150, temperature=0.7)
        print(f"\nPrompt: {custom_prompt}")
        print(f"Response: {response}")
    else:
        run_demo_prompts(model, tokenizer)
