# Adaptive Model 1A - Instruct Edition

This project implements an adaptive transformer-based language model with instruction following capabilities and reasoning through thinking tags.

## Key Features

- **Instruction Following:** The model is trained to follow instructions in a chat-style format
- **Custom System Prompt:** Ability to provide custom system prompts to guide model behavior
- **Reasoning with Thinking Tags:** The model uses `<think>...</think>` tags for explicit chain-of-thought reasoning
- **SafeTensors Format:** The model is saved in the efficient and secure SafeTensors format
- **Adaptive Architecture:** The model uses dynamic architecture with structural plasticity

## Model Architecture

- Vocabulary Size: 50,000 tokens
- Hidden Dimension: 256
- Attention Heads: 4
- Transformer Layers: 4
- Feedforward Dimension: 1024
- Maximum Sequence Length: 64
- Parameters: ~30M

## Using the Model

### Training

To train the model, run the `main.py` script:

```bash
python main.py
```

This will:
1. Train a tokenizer
2. Load and process the training data
3. Train the model with the custom system prompt embedded
4. Save the model in both PyTorch and SafeTensors formats

### Inference

To run inference with the model, use the `inference.py` script:

```bash
# Generate text in instruct mode
python inference.py --prompt "Write a short poem about AI" --max_length 50

# Start interactive chat mode
python inference.py --chat

# Use a custom system prompt
python inference.py --chat --system_prompt "You are a helpful AI assistant that answers questions accurately."
```

### Chat Mode Commands

When in chat mode, the following special commands are available:

- `exit`, `quit`, `bye`: End the conversation
- `debug`: Toggle debug mode for detailed information
- `clear`: Clear conversation history
- `system <prompt>`: Set a custom system prompt

## Thinking Tags for Reasoning

This model implements reasoning capabilities using thinking tags similar to those used in DeepSeek models. When the model is reasoning through a problem, it will use `<think>...</think>` tags to show its chain of thought.

Example:

```
