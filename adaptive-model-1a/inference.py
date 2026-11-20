import torch
import sys
import os
import logging
import argparse
from tokenizers import Tokenizer, models, decoders, pre_tokenizers
import re
import safetensors.torch as safetensors
import json

# Import our custom adaptive tokenizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from adaptive_tokenizer import AdaptiveTokenizer
except ImportError:
    # Fallback if module not found
    logging.warning("AdaptiveTokenizer module not found, using regular tokenizer")
    AdaptiveTokenizer = None

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import your model
from transformer_model import AdaptiveTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default system prompt that can be overridden
DEFAULT_SYSTEM_PROMPT = """You are an expert assistant trained to provide helpful, accurate, and coherent responses. When responding to users, think step-by-step, use precise language, and ensure your responses are well-structured and logically sound. Your objective is to provide the most useful and well-reasoned answers possible, drawing from your knowledge to help users understand complex topics."""

# Simple system prompt for small models
SIMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

def load_tokenizer(tokenizer_path=None, vocab_size=50000):
    """
    Load a pre-trained tokenizer from file, using AdaptiveTokenizer if available
    """
    # First, check the tokenizer directory if no specific path is provided
    if tokenizer_path is None:
        # Check default locations
        default_paths = [
            os.path.join("tokenizer", "tokenizer.json"),  # Default directory from training
            "tokenizer.json"  # Root directory
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                logging.info(f"Found tokenizer at {path}")
                tokenizer_path = path
                break
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        # Try to use the AdaptiveTokenizer if available
        if AdaptiveTokenizer is not None:
            try:
                logging.info(f"Loading tokenizer with AdaptiveTokenizer from {tokenizer_path}")
                tokenizer = AdaptiveTokenizer(
                    initial_vocab_size=vocab_size,
                    max_vocab_size=100000,
                    tokenizer_path=tokenizer_path
                )
                logging.info(f"Loaded adaptive tokenizer with vocabulary size: {tokenizer.get_vocab_size()}")
                
                # Verify special tokens
                special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<think>", "</think>"]
                for token in special_tokens:
                    token_id = tokenizer.token_to_id(token)
                    if token_id is not None:
                        logging.info(f"Special token '{token}' has ID: {token_id}")
                    else:
                        logging.warning(f"Special token '{token}' not found in tokenizer.")
                
                return tokenizer
            except Exception as e:
                logging.warning(f"Failed to load with AdaptiveTokenizer: {e}. Falling back to standard tokenizer.")
        
        # Fallback to standard tokenizer
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logging.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer.get_vocab())}")
        
        # Verify special tokens
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<think>", "</think>"]
        for token in special_tokens:
            token_id = tokenizer.token_to_id(token)
            if token_id is not None:
                logging.info(f"Special token '{token}' has ID: {token_id}")
            else:
                logging.warning(f"Special token '{token}' not found in tokenizer.")
        
        return tokenizer
    else:
        # No tokenizer found - provide clear error
        logging.error("Tokenizer not found. Run main.py first to train and save the tokenizer.")
        logging.error("Expected locations checked:")
        for path in default_paths:
            logging.error(f"  - {os.path.abspath(path)}")
        raise FileNotFoundError(f"Tokenizer file not found. Please run main.py first to train the model and save the tokenizer.")

def load_config():
    """Load model configuration if available"""
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading config.json: {e}")
    
    # Return default config values
    return {
        "model_type": "adaptive_transformer_instruct",
        "vocab_size": 50000,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 4,
        "d_ff": 1024,
        "max_seq_len": 64,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "supports_thinking_tags": True
    }

def format_thinking_output(text):
    """Format the output to properly display thinking tags"""
    # Use regex to find thinking tag content
    thinking_pattern = r'<think>(.*?)</think>'
    
    # Replace thinking tags with a nicely formatted version
    formatted = re.sub(
        thinking_pattern,
        r'\n[THINKING]\n\1\n[/THINKING]\n',
        text,
        flags=re.DOTALL
    )
    
    return formatted

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7,
                   system_prompt=None, instruct_mode=True, verbose=False, 
                   modality_inputs=None, domain_id=0):
    """
    Generate text from a given prompt with instruction formatting and thinking tag support
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for encoding/decoding text
        prompt: Text prompt to generate from
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        system_prompt: Optional custom system prompt to use
        instruct_mode: Whether to use instruction formatting
        verbose: Whether to log details about each step
        modality_inputs: Optional dictionary of multi-modal inputs (e.g., {'image': image_tensor})
        domain_id: Domain ID for multi-modal models
    """
    # Format prompt with instruction formatting if enabled
    if instruct_mode:
        if system_prompt:
            formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
    else:
        formatted_prompt = prompt
    
    # Handle multi-modal inputs
    has_modality_inputs = modality_inputs is not None and hasattr(model, 'is_multimodal') and model.is_multimodal
    
    if not has_modality_inputs:
        # Standard text-only mode
        # Encode the prompt
        encoded = tokenizer.encode(formatted_prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)  # Force long integer type
        
        # Add debugging information if verbose
        if verbose:
            logging.info(f"Formatted prompt: {formatted_prompt}")
            logging.info(f"Input shape: {input_ids.shape}")
            logging.info(f"First few tokens: {input_ids[0][:10].tolist()}")
            logging.info(f"Total prompt tokens: {len(encoded.ids)}")
        
        # Make sure input isn't empty
        if input_ids.shape[1] == 0:
            logging.error("Input sequence is empty. Tokenizer may not be working correctly.")
            return "Error: Unable to tokenize input"
        
        # Make sure input isn't too long for the model
        max_seq_len = model.max_seq_len if hasattr(model, 'max_seq_len') else 64
        
        # Ensure we have a reasonable max_length based on model's capacity
        # If max_seq_len is 64 and we need at least 10 tokens for input,
        # then max_length should be at most 54
        adjusted_max_length = min(max_length, max_seq_len - 10)
        
        # Fix bug where max_input_length could be negative
        max_input_length = max(10, max_seq_len - adjusted_max_length)  # At least 10 tokens for input
        
        if input_ids.shape[1] > max_input_length:
            # For chat, we want to keep the most recent parts of the conversation
            if not instruct_mode:
                # Keep system prompt (if any) and the last user message
                system_part = ""
                if system_prompt:
                    system_encoded = tokenizer.encode(f"System: {system_prompt}\n\n").ids
                    system_part_len = len(system_encoded)
                else:
                    system_part_len = 0
                    
                # Keep as much of the prompt as possible, prioritizing the end
                remaining_length = max_input_length - system_part_len
                if remaining_length > 0:
                    # Take tokens from the end of the prompt
                    input_ids = torch.cat([
                        input_ids[:, :system_part_len],  # System part
                        input_ids[:, -remaining_length:]  # Last part of the prompt
                    ], dim=1)
                else:
                    # Not enough room even for the last part
                    input_ids = input_ids[:, -max_input_length:]
            else:
                # Simple truncation for non-chat mode
                logging.warning(f"Input is too long ({input_ids.shape[1]} tokens). Truncating to {max_input_length} tokens.")
                input_ids = input_ids[:, -max_input_length:]  # Keep the last tokens which include assistant prompt
    else:
        # Multi-modal mode
        if 'text' not in modality_inputs and prompt:
            # Create text modality from prompt if not provided
            encoded = tokenizer.encode(formatted_prompt)
            modality_inputs['text'] = torch.tensor([encoded.ids], dtype=torch.long)
        
        if verbose:
            logging.info(f"Multi-modal input with modalities: {list(modality_inputs.keys())}")
            if 'text' in modality_inputs:
                logging.info(f"Text input shape: {modality_inputs['text'].shape}")
        
        # Set input_ids to None for multi-modal mode
        input_ids = None
    
    # Generate text auto-regressively
    device = next(model.parameters()).device
    
    # Move inputs to the correct device
    if input_ids is not None:
        input_ids = input_ids.to(device)
    
    if modality_inputs:
        # Move all modality inputs to the correct device
        for modality, tensor in modality_inputs.items():
            modality_inputs[modality] = tensor.to(device)
    
    # Store generated tokens
    generated_text = []
    
    with torch.no_grad():
        for i in range(max_length):
            try:
                # Forward pass with multi-modal support
                if has_modality_inputs:
                    outputs = model(None, modality_inputs, domain_id)
                else:
                    outputs = model(input_ids)
                
                # Log debugging info
                if verbose and i % 10 == 0:
                    if input_ids is not None:
                        logging.info(f"Generation step {i}, input shape: {input_ids.shape}")
                    else:
                        logging.info(f"Generation step {i}, using multi-modal inputs")
                    if i == 0:
                        logging.info(f"First output shape: {outputs.shape}")
                
                # Check if output tensor is valid
                if outputs.dim() < 3 or outputs.shape[1] == 0:
                    logging.error(f"Unexpected output shape: {outputs.shape}")
                    break
                
                # Get next token prediction (last token's logits)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Optional: Apply top-k filtering
                top_k = 40
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = next_token_logits.clone()
                next_token_logits[0].fill_(float('-inf'))
                next_token_logits[0, top_k_indices[0]] = top_k_logits[0]
                
                # Ensure we only sample from valid vocabulary
                vocab_size = tokenizer.get_vocab_size()
                
                # Zero out probabilities for tokens outside vocab range
                next_token_logits[0, vocab_size:] = float('-inf')
                
                # Apply top-p (nucleus) sampling for better coherence
                sorted_logits, sorted_indices = torch.sort(next_token_logits[0], descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold (0.92)
                sorted_indices_to_remove = cumulative_probs > 0.92
                # Shift indices to the right to keep also the first token above threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                # Only keep high quality tokens
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                
                # Double-check token is in vocabulary range
                if token_id < vocab_size:
                    generated_text.append(token_id)
                    
                    if verbose and i < 10:  # Log first few tokens
                        token_str = tokenizer.decode([token_id])
                        logging.info(f"Generated token {i}: {token_id} -> '{token_str}'")
                else:
                    # If we get an invalid token, log warning and use <unk> instead
                    if verbose:
                        logging.warning(f"Generated invalid token ID: {token_id}, outside vocab size {vocab_size}")
                    # Use <unk> token instead
                    unk_token_id = tokenizer.token_to_id("<unk>")
                    if unk_token_id is not None:
                        generated_text.append(unk_token_id)
                    # Skip adding to context if we can't even get a valid UNK token
                    continue
                
                # Append to input_ids - handle both multi-modal and text-only cases
                if has_modality_inputs:
                    if 'text' in modality_inputs:
                        # Update text modality with the new token
                        modality_inputs['text'] = torch.cat([modality_inputs['text'], next_token], dim=1)
                        
                        # Check for max sequence length
                        if modality_inputs['text'].shape[1] >= max_seq_len:
                            if verbose:
                                logging.info(f"Reached maximum sequence length of {max_seq_len}. Stopping generation.")
                            break
                    else:
                        # If no text modality, create one with just the generated tokens
                        modality_inputs['text'] = next_token.unsqueeze(0)
                else:
                    # Standard text-only mode
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Check for max sequence length to avoid dimension errors
                    if input_ids.shape[1] >= max_seq_len:
                        if verbose:
                            logging.info(f"Reached maximum sequence length of {max_seq_len}. Stopping generation.")
                        break
                
                # Stop if we generate an end of sequence token
                if token_id == tokenizer.token_to_id("</s>"):
                    if verbose:
                        logging.info("End of sequence token generated")
                    break
                    
            except Exception as e:
                logging.error(f"Error during generation at step {i}: {e}")
                if verbose:
                    import traceback
                    logging.error(traceback.format_exc())
                break
    
    # Attempt to decode
    try:
        # Use only the newly generated tokens
        if generated_text:
            decoded_text = tokenizer.decode(generated_text)

            # Add special handling for thinking tags to ensure they're properly preserved
            if "<think>" not in decoded_text and any(
                token_id == tokenizer.token_to_id("<think>") for token_id in generated_text
            ):
                # Reconstruct thinking tags if they're in the token IDs but not in the decoded text
                think_start_id = tokenizer.token_to_id("<think>")
                think_end_id = tokenizer.token_to_id("</think>")

                if think_start_id is not None and think_end_id is not None:
                    # Find indices of thinking tag tokens
                    indices = []
                    for i, token_id in enumerate(generated_text):
                        if token_id == think_start_id:
                            indices.append(("start", i))
                        elif token_id == think_end_id:
                            indices.append(("end", i))

                    # Reconstruct text with thinking tags
                    if indices and len(indices) >= 2:
                        parts = []
                        last_end = 0
                        for i in range(0, len(indices), 2):
                            if i+1 < len(indices):
                                start_type, start_idx = indices[i]
                                end_type, end_idx = indices[i+1]
                                if start_type == "start" and end_type == "end":
                                    # Add text before thinking block
                                    if last_end < start_idx:
                                        parts.append(tokenizer.decode(generated_text[last_end:start_idx]))

                                    # Add thinking block with proper tags
                                    think_text = tokenizer.decode(generated_text[start_idx+1:end_idx])
                                    parts.append(f"<think>{think_text}</think>")

                                    last_end = end_idx + 1

                        # Add any remaining text
                        if last_end < len(generated_text):
                            parts.append(tokenizer.decode(generated_text[last_end:]))

                        decoded_text = "".join(parts)
                        logging.info("Reconstructed thinking tags from token IDs")

            if verbose:
                logging.info(f"Raw generated text: '{decoded_text}'")

            # Format the output based on the mode
            if instruct_mode:
                result = f"{formatted_prompt} {decoded_text}"
                # Extract just the Assistant response
                match = re.search(r'Assistant:(.*)$', result, re.DOTALL)
                if match:
                    final_output = match.group(1).strip()
                    # Format thinking tags if present
                    final_output = format_thinking_output(final_output)
                    return final_output
                else:
                    # Fallback: return just the generated text
                    return decoded_text
            else:
                # For chat mode, just return the generated text
                return decoded_text
        else:
            # If we couldn't generate anything, provide a helpful message
            logging.warning("No tokens were generated")
            return "I'm having trouble generating a response. Please try again with a different question."
    except Exception as e:
        logging.error(f"Error decoding generated text: {e}")
        if verbose:
            import traceback
            logging.error(traceback.format_exc())
        return "I encountered an error while generating a response. Please try again."

def chat_with_model(model, tokenizer, system_prompt=None, context_size=2, 
                    max_length=100, debug_mode=False):
    """
    Interactive chat function for conversing with the model
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for encoding/decoding text
        system_prompt: Custom system prompt to use
        context_size: How many previous exchanges to keep as context (reduced from 3 to 2 to allow more room)
        max_length: Maximum length of generated responses (increased from 30 to 100)
        debug_mode: Enable debugging information
    """
    conversation = []
    max_seq_len = model.max_seq_len if hasattr(model, 'max_seq_len') else 64
    
    # Initialize with system prompt if provided
    if system_prompt:
        conversation.append(f"System: {system_prompt}")
    
    print("\n" + "="*50)
    print("Chat with Adaptive Instruct Model")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Type 'debug' to toggle debugging mode")
    print("Type 'clear' to clear conversation history")
    print("Type 'system <prompt>' to set a custom system prompt")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nEnding chat. Goodbye!")
            break
            
        # Check if user wants to toggle debug mode
        if user_input.lower() == 'debug':
            debug_mode = not debug_mode
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            continue
            
        # Check if user wants to clear history
        if user_input.lower() == 'clear':
            conversation = []
            if system_prompt:
                conversation.append(f"System: {system_prompt}")
            print("Conversation history cleared.")
            continue
            
        # Check if user wants to set a custom system prompt
        if user_input.lower().startswith('system '):
            system_prompt = user_input[7:].strip()
            conversation = [f"System: {system_prompt}"]
            print(f"System prompt updated to: {system_prompt}")
            continue
        
        # Add user message to conversation history
        conversation.append(f"User: {user_input}")
        
        try:
            # Create formatted prompt from conversation history
            context = ""
            
            # Always include system prompt if available
            system_idx = -1
            for i, msg in enumerate(conversation):
                if msg.startswith("System:"):
                    system_idx = i
                    break
                    
            if system_idx >= 0:
                context += conversation[system_idx] + "\n\n"
                
            # Add recent conversation context (excluding system prompt)
            recent_msgs = [msg for i, msg in enumerate(conversation) 
                          if i != system_idx and i >= len(conversation) - context_size]
            context += "\n".join(recent_msgs)
            
            # Add assistant prefix for the model to continue
            context += "\n\nAssistant:"
            
            # Check if context is already too long
            encoded_context = tokenizer.encode(context)
            if len(encoded_context.ids) > (max_seq_len - max_length):
                # Context is too long, reduce it
                if debug_mode:
                    print(f"[DEBUG] Context too long ({len(encoded_context.ids)} tokens). Reducing context...")
                
                # Keep system prompt but reduce conversation history
                if system_idx >= 0:
                    reduced_context = conversation[system_idx] + "\n\n"
                    # Just keep the most recent user message
                    reduced_context += conversation[-1] + "\n\nAssistant:"
                else:
                    reduced_context = conversation[-1] + "\n\nAssistant:"
                    
                context = reduced_context
                
                if debug_mode:
                    print(f"[DEBUG] Reduced context: '{context}'")
                    
                # Double-check the new context length
                encoded_context = tokenizer.encode(context)
                if len(encoded_context.ids) > (max_seq_len - max_length):
                    if debug_mode:
                        print(f"[DEBUG] Context still too long after reduction ({len(encoded_context.ids)} tokens).")
                    # Strip the system prompt if needed
                    context = conversation[-1] + "\n\nAssistant:"
            
            # Generate response with debugging info and improved settings
            response = generate_text(
                model,
                tokenizer,
                context,
                max_length=max_length, 
                temperature=0.6,  # Lower temperature for more coherent responses
                instruct_mode=False,  # Already formatted the prompt
                verbose=debug_mode
            )
            
            # Extract just the generated part after the context
            if response.startswith(context):
                ai_response = response[len(context):].strip()
            else:
                ai_response = response
            
            # Clean up response if needed (remove any "User:" tags that might have been generated)
            if "User:" in ai_response:
                ai_response = ai_response.split("User:")[0].strip()
                
            # Add AI response to conversation
            conversation.append(f"AI: {ai_response}")
            
            if debug_mode:
                print(f"[DEBUG] Raw response: {response}")
                print(f"[DEBUG] Processed response: {ai_response}")
            
            # Format the output for thinking tags
            display_response = format_thinking_output(ai_response)
            
            # Display AI response with reasoning highlighted if present
            if "<think>" in display_response and "</think>" in display_response:
                # Format reasoning in a highlighted way
                reasoning = display_response.split("<think>")[1].split("</think>")[0]
                response_parts = display_response.split("</think>")
                final_response = response_parts[-1] if len(response_parts) > 1 else display_response
                
                # Print with formatting
                print(f"AI: [Reasoning]: {reasoning}")
                print(f"AI: {final_response.strip()}")
            else:
                # Normal display
                print(f"AI: {display_response}")
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            print("AI: Sorry, I had trouble generating a response. Let's try again.")
            if debug_mode:
                import traceback
                print(f"[DEBUG] Error: {traceback.format_exc()}")

def main():
    # Parse command line arguments with better defaults for small models
    parser = argparse.ArgumentParser(description='Run the adaptive language model')
    parser.add_argument('--chat', action='store_true', help='Start in chat mode')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Initial prompt for text generation')
    parser.add_argument('--max_length', type=int, default=30, help='Maximum length of generated text (default 30)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation (higher = more random)')
    parser.add_argument('--system_prompt', type=str, help='Custom system prompt to use')
    parser.add_argument('--instruct', action='store_true', default=True, help='Use instruction format')
    parser.add_argument('--model_path', type=str, default='best_model.safetensors', help='Path to model weights')
    parser.add_argument('--simple', action='store_true', help='Use simple system prompt for small models')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load config
    config = load_config()
    
    # Get model parameters from config
    vocab_size = config.get("vocab_size", 50000)
    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 4)
    num_layers = config.get("num_layers", 4)
    d_ff = config.get("d_ff", 1024)
    max_seq_len = config.get("max_seq_len", 64)
    default_system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    
    # Use provided system prompt, simple prompt, or default from config
    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.simple:
        system_prompt = SIMPLE_SYSTEM_PROMPT
        logging.info("Using simple system prompt for small model")
    else:
        system_prompt = default_system_prompt
    
    print(f"Loading model...")
    
    # Initialize model (with multi-modal support if config specifies it)
    modality_config = config.get("modality_config", None)
    model = AdaptiveTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        device=device,
        modality_config=modality_config
    )
    
    # Try loading from the specified model path or fallbacks
    try:
        model_path = args.model_path
        
        # Check if model path exists, otherwise look for standard options
        if not os.path.exists(model_path):
            logging.warning(f"Specified model path {model_path} not found, searching for alternatives")
            
            # Try standard model file names
            model_paths = [
                'best_model.safetensors',
                'final_model.safetensors',
                'best_model.pt',
                'final_model.pt'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    logging.info(f"Found model at {model_path}")
                    break
        
        # Load model based on file extension
        if model_path.endswith('.safetensors'):
            logging.info(f"Loading model from safetensors format: {model_path}")
            
            # Load state dict from safetensors
            state_dict = safetensors.load_file(model_path)
            
            # Get current model layer IDs
            current_layer_ids = list(model.gating_params.keys())
            
            # Extract saved model layer IDs (from the gating_params)
            saved_layer_ids = [key.split('.')[1] for key in state_dict.keys()
                              if key.startswith('gating_params.')]
            
            # Create a mapping from saved IDs to current IDs based on position
            id_mapping = {}
            if len(saved_layer_ids) == len(current_layer_ids):
                for i, saved_id in enumerate(saved_layer_ids):
                    id_mapping[f'gating_params.{saved_id}'] = f'gating_params.{current_layer_ids[i]}'
            
            # Create a new state dict with mapped keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('gating_params.'):
                    if key in id_mapping:
                        new_state_dict[id_mapping[key]] = value
                else:
                    new_state_dict[key] = value
            
            # Load the modified state dict
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded successfully from safetensors: {model_path}!")
        # Load from PyTorch format
        else:
            # Load the saved state dict
            logging.info(f"Loading model from PyTorch format: {model_path}")
            saved_state = torch.load(model_path, map_location=device)
            
            # Get current model layer IDs
            current_layer_ids = list(model.gating_params.keys())
            
            # Extract saved model layer IDs (from the gating_params)
            saved_layer_ids = [key.split('.')[1] for key in saved_state.keys()
                             if key.startswith('gating_params.')]
            
            # Create a mapping from saved IDs to current IDs based on position
            id_mapping = {}
            if len(saved_layer_ids) == len(current_layer_ids):
                for i, saved_id in enumerate(saved_layer_ids):
                    id_mapping[f'gating_params.{saved_id}'] = f'gating_params.{current_layer_ids[i]}'
            
            # Create a new state dict with mapped keys
            new_state_dict = {}
            for key, value in saved_state.items():
                if key.startswith('gating_params.'):
                    if key in id_mapping:
                        new_state_dict[id_mapping[key]] = value
                else:
                    new_state_dict[key] = value
            
            # Load the modified state dict
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded successfully from PyTorch format: {model_path}!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error("Try running training again to generate a compatible model or modify the layer IDs manually.")
        return
    
    model.to(device)
    model.eval()
    
    # Check if the model has required attributes for operation
    required_attrs = ['embedding', 'output_layer', 'pos_encoder']
    for attr in required_attrs:
        if not hasattr(model, attr):
            logging.error(f"Model is missing required attribute: {attr}")
            return
    
    # Log model structure if debug is enabled
    if args.debug:
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        logging.info(f"Model has {len(model.gating_params)} gating parameters")
    
    # Load tokenizer with correct vocabulary size
    tokenizer = load_tokenizer(vocab_size=vocab_size)
    print("Tokenizer loaded successfully!")
    
    if args.chat:
        # Start interactive chat mode
        chat_with_model(
            model,
            tokenizer,
            system_prompt=system_prompt,
            max_length=args.max_length,
            debug_mode=args.debug
        )
    else:
        # Use normal text generation mode
        prompt = args.prompt
        print(f"Generating text from prompt: '{prompt}'...")
        
        try:
            # Generate text with improved parameters
            generated_text = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=min(200, args.max_length * 2),  # Doubled but capped at 200
                temperature=0.6,  # Lower temperature for more deterministic output
                system_prompt=system_prompt,
                instruct_mode=args.instruct,
                verbose=args.debug
            )
            print("\n" + "="*50)
            print(f"Prompt: {prompt}")
            print(f"Generated:\n{generated_text}")
            print("="*50)
            print("\nTip: Run with --chat flag to use interactive chat mode")
            print("     Run with --debug flag to see detailed logs")
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            import traceback
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()