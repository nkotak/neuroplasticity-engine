import sys
import os
import torch
import logging
import time
import math
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import Features, Value, Sequence
from tqdm import tqdm
import safetensors.torch as safetensors

# Add project_root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Existing code imports
from adaptive_network import (
    StructuralPlasticityController, TrainingManager, MetaLearner, MovingStats
)
from config import MultiModalConfig

# New transformer imports
from transformer_model import AdaptiveTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_bpe_tokenizer(texts, vocab_size=30000):
    """Use the adaptive WordPiece tokenizer if available, otherwise create a standard one."""
    # Check if we have an existing tokenizer to use
    tokenizer_path = "tokenizer/tokenizer.json"
    
    # Try to use the AdaptiveTokenizer if available
    try:
        from adaptive_tokenizer import AdaptiveTokenizer
        use_adaptive = True
        logging.info("AdaptiveTokenizer module is available")
    except ImportError:
        use_adaptive = False
        logging.info("AdaptiveTokenizer not available, will use standard tokenizer")
    
    if os.path.exists(tokenizer_path):
        if use_adaptive:
            # Use adaptive tokenizer with existing file
            logging.info(f"Loading existing tokenizer from {tokenizer_path} with adaptive features")
            tokenizer = AdaptiveTokenizer(
                initial_vocab_size=vocab_size,
                max_vocab_size=100000,
                tokenizer_path=tokenizer_path
            )
            current_vocab_size = tokenizer.get_vocab_size()
            logging.info(f"Using existing adaptive tokenizer with vocabulary size: {current_vocab_size}")
        else:
            # Use standard tokenizer
            logging.info(f"Loading existing tokenizer from {tokenizer_path}")
            tokenizer = Tokenizer.from_file(tokenizer_path)
            current_vocab_size = len(tokenizer.get_vocab())
            logging.info(f"Using existing tokenizer with vocabulary size: {current_vocab_size}")
    else:
        # Create a new tokenizer
        special_tokens = [
            "<pad>", "<unk>", "<s>", "</s>", 
            "<think>", "</think>",
            "<user>", "<assistant>", "<system>"
        ]
        
        if use_adaptive:
            # Create new adaptive tokenizer
            logging.info("Creating new adaptive WordPiece tokenizer")
            tokenizer = AdaptiveTokenizer(
                initial_vocab_size=vocab_size,
                max_vocab_size=100000,
                special_tokens=special_tokens
            )
            # Train the tokenizer
            tokenizer_path = tokenizer.train(texts)
            logging.info(f"Adaptive tokenizer trained and saved to {tokenizer_path}")
        else:
            # Create standard WordPiece tokenizer
            logging.info("Creating new standard WordPiece tokenizer")
            tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
            
            # Standard normalization
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(),
                normalizers.StripAccents(),
            ])
            
            # Use Whitespace pre-tokenizer
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Use WordPiece decoder
            tokenizer.decoder = decoders.WordPiece()
            
            # Create WordPiece trainer
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=special_tokens,
                show_progress=True
            )
            
            # Train the tokenizer
            logging.info(f"Training WordPiece tokenizer on {len(texts)} text samples with vocab size {vocab_size}")
            tokenizer.train_from_iterator(texts, trainer)
            
            # Make sure the output directory exists
            os.makedirs("tokenizer", exist_ok=True)
            
            # Save the tokenizer
            tokenizer.save(tokenizer_path)
            logging.info(f"Tokenizer saved to {tokenizer_path}")
            
            # Also save to root directory for inference
            tokenizer.save("tokenizer.json")
    
    # Test the tokenizer
    test_text = "System: You are a helpful assistant.\n\nUser: Hello, what is your name?\n\nAssistant:"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    logging.info(f"Tokenizer test - encoded to {len(encoded.ids)} tokens")
    logging.info(f"Roundtrip match: {test_text == decoded}")
    
    # Special test for thinking tags
    thinking_test = "System: You are a helpful assistant.\n\nUser: What is 2+2?\n\nAssistant: <think>This is a simple math problem.</think> The answer is 4."
    encoded = tokenizer.encode(thinking_test)
    decoded = tokenizer.decode(encoded.ids)
    logging.info(f"Thinking tag test - parsed thinking tags: {'<think>' in decoded and '</think>' in decoded}")
        
    return tokenizer

def get_combined_text(datasets_info, num_examples_per_dataset=10000):
    """Extract text from various columns across all datasets."""
    combined_text = []
    text_columns = [
    'text', 'messages', 'reasoning', 'answer',  # Core dolphin columns
    'input', 'output', 'question', 'response',  # Original columns
    'prompt', 'content', 'reasoning_content',
    'solution', 'conversations', 'system'
]
    
    for name, config in datasets_info:
        try:
            # Handle UltraChat dataset separately with its specific split
            if name == "HuggingFaceH4/ultrachat_200k":
                load_params = {
                    'path': name,
                    'trust_remote_code': True,
                    'split': 'train_sft'  # UltraChat's specific split
                }
            else:
                # Standard loading for other datasets
                load_params = {
                    'path': name,
                    'trust_remote_code': True,
                    'split': 'train'
                }
                if config is not None:
                    load_params['name'] = config
            
            dataset = load_dataset(**load_params)
            logging.info(f"Loaded dataset {name} with config {config}")
            
            # Find available text-like columns
            available_columns = [col for col in text_columns if col in dataset.column_names]
            if not available_columns:
                logging.warning(f"No text-like columns found in {name}")
                continue
            
            # Extract up to num_examples_per_dataset, but use a larger number
            logging.info(f"Processing {min(num_examples_per_dataset, len(dataset))} examples from {name}")
            for i in range(min(num_examples_per_dataset, len(dataset))):
                example_text = []
                for col in available_columns:
                    value = dataset[i][col]
                    if isinstance(value, str):
                        # Only add if the string has meaningful content
                        if len(value.strip()) > 5:  # Skip very short strings
                            example_text.append(value)
                    elif isinstance(value, list):
                        if col in ['conversations', 'messages']:
                            # Enhanced message processing with better formatting
                            for conv in value:
                                if isinstance(conv, dict) and 'content' in conv and len(str(conv.get('content', '')).strip()) > 5:
                                    role = conv.get('role', '')
                                    content = conv.get('content', '')
                                    example_text.append(f"{role}: {content}")
                        else:
                            # Filter out empty or very short strings from lists
                            example_text.extend([str(item) for item in value if len(str(item).strip()) > 5])
                    elif value is not None and len(str(value).strip()) > 5:
                        example_text.append(str(value))
                
                if example_text:
                    # Join with newlines between different data elements for better context boundaries
                    combined_text.append("\n".join(example_text))
                
                # Log progress periodically
                if i > 0 and i % 5000 == 0:
                    logging.info(f"Processed {i} examples from {name}")
                    
        except Exception as e:
            logging.error(f"Failed to load dataset {name} with config {config}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    # Log the amount of text collected for tokenization
    total_chars = sum(len(text) for text in combined_text)
    logging.info(f"Collected {len(combined_text)} text samples with {total_chars} total characters for tokenization")
    
    return combined_text

# Custom system prompt to inject during training - modified for better language modeling
SYSTEM_PROMPT = """You are an expert assistant trained to provide helpful, accurate, and coherent responses. When responding to users, think step-by-step, use precise language, and ensure your responses are well-structured and logically sound. Your objective is to provide the most useful and well-reasoned answers possible, drawing from your knowledge to help users understand complex topics."""

# Update tokenize_dataset to handle instruction formatting and thinking tags
def tokenize_dataset(dataset, tokenizer, max_length=128, add_system_prompt=True, add_thinking_tags=True):
    """Tokenize a dataset with instruction formatting, system prompt, and thinking tags"""
    text_columns = [
        'text', 'messages', 'reasoning', 'answer',
        'input', 'output', 'question', 'response',
        'prompt', 'content', 'reasoning_content',
        'solution', 'conversations', 'system'
    ]
    
    available_columns = [col for col in text_columns if col in dataset.column_names]
    
    def tokenize_function(examples):
        texts = []
        
        for i in range(len(examples[available_columns[0]])):
            # Format as instruction-following format
            instruction = ""
            response = ""
            thinking = ""
            
            # Extract instruction/response and thinking from different columns
            for col in available_columns:
                value = examples[col][i]
                
                if col in ['conversations', 'messages'] and isinstance(value, list):
                    # Process conversation format
                    for turn in value:
                        if turn.get('role', '') == 'user':
                            instruction += f"{turn.get('content', '')}\n"
                        elif turn.get('role', '') == 'assistant':
                            response += f"{turn.get('content', '')}\n"
                        elif turn.get('role', '') == 'system':
                            # Extract system message, but we'll use our custom one
                            pass
                
                elif col in ['input', 'question', 'prompt']:
                    # These columns likely contain instructions
                    if isinstance(value, str):
                        instruction += f"{value}\n"
                
                elif col in ['reasoning', 'reasoning_content']:
                    # These columns contain reasoning content for thinking tags
                    if isinstance(value, str):
                        thinking += f"{value}\n"
                
                elif col in ['output', 'response', 'solution', 'answer']:
                    # These columns likely contain responses
                    if isinstance(value, str):
                        response += f"{value}\n"
            
            # Format the final text with system prompt and thinking tags
            formatted_text = ""
            
            # Add system prompt 
            if add_system_prompt:
                formatted_text += f"System: {SYSTEM_PROMPT}\n\n"
            
            # Add instruction
            if instruction:
                formatted_text += f"User: {instruction.strip()}\n\n"
            
            # Add thinking tags
            if add_thinking_tags and thinking:
                formatted_text += f"Assistant: <think>{thinking.strip()}</think>\n\n"
            
            # Add response
            if response:
                formatted_text += f"Assistant: {response.strip()}"
            
            texts.append(formatted_text)
        
        # Manual padding/truncation
        pad_token_id = tokenizer.token_to_id("<pad>")
        input_ids = []
        for text in texts:
            encoding = tokenizer.encode(text)
            ids = encoding.ids[:max_length]
            ids += [pad_token_id] * (max_length - len(ids))
            input_ids.append(ids[:max_length])
        
        return {'input_ids': input_ids}
    
    tokenized = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        features=Features({'input_ids': Sequence(Value('int32'))})
    )
    tokenized.set_format(type='torch', columns=['input_ids'])
    return tokenized
    
def main():
    # Define device first
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Define datasets with optional configurations - expanded dataset list
    datasets_info = [
        ("cognitivecomputations/dolphin-r1", "reasoning-deepseek"),
        # UltraChat needs the specific train_sft split
        ("HuggingFaceH4/ultrachat_200k", "train_sft"),
        ("tatsu-lab/alpaca", None),
        ("databricks/databricks-dolly-15k", None)
    ]

    # Step 1: Load or train the adaptive tokenizer
    logging.info("Loading or training adaptive WordPiece tokenizer...")
    combined_text = get_combined_text(datasets_info)
    tokenizer = train_bpe_tokenizer(combined_text)
    pad_token_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    logging.info(f"Tokenizer trained with vocab size: {vocab_size}")
    
    # Print token distribution statistics
    token_counts = {}
    for text in combined_text[:1000]:  # Use a subset for counting
        encoded = tokenizer.encode(text)
        for token_id in encoded.ids:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    logging.info(f"Token distribution: min={min(token_counts.values())}, max={max(token_counts.values())}, "
                f"unique={len(token_counts)}/{vocab_size}")
    
    # Save the tokenizer files
    tokenizer_dir = "tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save the tokenizer in the HuggingFace format
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logging.info(f"Tokenizer saved to {tokenizer_path}")

    # Step 2: Tokenize all datasets with shorter sequence length
    tokenized_datasets = []
    max_length = 64  # Reduced from 128 for faster training
    for name, config in datasets_info:
        try:
            logging.info(f"Loading and tokenizing dataset: {name} with config {config}")
            load_params = {
                'path': name,
                'trust_remote_code': True,
                'split': 'train'
            }
            if config is not None:
                load_params['name'] = config
            
            # Load and standardize schema first
            dataset = load_dataset(**load_params)
            
            tokenized = tokenize_dataset(dataset, tokenizer, max_length, 
                                        add_system_prompt=True, 
                                        add_thinking_tags=True)
            tokenized_datasets.append(tokenized)
        except Exception as e:
            logging.warning(f"Failed to process dataset {name} with config {config}: {e}")

    # Step 3: Combine tokenized datasets
    if not tokenized_datasets:
        logging.error("No datasets were successfully tokenized. Exiting.")
        return
    logging.info("Concatenating tokenized datasets...")
    combined_tokenized = concatenate_datasets(tokenized_datasets)

    # Step 4: Split into train and validation sets
    split_ratio = 0.9
    num_examples = len(combined_tokenized)
    train_size = int(split_ratio * num_examples)
    train_dataset = combined_tokenized.select(range(train_size))
    val_dataset = combined_tokenized.select(range(train_size, num_examples))

    # Update the DataLoader creation to handle tensor conversion
    # Step 5: Create DataLoaders with proper collation
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch])
        }

    # Use smaller batch size for faster training
    batch_size = 32
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Step 6: Initialize a much smaller adaptive transformer model (~30M parameters)
    # To enable multi-modal training with text + image:
    #   mm_config = MultiModalConfig(
    #       text_vocab_size=vocab_size, text_d_model=256,
    #       image_size=256, image_patch_size=16, image_d_model=256,
    #       enabled_modalities=['text', 'image'],
    #   )
    #   modality_config = mm_config.to_modality_dict()
    modality_config = None  # Text-only mode for initial training

    model = AdaptiveTransformer(
        vocab_size=vocab_size,
        d_model=256,     # Reduced from 384 to 256
        nhead=4,         # Reduced from 6 to 4
        num_layers=4,    # Reduced from 6 to 4
        d_ff=1024,       # Reduced from 1536 to 1024
        max_seq_len=max_length,
        device=device,
        modality_config=modality_config
    )
    
    logging.info(f"Initialized model with approximately 30M parameters")

    # Step 7: Set up structural plasticity controller with gentler settings
    sp_controller = StructuralPlasticityController(
        prune_threshold=0.001,  # Reduced from 0.005 for less aggressive pruning
        growth_rate=0.1,        # Reduced from 0.2 for more gradual growth
        split_threshold=0.6,    # Increased from 0.5 to make splitting more selective
        merge_threshold=0.5,    # Increased from 0.4 to make merging more selective
        prune_frequency=20,     # Increased from 10 to prune less frequently
        structural_frequency=30, # Increased from 20 for more stability
        min_connections=2000    # Increased from 1000 for better connectivity
    )

    # Step 8: Customize TrainingManager for language modeling with mixed precision and progress tracking
    class LMTrainingManager(TrainingManager):
        def __init__(self, network, sp_controller, lr, batch_size, pad_token_id, device):
            # Store lr attribute first
            self.lr = lr  # Explicitly set lr before using it
            # Initialize the base class without device parameter
            super().__init__(network, sp_controller, lr=lr, batch_size=batch_size)
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
            self.device = device  # Store device separately
            self.network.to(device)  # Ensure network is on correct device
            
            # Initialize training components with more stability
            # For Apple Silicon (MPS), we adapt our approach
            # Disable mixed precision on CUDA since it's causing issues
            self.use_mixed_precision = False  # Disable mixed precision entirely
            self.scaler = None  # No scaler needed
            
            # Special CUDA-specific optimizer settings for stability
            if device.type == 'cuda':
                # For CUDA, use a very small initial learning rate with warmup
                self.base_lr = 1e-5  # Start extremely small
                self.target_lr = self.lr  # Target LR after warmup
                self.warmup_steps = 5000  # Number of steps for warmup
                self.current_step = 0
                
                # Use standard regularization (match what worked on MPS)
                self.optimizer = torch.optim.AdamW(
                    self.network.parameters(),
                    lr=self.base_lr,  # Start with very small LR
                    weight_decay=0.01,  # Standard regularization - like MPS setting
                    eps=1e-8,
                    betas=(0.9, 0.95)  # Standard momentum parameters
                )
                
                # Set gradient accumulation for more stable updates
                self.grad_accumulation_steps = 4
                self.accumulated_steps = 0
            else:
                # For other devices (MPS/CPU) use original settings
                self.optimizer = torch.optim.AdamW(
                    self.network.parameters(),
                    lr=self.lr,
                    weight_decay=0.01,
                    eps=1e-8
                )
                self.grad_accumulation_steps = 1  # No accumulation
            
            # Track time and iterations for better progress reporting
            self.iteration_time = 0
            self.tokens_processed = 0
            self.start_time = time.time()
            
        def train_epoch(self, train_data, val_data=None):
            """Override TrainingManager's train_epoch with improved progress tracking"""
            self.network.train()
            train_loss = 0.0
            num_batches = 0
            total_tokens = 0
            batch_times = []
            
            # Add progress bar for training batches
            with tqdm(total=len(train_data), desc="Training", position=1, leave=False) as batch_pbar:
                for batch in train_data:
                    batch_start = time.time()
                    metrics = self.train_step(batch)
                    
                    # Update metrics
                    train_loss += metrics['loss']
                    num_batches += 1
                    
                    # Track batch time
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    # Update progress bar with current metrics
                    batch_pbar.update(1)
                    batch_pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'tok/s': f"{metrics.get('tokens_per_sec', 0):.0f}"
                    })
                
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            # Apply structural plasticity update per epoch
            logging.info("Applying structural plasticity updates...")
            self.structural_plasticity.step(self.network)
            
            # Import constants from adaptive_network or define them here
            ADAPT_REWARD_SCALE = 1.0
            TARGET_SPARSITY = 0.0025
            
            # Compute adaptation reward
            current_global_sparsity = self.structural_plasticity.get_global_sparsity(self.network)
            adaptation_reward = -ADAPT_REWARD_SCALE * abs(current_global_sparsity - TARGET_SPARSITY)
            
            if val_data is not None:
                val_metrics = self.validate(val_data)
                self.network.meta_step(val_metrics['val_loss'])
                return {
                    'train_loss': avg_train_loss,
                    'val_loss': val_metrics['val_loss'],
                    'tokens_per_sec': total_tokens / sum(batch_times) if batch_times else 0,
                    'avg_batch_time': avg_batch_time,
                    'adaptation_reward': adaptation_reward
                }
            else:
                return {
                    'train_loss': avg_train_loss,
                    'tokens_per_sec': total_tokens / sum(batch_times) if batch_times else 0,
                    'avg_batch_time': avg_batch_time,
                    'adaptation_reward': adaptation_reward
                }
            
        def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
            """Training step with CUDA-specific adaptations and multi-modal support"""
            self.network.train()
            
            # Using zero_grad(set_to_none=True) for better performance and memory efficiency
            self.optimizer.zero_grad(set_to_none=True)
            
            step_start = time.time()
            
            # Handle multi-modal batches
            modality_inputs = None
            targets = None
            domain_id = 0
            
            if hasattr(self.network, 'is_multimodal') and self.network.is_multimodal:
                # Multi-modal mode
                if any(k in {'text', 'image', 'audio'} for k in batch.keys()):
                    # Multi-modal batch
                    modality_inputs = {
                        k: v.to(self.device) for k, v in batch.items()
                        if k in {'text', 'image', 'audio'}
                    }
                    
                    # If we have text, use it for targets
                    if 'text' in modality_inputs:
                        text_input = modality_inputs['text']
                        targets = text_input[:, 1:]  # Shift for autoregressive prediction
                        
                    # If we have a domain identifier
                    if 'domain_id' in batch:
                        domain_id = batch['domain_id'].item()
                        
                    # Set input_ids to None for multi-modal mode
                    input_ids = None
                else:
                    # Fall back to text-only mode with input_ids
                    input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
            else:
                # Standard text-only mode
                input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
            
            # Add noise to prevent model from memorizing
            if self.device.type == 'cuda':
                # Teacher forcing with a lower probability
                teacher_forcing_ratio = 0.7  # Start lower to avoid overfitting
                
                # Occasionally replace some tokens with random ones (10%)
                if np.random.random() < 0.1:
                    random_mask = torch.rand_like(input_ids.float()) < 0.05
                    random_tokens = torch.randint_like(input_ids, 0, self.network.embedding.weight.size(0))
                    input_ids = torch.where(random_mask, random_tokens, input_ids)
            batch_size, seq_len = input_ids.size()
            self.tokens_processed += batch_size * seq_len
            
            # Mixed precision training
            if self.use_mixed_precision and self.device.type == 'cuda':
                try:
                    # CUDA mixed precision path with error checking
                    with torch.amp.autocast('cuda'):
                        # Handle multi-modal vs text-only forward pass
                        if modality_inputs is not None:
                            output = self.network(None, modality_inputs, domain_id)
                            
                            # For multi-modal, we might need to set targets
                            if targets is None:
                                # If no explicit targets provided (e.g. image-only input),
                                # create dummy targets for loss calculation
                                targets = torch.zeros(
                                    output.size(0), output.size(1) - 1, 
                                    dtype=torch.long, device=output.device
                                )
                        else:
                            output = self.network(input_ids)
                            # Get targets from input_ids for autoregressive prediction
                            targets = input_ids[:, 1:].contiguous()
                            
                        # CUDA-specific approach with temperature scaling
                        logits = output[:, :-1, :].contiguous()
                        
                        batch_size, seq_len, vocab_size = logits.size()
                        
                        # Use standard temperature (no scaling) for better modeling
                        temperature = 1.0  # No temperature scaling during training
                        logits_scaled = logits / temperature
                        
                        # Completely remove token uniformity regularization - using standard loss instead
                        distribution_loss = 0.0  # No distribution loss
                        
                        logits_flat = logits_scaled.view(batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        
                        # Verify no out-of-bounds indices
                        if torch.max(targets_flat) >= vocab_size:
                            logging.error(f"Target index {torch.max(targets_flat).item()} >= vocab size {vocab_size}")
                            targets_flat = torch.clamp(targets_flat, 0, vocab_size-1)
                        
                        # Switch to standard cross entropy without label smoothing
                        # This matches what worked on MPS
                        loss = F.cross_entropy(
                            logits_flat,
                            targets_flat,
                            reduction='mean',
                            ignore_index=self.criterion.ignore_index
                        )
                        
                        # Prevent absurdly small losses and exploding gradients
                        # Allow proper loss values for learning
                        loss = torch.clamp(loss, min=0.0, max=30.0)
                    
                    # Check for non-finite loss
                    loss_value = loss.item()
                    
                    if not torch.isfinite(loss):
                        logging.error(f"Non-finite loss detected: {loss_value}")
                        # Fall back to non-mixed precision
                        with torch.no_grad():
                            self.network = self.network.to(torch.float32)
                            output = self.network(input_ids)
                            logits = output[:, :-1, :].contiguous()
                            targets = input_ids[:, 1:].contiguous()
                            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    # Always add a small amount of L2 regularization to prevent collapse
                    l2_reg = sum(param.pow(2.0).sum() for param in self.network.parameters())
                    loss = loss + 0.0001 * l2_reg

                    # Add auxiliary loss from domain experts (multi-modal MoE balancing)
                    if self.network.last_aux_loss is not None:
                        loss = loss + self.network.last_aux_loss

                    # Apply gradient clipping to prevent numerical issues
                    self.scaler.scale(loss).backward()
                    
                    # Collect metrics before optimization
                    gradients = self._collect_gradient_metrics()
                    activations = self._collect_activation_metrics()
                    
                    # Update skip connections if controller exists
                    if hasattr(self.network, 'skip_connection_controller'):
                        skip_connections_added = self.network.skip_connection_controller.step(
                            self.network.registry, self.network, gradients
                        )
                        if skip_connections_added > 0:
                            logging.info(f"Added {skip_connections_added} new skip connections")
                    
                    # Clip gradients to prevent exploding gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    
                    # Perform scaled optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception as e:
                    logging.error(f"Error in CUDA mixed precision path: {e}")
                    # Fallback to standard precision path
                    self.network = self.network.to(torch.float32)
                    
                    # Handle multi-modal vs text-only forward pass
                    if modality_inputs is not None:
                        output = self.network(None, modality_inputs, domain_id)
                        
                        # For multi-modal, we might need to set targets
                        if targets is None:
                            # If no explicit targets provided (e.g. image-only input),
                            # create dummy targets for loss calculation
                            targets = torch.zeros(
                                output.size(0), output.size(1) - 1, 
                                dtype=torch.long, device=output.device
                            )
                    else:
                        output = self.network(input_ids)
                        # Get targets from input_ids for autoregressive prediction
                        targets = input_ids[:, 1:].contiguous()
                    
                    # Apply same careful checks in non-mixed precision path
                    logits = output[:, :-1, :].contiguous()
                    
                    batch_size, seq_len, vocab_size = logits.size()
                    logits_flat = logits.view(batch_size * seq_len, vocab_size)
                    targets_flat = targets.view(batch_size * seq_len)
                    
                    if torch.max(targets_flat) >= vocab_size:
                        logging.error(f"Target index {torch.max(targets_flat).item()} >= vocab size {vocab_size}")
                        targets_flat = torch.clamp(targets_flat, 0, vocab_size-1)
                        
                    loss = self.criterion(logits_flat, targets_flat)

                    # Add auxiliary loss from domain experts (multi-modal MoE balancing)
                    if self.network.last_aux_loss is not None:
                        loss = loss + self.network.last_aux_loss

                    if loss.item() < 1e-8 or torch.isnan(loss):
                        logging.error(f"Loss is problematic: {loss.item()}, using fallback loss")
                        loss = loss + torch.randn(1, device=self.device)[0] * 0.01 + 5.0
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                # MPS or CPU path - standard execution
                if self.device.type == 'mps':
                    # Use standard float32 on MPS for compatibility
                    # MPS doesn't handle mixed precision well with bfloat16
                    network_dtype = next(self.network.parameters()).dtype
                    if network_dtype != torch.float32:
                        self.network = self.network.to(torch.float32)
                
                # Ensure consistent loss calculations between CUDA and MPS paths
                
                # Handle multi-modal vs text-only forward pass
                if modality_inputs is not None:
                    output = self.network(None, modality_inputs, domain_id)
                    
                    # For multi-modal, we might need to set targets
                    if targets is None:
                        # If no explicit targets provided (e.g. image-only input),
                        # create dummy targets for loss calculation
                        targets = torch.zeros(
                            output.size(0), output.size(1) - 1, 
                            dtype=torch.long, device=output.device
                        )
                else:
                    output = self.network(input_ids)
                    # Get targets from input_ids for autoregressive prediction
                    targets = input_ids[:, 1:].contiguous()
                
                logits = output[:, :-1, :].contiguous()
                
                batch_size, seq_len, vocab_size = logits.size()
                
                # Use standard temperature for consistency
                temperature = 1.0
                logits_scaled = logits / temperature
                
                # Remove token uniformity regularization here too for consistency
                distribution_loss = 0.0  # No distribution loss
                
                # Reshape for loss calculation
                logits_flat = logits_scaled.view(batch_size * seq_len, vocab_size)
                targets_flat = targets.view(batch_size * seq_len)
                
                # Validate target values
                if torch.max(targets_flat) >= vocab_size:
                    logging.warning(f"Validation: Target index {torch.max(targets_flat).item()} >= vocab size {vocab_size}")
                    targets_flat = torch.clamp(targets_flat, 0, vocab_size-1)
                
                # Use same loss calculation as training (label smoothing + regularization)
                smoothing = 0.1
                one_hot = torch.zeros_like(logits_flat).scatter_(
                    1, targets_flat.unsqueeze(1), 1.0
                )
                smooth_targets = one_hot * (1.0 - smoothing) + smoothing / vocab_size
                
                # Cross entropy with label smoothing
                loss = -(smooth_targets * F.log_softmax(logits_flat, dim=1)).sum(dim=1).mean()
                
                # Add distribution regularization
                loss = loss + distribution_loss

                # Add auxiliary loss from domain experts (multi-modal MoE balancing)
                if self.network.last_aux_loss is not None:
                    loss = loss + self.network.last_aux_loss

                # Force a minimum high loss value for CUDA to prevent collapse
                if self.device.type == 'cuda':
                    # Remove loss clamping entirely - let the model learn naturally
                    # loss = torch.clamp(loss, min=0.5, max=15.0)
                    
                    # Scale loss by accumulation steps
                    scaled_loss = loss / self.grad_accumulation_steps
                    scaled_loss.backward()
                    
                    # Learning rate warmup for CUDA
                    self.current_step += 1
                    if self.current_step < self.warmup_steps:
                        # Linear warmup
                        lr = self.base_lr + (self.target_lr - self.base_lr) * (self.current_step / self.warmup_steps)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    
                    # Only update on accumulation boundary or last batch
                    self.accumulated_steps += 1
                    if self.accumulated_steps >= self.grad_accumulation_steps:
                        # Collect metrics before optimization
                        gradients = self._collect_gradient_metrics()
                        activations = self._collect_activation_metrics()
                        
                        # Update skip connections if controller exists
                        if hasattr(self.network, 'skip_connection_controller'):
                            skip_connections_added = self.network.skip_connection_controller.step(
                                self.network.registry, self.network, gradients
                            )
                            if skip_connections_added > 0:
                                logging.info(f"Added {skip_connections_added} new skip connections")
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        self.accumulated_steps = 0
                    else:
                        # Skip metrics collection on non-update steps
                        gradients = {}
                        activations = {}
                else:
                    # Normal path for MPS/CPU
                    # Allow a wider range of loss values for better learning
                    loss = torch.clamp(loss, min=0.0, max=30.0)
                    loss.backward()
                    gradients = self._collect_gradient_metrics()
                    activations = self._collect_activation_metrics()
                    
                    # Update skip connections if controller exists
                    if hasattr(self.network, 'skip_connection_controller'):
                        skip_connections_added = self.network.skip_connection_controller.step(
                            self.network.registry, self.network, gradients
                        )
                        if skip_connections_added > 0:
                            logging.info(f"Added {skip_connections_added} new skip connections")
                    
                    self.optimizer.step()
                
                # Return to original dtype if changed
                if self.device.type == 'mps' and network_dtype != torch.float32:
                    self.network = self.network.to(network_dtype)

            self.network.update_metrics(gradients, activations)
            self.network.increment_warmup()
            
            # Update timing metrics
            self.iteration_time = time.time() - step_start
            elapsed = time.time() - self.start_time
            tokens_per_sec = self.tokens_processed / elapsed if elapsed > 0 else 0
            
            return {
                'loss': loss.item(),
                'aux_loss': self.network.last_aux_loss.item() if self.network.last_aux_loss is not None else 0.0,
                'tokens_per_sec': tokens_per_sec,
                'iteration_time': self.iteration_time
            }

        def validate(self, val_data: DataLoader) -> dict[str, float]:
            self.network.eval()
            val_loss = 0.0
            num_batches = 0
            total_tokens = 0
            
            # Use tqdm for validation progress display
            with torch.no_grad(), tqdm(total=len(val_data), desc="Validating", leave=False) as pbar:
                for batch in val_data:
                    # Handle multi-modal batches
                    modality_inputs = None
                    targets = None
                    domain_id = 0
                    
                    if hasattr(self.network, 'is_multimodal') and self.network.is_multimodal:
                        # Multi-modal mode
                        if any(k in {'text', 'image', 'audio'} for k in batch.keys()):
                            # Multi-modal batch
                            modality_inputs = {
                                k: v.to(self.device) for k, v in batch.items()
                                if k in {'text', 'image', 'audio'}
                            }
                            
                            # If we have text, use it for targets
                            if 'text' in modality_inputs:
                                text_input = modality_inputs['text']
                                targets = text_input[:, 1:]
                                
                            # If we have a domain identifier
                            if 'domain_id' in batch:
                                domain_id = batch['domain_id'].item()
                                
                            # For token counting
                            batch_size = list(modality_inputs.values())[0].size(0)
                            seq_len = 0
                            for v in modality_inputs.values():
                                if len(v.shape) > 1:
                                    seq_len += v.size(1)
                            total_tokens += batch_size * seq_len
                            
                            # No input_ids for multi-modal mode
                            input_ids = None
                        else:
                            # Fall back to text-only mode with input_ids
                            input_ids = batch['input_ids'].to(self.device)
                            batch_size, seq_len = input_ids.size()
                            total_tokens += batch_size * seq_len
                    else:
                        # Standard text-only mode
                        input_ids = batch['input_ids'].to(self.device)
                        batch_size, seq_len = input_ids.size()
                        total_tokens += batch_size * seq_len
                    
                    # Standard handling for MPS - needed only for MPS device
                    if self.device.type == 'mps':
                        original_dtype = next(self.network.parameters()).dtype
                        if original_dtype != torch.float32:
                            self.network = self.network.to(torch.float32)
                    
                    # Use EXACT same approach as training for validation
                    if modality_inputs is not None:
                        output = self.network(None, modality_inputs, domain_id)
                        
                        # For multi-modal, we might need to set targets
                        if targets is None:
                            # If no explicit targets provided (e.g. image-only input),
                            # create dummy targets for loss calculation
                            targets = torch.zeros(
                                output.size(0), output.size(1) - 1, 
                                dtype=torch.long, device=output.device
                            )
                    else:
                        output = self.network(input_ids)
                        # Get targets from input_ids for autoregressive prediction
                        targets = input_ids[:, 1:].contiguous()
                    
                    logits = output[:, :-1, :].contiguous()
                    
                    batch_size, seq_len, vocab_size = logits.size()
                    
                    # Use temperature scaling consistently with training
                    temperature = 1.2  # Same as training
                    logits_scaled = logits / temperature
                    
                    # Completely remove token uniformity regularization in validation too
                    distribution_loss = 0.0  # No distribution loss
                    
                    # Reshape and validate same as training
                    logits_flat = logits_scaled.view(batch_size * seq_len, vocab_size)
                    targets_flat = targets.view(batch_size * seq_len)
                    
                    if torch.max(targets_flat) >= vocab_size:
                        logging.warning(f"Validation: Target index {torch.max(targets_flat).item()} >= vocab size {vocab_size}")
                        targets_flat = torch.clamp(targets_flat, 0, vocab_size-1)
                    
                    # Use standard cross entropy loss in validation too
                    loss = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        reduction='mean',
                        ignore_index=self.criterion.ignore_index
                    )
                    
                    # NO clamping for validation to get true loss value
                    val_loss += loss.item()
                    
                    # Return to original dtype if needed
                    if self.device.type == 'mps' and original_dtype != torch.float32:
                        self.network = self.network.to(original_dtype)
                    
                    # Update progress
                    num_batches += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())
                    
            avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
            
            # More robust perplexity calculation
            try:
                # Cap perplexity at a large but finite value for readability
                if avg_val_loss > 20:
                    perplexity = 9999.99  # Cap at high but finite value for display
                    logging.warning(f"Very high validation loss: {avg_val_loss:.4f}, capping perplexity display value")
                else:
                    perplexity = math.exp(avg_val_loss)
            except OverflowError:
                # Handle math overflow for extremely large losses
                perplexity = 9999.99
                logging.warning(f"Perplexity computation overflow for loss: {avg_val_loss:.4f}")
            
            return {
                'val_loss': avg_val_loss,
                'perplexity': perplexity,
                'tokens_processed': total_tokens
            }

    # Step 9: Initialize training manager with smaller batch size
    trainer = LMTrainingManager(
        model,
        sp_controller,
        lr=0.003,  # Increased learning rate for better training
        batch_size=batch_size,  # Use the same batch size as DataLoader
        pad_token_id=pad_token_id,
        device=device
    )

    # Step 10: Training loop with improved parameters
    num_epochs = 40  # Increased for better learning
    early_stopping_patience = 10  # Increased patience to allow more exploration
    min_epochs_before_stopping = 15  # Minimum number of epochs before early stopping kicks in
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'adaptation_reward': [], 'perplexity': []}
    meta_learner = MetaLearner(state_size=256, action_size=2)  # Adjusted to match model's d_model

    # Add progress tracking for epochs
    with tqdm(total=num_epochs, desc="Training Progress", position=0) as epoch_pbar:
        for epoch in range(num_epochs):
            # Track time for each epoch
            epoch_start_time = time.time()
            
            # Train with improved progress tracking
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training and validation with progress bars handled internally
            train_metrics = trainer.train_epoch(train_loader, val_loader)
            val_metrics = trainer.validate(val_loader)

            state = torch.randn(256)  # Placeholder state (adjusted to match d_model)
            action = torch.tensor([0])  # Placeholder action
            reward = -val_metrics['val_loss']
            adaptation_reward = train_metrics.get('adaptation_reward', 0.0)

            meta_learner.memory.push({
                'state': state,
                'action': action,
                'reward': reward,
                'adaptation_reward': adaptation_reward
            })

            if epoch % 10 == 0 and len(meta_learner.memory) >= meta_learner.batch_size:
                meta_stats = meta_learner.update_from_experience()
                logging.info(f"Meta-Learner Update: {meta_stats}")
                
                # Check if we should grow the tokenizer vocabulary every 10 epochs
                from adaptive_tokenizer import AdaptiveTokenizer
                if hasattr(tokenizer, 'grow_vocabulary') or isinstance(tokenizer, AdaptiveTokenizer):
                    try:
                        logging.info("Checking if tokenizer vocabulary needs to grow...")
                        # Use a sample of text for vocabulary growth
                        sample_text = combined_text[:2000]  # Sample for efficiency
                        if isinstance(tokenizer, AdaptiveTokenizer):
                            tokenizer_path = tokenizer.grow_vocabulary(sample_text)
                            if tokenizer_path:
                                logging.info(f"Tokenizer vocabulary has grown, saved to {tokenizer_path}")
                                # Update vocab size for the model
                                vocab_size = tokenizer.get_vocab_size()
                        else:
                            logging.info("Tokenizer is not adaptive, skipping vocabulary growth")
                    except Exception as e:
                        logging.warning(f"Could not grow tokenizer vocabulary: {e}")

            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['adaptation_reward'].append(adaptation_reward)

            # Calculate performance metrics
            epoch_time = time.time() - epoch_start_time
            # Use same perplexity calculation as in validate function for consistency
            try:
                if val_metrics['val_loss'] > 20:
                    perplexity = 9999.99
                else:
                    perplexity = math.exp(val_metrics['val_loss'])
            except (OverflowError, KeyError):
                perplexity = 9999.99
            tokens_per_sec = train_metrics.get('tokens_per_sec', 0.0)
            
            # Calculate ETA
            remaining_epochs = num_epochs - epoch - 1
            eta_seconds = epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Detailed logging with multiple metrics including ETA
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Perplexity: {perplexity:.2f} - "
                f"Speed: {tokens_per_sec:.0f} tokens/sec - "
                f"Epoch Time: {epoch_time:.1f}s - "
                f"ETA: {eta_hours:.1f}h {eta_minutes % 60:.1f}m - "
                f"Adaptation Reward: {adaptation_reward:.4f}"
            )
            
            # Update epoch progress bar
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'loss': f"{train_metrics['train_loss']:.4f}",
                'val_loss': f"{val_metrics['val_loss']:.4f}",
                'ppl': f"{perplexity:.2f}"
            })
            
            # Calculate perplexity more reliably and store it
            try:
                if val_metrics['val_loss'] > 20:
                    perplexity = 9999.99
                else:
                    perplexity = math.exp(val_metrics['val_loss'])
            except (OverflowError, KeyError):
                perplexity = 9999.99
                
            # Store perplexity in history
            history['perplexity'].append(perplexity)
            
            # Early stopping check - only after minimum epochs
            if val_metrics['val_loss'] < best_val_loss:
                improvement = best_val_loss - val_metrics['val_loss']
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Log meaningful improvement
                logging.info(f"Validation loss improved by {improvement:.6f} (relative improvement: {improvement/best_val_loss*100:.2f}%)")
                
                # Save model in two formats: PyTorch and safetensors
                # 1. Save PyTorch format
                torch.save(model.state_dict(), 'best_model.pt')
                
                # 2. Save safetensors format
                safetensors.save_model(model, 'best_model.safetensors')
                
                # Count the actual number of layers in the model
                active_layer_count = sum(1 for lid, layer_info in model.registry.layers.items() 
                                         if layer_info['active'])
                
                # Save model config information with accurate layer count
                config = {
                    "model_type": "adaptive_transformer_instruct",
                    "vocab_size": vocab_size,
                    "d_model": 256,
                    "nhead": 4,
                    "num_layers": active_layer_count,  # Use actual count of active layers
                    "d_ff": 1024,
                    "max_seq_len": max_length,
                    "system_prompt": SYSTEM_PROMPT,
                    "supports_thinking_tags": True,
                    "best_epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "perplexity": perplexity,
                    "layer_ids": [lid for lid, layer_info in model.registry.layers.items()
                                 if layer_info['active']],
                    "modality_config": modality_config,  # None for text-only, dict for multi-modal
                }
                
                # Save config
                import json
                with open('config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Copy tokenizer
                os.system(f"cp {tokenizer_path} .")
                
                epoch_pbar.set_description(f"Training Progress (new best: {best_val_loss:.4f}, ppl: {perplexity:.2f})")
            else:
                patience_counter += 1
                
                # Only perform early stopping after minimum number of epochs
                if epoch + 1 >= min_epochs_before_stopping and patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1} after {patience_counter} epochs without improvement")
                    logging.info(f"Best validation loss was {best_val_loss:.6f} with perplexity {math.exp(best_val_loss):.2f} if valid")
                    
                    # Save final model even if it's not the best
                    logging.info("Saving final model...")
                    # 1. Save PyTorch format
                    torch.save(model.state_dict(), 'final_model.pt')
                    # 2. Save safetensors format
                    safetensors.save_model(model, 'final_model.safetensors')
                    
                    # Don't automatically overwrite best model - keep the actual best one
                    logging.info("Final model saved as final_model.pt/.safetensors (best model preserved)")
                    
                    break
                elif patience_counter > 0 and epoch + 1 < min_epochs_before_stopping:
                    logging.info(f"No improvement for {patience_counter} epochs, but continuing until minimum {min_epochs_before_stopping} epochs completed")

    logging.info("Final Structural Plasticity Stats:")
    for key, value in sp_controller.stats.items():
        logging.info(f"{key}: {value[-1] if value else 'N/A'}")
    logging.info("Training complete.")

if __name__ == "__main__":
    main()