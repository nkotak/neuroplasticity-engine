# transformer_model/adaptive_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple
import logging
import numpy as np

# Internal imports
from .adaptive_feedforward import AdaptiveFeedForward
from .multi_modal_embedder import MultiModalEmbedder
from .concept_embedding import ConceptEmbeddingLayer
from .domain_expert_controller import DomainExpertController

# Assuming these are in adaptive_network.py at the root
from adaptive_network import GlobalLayerRegistry, MetaController

# Import skip connection controller
from skip_connection_controller import SkipConnectionController

class AdaptiveTransformerLayer(nn.Module):
    """Single transformer layer with adaptive feedforward module."""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = AdaptiveFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # Ensure inputs are on the same device as the module weights
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class AdaptiveTransformer(nn.Module):
    """Transformer model with dynamic layer adaptation."""
    
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16, 
                 num_layers: int = 24, d_ff: int = 4096, max_seq_len: int = 512, 
                 dropout: float = 0.1, device=None, modality_config: Optional[Dict] = None):
        super().__init__()
        self.d_model = d_model
        self.registry = GlobalLayerRegistry()
        self.meta_controller = MetaController(max_layers=30)  # Allow some growth
        self.device = device or torch.device('cpu')
        self.max_seq_len = max_seq_len

        # pos_encoder is defined as nn.Parameter below (not a buffer)

        # Initialize weights on first forward pass
        self._initialized = False

        # Auxiliary loss from DomainExpertController (populated during forward pass)
        # None in text-only mode; tensor in multi-modal mode
        self.last_aux_loss: Optional[Tensor] = None
        
        # Handle multi-modal inputs if config provided
        self.is_multimodal = modality_config is not None
        if self.is_multimodal:
            # Multi-modal embedder implementation
            self.modality_embedder = MultiModalEmbedder(
                modality_config, fusion_dim=d_model, device=device
            )
            
            # Add concept embedding layer
            self.concept_embedding = ConceptEmbeddingLayer(
                input_dim=d_model, concept_dim=d_model, num_concepts=200
            )
            
            # Add domain experts
            self.domain_controller = DomainExpertController(
                input_dim=d_model, output_dim=d_model, num_experts=3
            )
        else:
            # Traditional text-only embedding
            self.embedding = nn.Embedding(vocab_size, d_model)
            with torch.no_grad():
                # Initialize embedding with small values to prevent early loss collapse
                self.embedding.weight.data.uniform_(-0.1, 0.1)
            
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize output layer weights
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Gating parameters for residual connections
        self.gating_params = nn.ParameterDict()
        
        # Initialize transformer layers
        for i in range(num_layers):
           layer = AdaptiveTransformerLayer(d_model, nhead, d_ff, dropout)
           self.add_module(f"layer_{i}", layer)  # Register as submodule
           layer_id = self.registry.add_layer(
               layer, 'transformer', d_model, d_model, position=i
           )
           self.gating_params[layer_id] = nn.Parameter(torch.tensor(0.0).to(device))
        
        # Initialize Skip Connection Controller
        self.skip_connection_controller = SkipConnectionController(
            correlation_threshold=0.7,    # Lowered from 0.999 to allow skip connections to form
            gradient_threshold=0.4,       # Lowered from 0.6 for earlier activation
            initial_gate_value=0.01,
            warmup_steps=50,              # Reduced from 100
            max_skip_connections=6,
            activation_window=100
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Move model to specified device if provided
        if device is not None:
            self.to(device)
    
    def _initialize_weights(self, device):
        """Device-aware weight initialization"""
        if not self._initialized:
            nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
            self.pos_encoder = self.pos_encoder.to(device)
            self._initialized = True
    
    def forward(self, x: Tensor, modality_inputs: Optional[Dict[str, Tensor]] = None, domain_id: int = 0) -> Tensor:
        """
        Forward pass with skip connections and proper device handling.
        
        Args:
            x: For text-only mode, token IDs [batch_size, seq_len]
                For multi-modal mode, can be None if modality_inputs is provided
            modality_inputs: Optional dict of modality inputs for multi-modal mode
                Example: {'text': text_tokens, 'image': images}
            domain_id: Domain identifier for concept updates
        """
        # Handle multi-modal inputs
        if self.is_multimodal and modality_inputs is not None:
            # Get device from any modality input
            for input_tensor in modality_inputs.values():
                device = input_tensor.device
                break
            
            # Initialize weights if needed
            if not self._initialized:
                self._initialize_weights(device)
            
            # Process through modality embedder
            embeddings = self.modality_embedder(modality_inputs)
            
            # Process through concept layer
            embeddings = self.concept_embedding(embeddings, domain_id)
            
            # Process through domain experts
            embeddings, aux_loss = self.domain_controller(embeddings)
            self.last_aux_loss = aux_loss

            # Apply dropout
            x = self.dropout(embeddings)

        else:
            # Standard text embedding path — no aux_loss
            self.last_aux_loss = None
            if x is None:
                raise ValueError("Input tensor cannot be None when not in multi-modal mode")
                
            device = x.device
            
            # Initialize weights on first pass
            if not self._initialized:
                self._initialize_weights(device)

            # Embedding and positional encoding
            seq_len = x.size(1)
            
            # Ensure positional encoder is on correct device
            pos_encoder_device = self.pos_encoder.device
            if pos_encoder_device != device:
                self.pos_encoder = self.pos_encoder.to(device)
                
            x = self.embedding(x) * (self.d_model ** 0.5) + self.pos_encoder[:, :seq_len, :]
            x = self.dropout(x)
        
        # Execute active layers with skip connections
        execution_path = self.registry.get_active_forward_path()
        
        # Track layer outputs for skip connections
        layer_outputs = {}
        
        for layer_id in execution_path:
            layer_info = self.registry.layers[layer_id]
            if not layer_info['active']:
                continue
                
            # Ensure gate parameter is on correct device
            gate_param = self.gating_params[layer_id]
            if gate_param.device != device:
                self.gating_params[layer_id] = gate_param.to(device)
                
            gate = torch.sigmoid(self.gating_params[layer_id])
            
            # Make sure module is on correct device
            if hasattr(layer_info['module'], 'to'):
                module_device = next(layer_info['module'].parameters()).device
                if module_device != device:
                    layer_info['module'] = layer_info['module'].to(device)
                    
            layer_output = layer_info['module'](x)
            
            # Convert warmup_factor to tensor on same device if it's not already
            warmup_factor = layer_info['warmup_factor']
            if not isinstance(warmup_factor, torch.Tensor):
                warmup_factor = torch.tensor(warmup_factor, device=device)
            elif warmup_factor.device != device:
                warmup_factor = warmup_factor.to(device)
                
            # Process skip connections targeting this layer
            skip_contribution = torch.zeros_like(x)
            for (source_id, target_id), connection in self.registry.skip_connections.items():
                if target_id == layer_id and source_id in layer_outputs and connection['active']:
                    # Move connection parameters to current device if needed
                    if connection['gate'].device != device:
                        connection['gate'] = connection['gate'].to(device)
                        connection['weight'] = connection['weight'].to(device)
                        connection['weight_mask'] = connection['weight_mask'].to(device)
                        connection['warmup_factor'] = connection['warmup_factor'].to(device)
                    
                    # Get skip connection parameters
                    skip_gate = torch.sigmoid(connection['gate'])
                    skip_weight = connection['weight'] * connection['weight_mask']
                    warmup = connection['warmup_factor']
                    
                    # Apply skip connection
                    source_output = layer_outputs[source_id]
                    skip_signal = F.linear(source_output, skip_weight)
                    skip_contribution = skip_contribution + skip_gate * skip_signal * warmup
                    
                    # Update usage metrics
                    connection['metrics']['usage'] = 0.9 * connection['metrics'].get('usage', 0.0) + 0.1
            
            # Now all tensors should be on the same device
            # Check for NaN values to prevent propagation
            if torch.isnan(layer_output).any():
                logging.warning(f"NaN values detected in layer output for {layer_id}")
                layer_output = torch.where(torch.isnan(layer_output),
                                         torch.zeros_like(layer_output),
                                         layer_output)
            
            # Add layer output and skip contribution
            x = x + gate * layer_output * warmup_factor + skip_contribution
            
            # Store output for potential skip connections
            layer_outputs[layer_id] = x
            
            # Update usage metrics
            layer_info['metrics']['usage_frequency'] = (
                0.9 * layer_info['metrics'].get('usage_frequency', 0.0) + 0.1
            )
            
            # Track intermediate activations for skip connection analysis
            if self.training and hasattr(self, 'skip_connection_controller'):
                device_x = x.detach()
                self.skip_connection_controller.update_activations(layer_id, device_x)
        
        # Output projection
        x = self.output_layer(x)
        return x
    
    def update_metrics(self, gradients: Dict[str, float], activations: Dict[str, float]) -> None:
        """Update layer metrics for meta-learning."""
        for lid in self.registry.layers:
            if lid in gradients:
                self.registry.update_metrics(lid, {'gradient_mag': gradients[lid]})
            if lid in activations:
                self.registry.update_metrics(lid, {'activation_sparsity': activations[lid]})
    
    def increment_warmup(self):
        """Gradually activate new layers with device awareness"""
        device = next(self.parameters()).device  # Get device from model parameters
        
        for lid in self.registry.layers:
            warmup_factor = self.registry.layers[lid]['warmup_factor']
            
            # Convert scalar to tensor if needed
            if not isinstance(warmup_factor, torch.Tensor):
                warmup_factor = torch.tensor(warmup_factor, device=device)
            # Ensure tensor is on correct device  
            elif warmup_factor.device != device:
                warmup_factor = warmup_factor.to(device)
                
            # Calculate new value
            new_warmup = torch.minimum(
                torch.tensor(1.0, device=device),
                warmup_factor + torch.tensor(1.0/self.meta_controller.warmup_steps, device=device)
            )
            
            # Store back in registry
            self.registry.layers[lid]['warmup_factor'] = new_warmup
    
    def meta_step(self, val_loss: float) -> None:
        """Perform meta-learning step to add/remove layers and update skip connections."""
        add_layer, remove_layer = self.meta_controller.analyze_performance(self.registry, val_loss)
        
        if add_layer:
            grad_mags = [
                (i, self.registry.layers[lid]['metrics']['gradient_mag'])
                for i, lid in enumerate(self.registry.order)
            ]
            position = max(grad_mags, key=lambda x: x[1])[0] + 1
            new_layer = AdaptiveTransformerLayer(self.d_model, 16, 4096)
            layer_id = self.registry.add_layer(
                new_layer, 'transformer', self.d_model, self.d_model,
                position=min(position, len(self.registry.order))
            )
            self.gating_params[layer_id] = nn.Parameter(torch.tensor(0.0))
        
        if remove_layer:
            redundancy_scores = [
                (lid, l['metrics']['activation_sparsity'] * (1 - l['metrics']['usage_frequency']))
                for lid, l in self.registry.layers.items()
            ]
            if redundancy_scores:
                redundant_id = max(redundancy_scores, key=lambda x: x[1])[0]
                if redundancy_scores[0][1] > self.meta_controller.redundancy_threshold:
                    self.registry.remove_layer(redundant_id)
                    if redundant_id in self.gating_params:
                        del self.gating_params[redundant_id]
        
        # Update skip connections if controller exists
        if hasattr(self, 'skip_connection_controller'):
            # Collect gradient information for skip connection decisions
            gradients = {}
            for lid in self.registry.layers:
                if 'gradient_mag' in self.registry.layers[lid]['metrics']:
                    gradients[lid] = self.registry.layers[lid]['metrics']['gradient_mag']
            
            # Update skip connections
            skip_connections_added = self.skip_connection_controller.step(
                self.registry, self, gradients
            )
            if skip_connections_added > 0:
                logging.info(f"Added {skip_connections_added} new skip connections during meta step")
        
        # Update domain experts for multi-modal support
        if self.is_multimodal and self.training:
            # Expert spawn/remove logic
            if val_loss < 3.0:  # Only after some training
                if np.random.random() < 0.1:  # 10% chance to try
                    if self.domain_controller.spawn_new_expert():
                        logging.info(f"Spawned new domain expert, now have {self.domain_controller.num_experts}")
                    
            if val_loss < 2.0:  # Only after more training
                if np.random.random() < 0.05:  # 5% chance to try
                    if self.domain_controller.remove_underutilized_expert():
                        logging.info(f"Removed underutilized expert, now have {self.domain_controller.num_experts}")
                    
            # Concept pruning logic
            if np.random.random() < 0.2:  # 20% chance each meta-step
                num_pruned = self.concept_embedding.prune_unused_concepts(threshold=0.01)
                if num_pruned > 0:
                    logging.info(f"Pruned {num_pruned} unused concepts")