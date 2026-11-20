# transformer_model/adaptive_feedforward.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import logging

# Assuming MovingStats is in adaptive_network.py at the root
from adaptive_network import MovingStats

class AdaptiveFeedForward(nn.Module):
    """Adaptive feedforward module for transformer layers with dynamic plasticity."""
    
    def __init__(self, d_model: int, d_ff: int, 
                 activation=F.gelu, 
                 split_noise_scale: float = 0.1, 
                 merge_similarity_threshold: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers mimicking standard FFN
        self.input_layer = nn.Linear(d_model, d_ff)  # First layer: d_model -> d_ff
        self.output_layer = nn.Linear(d_ff, d_model) # Second layer: d_ff -> d_model
        
        # Weight masks for sparsity
        self.input_layer_weight_mask = nn.Parameter(torch.ones_like(self.input_layer.weight), requires_grad=False)
        self.output_layer_weight_mask = nn.Parameter(torch.ones_like(self.output_layer.weight), requires_grad=False)
        
        # Activation function
        self.activation = activation
        
        # Activation tracking
        self.input_activation_stats = MovingStats()
        self.intermediate_activation_stats = MovingStats()
        self.output_activation_stats = MovingStats()
        
        # Structural adaptation parameters
        self.split_noise_scale = split_noise_scale
        self.merge_similarity_threshold = merge_similarity_threshold
        self.neuron_age = torch.zeros(d_ff)  # Tracks age of intermediate neurons
        
        # Layer normalization for stability
        self.intermediate_norm = nn.LayerNorm(d_ff)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with dynamic weight masking and device handling."""
        # Ensure input and weights are on same device
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)
            
        if self.training:
            self.input_activation_stats.update(x.detach().mean().item(), x.detach())
        
        # Input layer with mask
        x = F.linear(x, self.input_layer.weight * self.input_layer_weight_mask, self.input_layer.bias)
        x = self.intermediate_norm(x)
        x = self.activation(x)
        
        if self.training:
            self.intermediate_activation_stats.update(x.detach().mean().item(), x.detach())
            self.neuron_age += 1  # Increment age of intermediate neurons
        
        # Output layer with mask
        x = F.linear(x, self.output_layer.weight * self.output_layer_weight_mask, self.output_layer.bias)
        
        if self.training:
            self.output_activation_stats.update(x.detach().mean().item(), x.detach())
        
        return x
    
    def split_neuron(self, layer_name: str, neuron_idx: int) -> None:
        """Split an intermediate neuron, updating both layers."""
        if layer_name != 'input':
            self.logger.warning(f"Only 'input' layer splitting supported, got {layer_name}")
            return
        
        with torch.no_grad():
            new_d_ff = self.d_ff + 1
            
            # Update input_layer (d_model -> new_d_ff)
            new_input_layer = nn.Linear(self.d_model, new_d_ff)
            new_input_layer.weight.data[:self.d_ff] = self.input_layer.weight.data
            new_input_layer.bias.data[:self.d_ff] = self.input_layer.bias.data
            # New neuron weights: copy with noise
            new_input_layer.weight.data[new_d_ff - 1] = (
                self.input_layer.weight.data[neuron_idx] + 
                torch.randn_like(self.input_layer.weight.data[neuron_idx]) * self.split_noise_scale
            )
            new_input_layer.bias.data[new_d_ff - 1] = (
                self.input_layer.bias.data[neuron_idx] + 
                torch.randn_like(self.input_layer.bias.data[neuron_idx]) * self.split_noise_scale
            )
            
            # Update input_layer mask
            new_input_mask = torch.ones_like(new_input_layer.weight)
            new_input_mask[:self.d_ff] = self.input_layer_weight_mask
            new_input_mask[new_d_ff - 1] = 1.0
            
            # Update output_layer (new_d_ff -> d_model)
            new_output_layer = nn.Linear(new_d_ff, self.d_model)
            new_output_layer.weight.data[:, :self.d_ff] = self.output_layer.weight.data
            new_output_layer.bias.data = self.output_layer.bias.data
            # New input feature weights: copy with noise
            new_output_layer.weight.data[:, new_d_ff - 1] = (
                self.output_layer.weight.data[:, neuron_idx] + 
                torch.randn_like(self.output_layer.weight.data[:, neuron_idx]) * self.split_noise_scale
            )
            
            # Update output_layer mask
            new_output_mask = torch.ones_like(new_output_layer.weight)
            new_output_mask[:, :self.d_ff] = self.output_layer_weight_mask
            new_output_mask[:, new_d_ff - 1] = 1.0
            
            # Assign updated layers and masks
            self.input_layer = new_input_layer
            self.output_layer = new_output_layer
            self.input_layer_weight_mask = nn.Parameter(new_input_mask, requires_grad=False)
            self.output_layer_weight_mask = nn.Parameter(new_output_mask, requires_grad=False)
            
            # Update dimensions and related attributes
            self.d_ff = new_d_ff
            self.intermediate_norm = nn.LayerNorm(new_d_ff)
            self.neuron_age = torch.cat([self.neuron_age, torch.tensor([0.0])])
            
            self.logger.info(f"Split neuron {neuron_idx} in input layer, new d_ff={self.d_ff}")
    
    def merge_neurons(self, layer_name: str, neuron_indices: List[int]) -> None:
        """Merge intermediate neurons, updating both layers."""
        if layer_name != 'input':
            self.logger.warning(f"Only 'input' layer merging supported, got {layer_name}")
            return
        
        with torch.no_grad():
            num_to_merge = len(neuron_indices)
            if num_to_merge < 2:
                return
            new_d_ff = self.d_ff - num_to_merge + 1
            
            # Update input_layer
            new_input_layer = nn.Linear(self.d_model, new_d_ff)
            valid_indices = [i for i in range(self.d_ff) if i not in neuron_indices]
            new_input_layer.weight.data[:-1] = self.input_layer.weight.data[valid_indices]
            new_input_layer.bias.data[:-1] = self.input_layer.bias.data[valid_indices]
            # Merged neuron weights
            new_input_layer.weight.data[-1] = self.input_layer.weight.data[neuron_indices].mean(dim=0)
            new_input_layer.bias.data[-1] = self.input_layer.bias.data[neuron_indices].mean()
            
            # Update input_layer mask
            new_input_mask = torch.ones_like(new_input_layer.weight)
            new_input_mask[:-1] = self.input_layer_weight_mask[valid_indices]
            new_input_mask[-1] = self.input_layer_weight_mask[neuron_indices].max(dim=0)[0]
            
            # Update output_layer
            new_output_layer = nn.Linear(new_d_ff, self.d_model)
            new_output_layer.weight.data[:, :-1] = self.output_layer.weight.data[:, valid_indices]
            new_output_layer.bias.data = self.output_layer.bias.data
            new_output_layer.weight.data[:, -1] = self.output_layer.weight.data[:, neuron_indices].mean(dim=1)
            
            # Update output_layer mask
            new_output_mask = torch.ones_like(new_output_layer.weight)
            new_output_mask[:, :-1] = self.output_layer_weight_mask[:, valid_indices]
            new_output_mask[:, -1] = self.output_layer_weight_mask[:, neuron_indices].max(dim=1)[0]
            
            # Assign updated layers and masks
            self.input_layer = new_input_layer
            self.output_layer = new_output_layer
            self.input_layer_weight_mask = nn.Parameter(new_input_mask, requires_grad=False)
            self.output_layer_weight_mask = nn.Parameter(new_output_mask, requires_grad=False)
            
            # Update dimensions and related attributes
            self.d_ff = new_d_ff
            self.intermediate_norm = nn.LayerNorm(new_d_ff)
            self.neuron_age = torch.tensor([self.neuron_age[i] for i in range(self.d_ff) if i not in neuron_indices] + [0.0])
            
            self.logger.info(f"Merged neurons {neuron_indices} in input layer, new d_ff={self.d_ff}")