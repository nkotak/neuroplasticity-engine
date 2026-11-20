import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
import logging
from collections import deque, defaultdict
import numpy as np

class SkipConnectionController:
    """
    Controller that analyzes layer activations and manages skip connections.
    This controller identifies potential skip connections between non-adjacent
    layers and manages their lifecycle.
    """
    
    def __init__(self, 
                correlation_threshold: float = 0.6,
                gradient_threshold: float = 0.4,
                initial_gate_value: float = 0.01,
                warmup_steps: int = 100,
                max_skip_connections: int = 10,
                activation_window: int = 100):
        self.correlation_threshold = correlation_threshold
        self.gradient_threshold = gradient_threshold
        self.initial_gate_value = initial_gate_value
        self.warmup_steps = warmup_steps
        self.max_skip_connections = max_skip_connections
        self.activation_window = activation_window
        
        # Activity tracking
        self.layer_activations = {}  # Layer ID -> deque of activations
        self.layer_correlations = {}  # (source_id, target_id) -> correlation value
        
        # Statistics tracking
        self.stats = defaultdict(list)
        self.epoch_counter = 0
        
        self.logger = logging.getLogger('SkipConnectionController')
        
    def update_activations(self, layer_id: str, activations: torch.Tensor):
        """
        Track layer activations for correlation analysis
        
        Args:
            layer_id: Unique identifier for the layer
            activations: Tensor of activations from the layer (batch_size, features)
        """
        if layer_id not in self.layer_activations:
            self.layer_activations[layer_id] = deque(maxlen=self.activation_window)
            
        # Store activation statistics (mean across batch)
        try:
            # Use mean across batch dimension to get per-neuron activation pattern
            if activations.dim() >= 2:
                activation_stats = activations.detach().mean(dim=0).cpu()
                self.layer_activations[layer_id].append(activation_stats)
            else:
                self.logger.warning(f"Activations for layer {layer_id} have unexpected dimensions: {activations.shape}")
        except Exception as e:
            self.logger.warning(f"Error updating activations for layer {layer_id}: {str(e)}")
    
    def analyze_layer_correlations(self, registry) -> List[Tuple[str, str]]:
        """
        Analyze correlations between layers and find candidates for skip connections
        
        Args:
            registry: The GlobalLayerRegistry containing the layers
            
        Returns:
            List of (source_id, target_id) tuples representing candidate skip connections
        """
        # Only analyze if we have enough activation data
        candidates = []
        layer_ids = registry.order
        
        # Check correlations between all pairs of non-adjacent layers
        for i, source_id in enumerate(layer_ids):
            if source_id not in self.layer_activations or len(self.layer_activations[source_id]) < 10:
                continue
                
            source_activations = torch.stack(list(self.layer_activations[source_id]))
            
            # Look for potential target layers (skipping adjacent layers)
            for j, target_id in enumerate(layer_ids):
                # Skip if target doesn't have activations, is adjacent, or connection already exists
                if (target_id not in self.layer_activations or 
                    abs(i - j) <= 1 or  # Skip adjacent layers
                    (source_id, target_id) in registry.skip_connections):  # Skip existing
                    continue
                
                # Skip if we don't have enough data
                if len(self.layer_activations[target_id]) < 10:
                    continue
                
                target_activations = torch.stack(list(self.layer_activations[target_id]))
                
                # Ensure we have matching dimensions
                min_len = min(len(source_activations), len(target_activations))
                if min_len < 10:  # Need enough samples for correlation
                    continue
                    
                source_acts = source_activations[-min_len:]
                target_acts = target_activations[-min_len:]
                
                # Compute correlation
                try:
                    # Flatten if needed
                    source_flattened = source_acts.view(min_len, -1)
                    target_flattened = target_acts.view(min_len, -1)
                    
                    # Normalize
                    source_norm = F.normalize(source_flattened, p=2, dim=1)
                    target_norm = F.normalize(target_flattened, p=2, dim=1)
                    
                    # Compute correlation using cosine similarity
                    correlation = torch.mm(source_norm, target_norm.t()).diag().mean().item()
                    
                    # Store correlation
                    self.layer_correlations[(source_id, target_id)] = correlation
                    
                    # If correlation exceeds threshold, add as candidate
                    if correlation > self.correlation_threshold:
                        candidates.append((source_id, target_id))
                        self.logger.info(f"Found skip connection candidate: {source_id} -> {target_id} (correlation: {correlation:.4f})")
                except Exception as e:
                    self.logger.warning(f"Could not compute correlation for {source_id} -> {target_id}: {str(e)}")
                
        return candidates
    
    def step(self, registry, network, gradients: Dict[str, float] = None) -> int:
        """
        Analyze network and add/remove skip connections as needed
        
        Args:
            registry: The GlobalLayerRegistry
            network: The neural network
            gradients: Dictionary mapping layer IDs to gradient magnitudes
            
        Returns:
            Number of connections added
        """
        self.epoch_counter += 1
        
        # Don't add skip connections too early in training
        # Wait until at least 20 epochs to let the model learn first
        if self.epoch_counter < 20:
            return 0
        
        # Analyze correlation between layers
        candidates = self.analyze_layer_correlations(registry)
        
        # Track how many connections we add
        added_connections = 0
        
        # Skip if we already have maximum connections
        current_connections = len(registry.skip_connections)
        if current_connections >= self.max_skip_connections:
            # Consider removing underperforming skip connections instead
            self._prune_skip_connections(registry, network)
            return 0
        
        # Sort candidates by correlation
        candidates.sort(key=lambda x: self.layer_correlations.get((x[0], x[1]), 0.0), reverse=True)
        
        # Add top candidates as skip connections
        device = next(network.parameters()).device
        for source_id, target_id in candidates[:self.max_skip_connections - current_connections]:
            # Get dimensions for compatible skip connection
            source_dim = registry.layers[source_id]['output_dim']
            target_dim = registry.layers[target_id]['input_dim']
            
            # For now we only support skip connections between layers with matching dimensions
            if source_dim != target_dim:
                self.logger.warning(f"Dimension mismatch for skip connection: {source_id}({source_dim}) -> {target_id}({target_dim})")
                continue
                
            # Add skip connection to registry
            registry.add_skip_connection(source_id, target_id, (target_dim, source_dim), device)
            added_connections += 1
            
            # Limit to one new connection per step to avoid instability
            if added_connections > 0:
                break
                
        # Update metrics for existing skip connections
        self._update_skip_connection_metrics(registry, network, gradients)
        
        # Update stats
        self.stats['connections_added'].append(added_connections)
        self.stats['total_connections'].append(len(registry.skip_connections))
        
        return added_connections
        
    def _update_skip_connection_metrics(self, registry, network, gradients: Dict[str, float] = None):
        """
        Update performance metrics for skip connections
        
        Args:
            registry: The GlobalLayerRegistry
            network: The neural network
            gradients: Dictionary mapping layer IDs to gradient magnitudes
        """
        for (source_id, target_id), connection in registry.skip_connections.items():
            # Update correlation metric
            correlation = self.layer_correlations.get((source_id, target_id), 0.0)
            connection['metrics']['correlation'] = correlation
            
            # Update gradient flow metric (if available)
            if gradients is not None and source_id in gradients and target_id in gradients:
                gradient_flow = min(gradients[source_id], gradients[target_id])
                connection['metrics']['gradient_flow'] = gradient_flow
            
            # Increment warmup factor if not already at 1.0
            if connection['warmup_factor'].item() < 1.0:
                connection['warmup_factor'] = torch.min(
                    torch.tensor(1.0, device=connection['warmup_factor'].device),
                    connection['warmup_factor'] + 1.0/self.warmup_steps
                )
        
    def _prune_skip_connections(self, registry, network):
        """
        Remove skip connections that aren't providing benefit
        
        Args:
            registry: The GlobalLayerRegistry
            network: The neural network
        """
        to_remove = []
        
        for (source_id, target_id), connection in registry.skip_connections.items():
            # Skip if not fully warmed up
            if connection['warmup_factor'].item() < 0.9:
                continue
                
            # Check correlation and usage metrics
            correlation = connection['metrics']['correlation']
            gate_value = torch.sigmoid(connection['gate']).item()
            
            # If correlation is low and gate value is low, connection isn't useful
            if correlation < self.correlation_threshold/2 and gate_value < 0.2:
                to_remove.append((source_id, target_id))
                
        # Remove underperforming connections
        for source_id, target_id in to_remove:
            registry.remove_skip_connection(source_id, target_id)
            self.logger.info(f"Removed underperforming skip connection: {source_id} -> {target_id}")
            
        # Update stats
        self.stats['connections_removed'].append(len(to_remove))