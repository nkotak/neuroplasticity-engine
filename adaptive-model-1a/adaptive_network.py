import numpy as np
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from sklearn.cluster import DBSCAN
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from collections import deque, defaultdict, OrderedDict
import logging

# Import AdaptiveFeedForward for transformer plasticity support
try:
    from transformer_model.adaptive_feedforward import AdaptiveFeedForward
except ImportError:
    AdaptiveFeedForward = None

TARGET_SPARSITY = 0.0025
ADAPT_REWARD_SCALE = 1.0

class EnhancedAdaptiveNode(nn.Module):
    """Enhanced implementation supporting dynamic structural changes"""
    
    def __init__(self, input_dim: int, 
                 hidden_dim: int = 64,
                 activation_history_size: int = 1000,
                 split_noise_scale: float = 0.1,
                 merge_similarity_threshold: float = 0.8):
        super().__init__()
        
        # Dynamic size tracking
        self._current_input_dim = input_dim
        self._current_hidden_dim = hidden_dim
        
        # Core neural components
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Structural plasticity components
        self.input_layer_weight_mask = nn.Parameter(torch.ones_like(self.input_layer.weight.data), requires_grad=False)
        self.hidden_layer_weight_mask = nn.Parameter(torch.ones_like(self.hidden_layer.weight.data), requires_grad=False)
        self.output_layer_weight_mask = nn.Parameter(torch.ones_like(self.output_layer.weight.data), requires_grad=False)
        
        # Activation tracking
        self.input_activation_stats = MovingStats()
        self.hidden_activation_stats = MovingStats()
        self.output_activation_stats = MovingStats()
        self.recent_raw_input = None  # Initialize recent_raw_input
        
        # Structural adaptation parameters
        self.split_noise_scale = split_noise_scale
        self.merge_similarity_threshold = merge_similarity_threshold
        self.neuron_age = torch.zeros(hidden_dim)  # Track neuron utilization
        self.activation_history = torch.zeros(activation_history_size)  # Simplified for this example
        self.recent_inputs = []  # Simplified storage
        
        # Adaptive parameters
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1))
        self.plasticity_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Initialize LayerNorm for stability
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for plasticity updates
        if self.training:
            # Store raw input and clear previous activation history
            self.recent_raw_input = x.detach().clone()
            self.input_activation_stats.recent_activations = None
            self.hidden_activation_stats.recent_activations = None
            self.output_activation_stats.recent_activations = None
            self.neuron_age += 1  # Track neuron utilization
        
        # Dynamic forward pass using weight masks
        x = self._dynamic_linear(x, self.input_layer, self.input_layer_weight_mask)
        x = self.input_norm(x)
        x = F.relu(x)
        self._update_activation_stats(x, self.input_activation_stats)
        
        x = self._dynamic_linear(x, self.hidden_layer, self.hidden_layer_weight_mask)
        x = self.hidden_norm(x)
        x = F.relu(x)
        self._update_activation_stats(x, self.hidden_activation_stats)
        
        x = self._dynamic_linear(x, self.output_layer, self.output_layer_weight_mask)
        self._update_activation_stats(x, self.output_activation_stats)
        
        return x
    
    def _dynamic_linear(self, x: torch.Tensor, layer: nn.Linear, mask: torch.Tensor) -> torch.Tensor:
        return F.linear(x, layer.weight * mask, layer.bias)
    
    def _update_activation_stats(self, activations: torch.Tensor, stats: 'MovingStats'):
        if self.training:
            stats.update(activations.detach().mean().item(), activations.detach(), activations.size(0))
            stats.update_outputs(activations.detach(), activations.size(0))
    
    def split_neuron(self, layer_name: str, neuron_idx: int):
        """Split a neuron in the specified layer"""
        layer = getattr(self, f"{layer_name}_layer")
        mask = getattr(self, f"{layer_name}_layer_weight_mask")
        stats = getattr(self, f"{layer_name}_activation_stats")
        
        # Create new layer dimensions
        out_features, in_features = layer.weight.shape
        new_out = out_features + 1
        
        # Initialize new layer
        new_layer = nn.Linear(in_features, new_out)
        new_mask = torch.ones_like(new_layer.weight.data)
        
        # Copy existing weights and masks
        new_layer.weight.data[:out_features] = layer.weight.data
        new_layer.bias.data[:out_features] = layer.bias.data if layer.bias is not None else 0
        new_mask[:out_features] = mask.data
        
        # Add split neuron with noise
        new_layer.weight.data[out_features] = layer.weight.data[neuron_idx] + \
            torch.randn_like(layer.weight.data[neuron_idx]) * self.split_noise_scale
        if layer.bias is not None:
            new_layer.bias.data[out_features] = layer.bias.data[neuron_idx] + \
                torch.randn_like(layer.bias.data[neuron_idx]) * self.split_noise_scale
        new_mask[out_features] = 1.0
        
        # Update layer and mask
        setattr(self, f"{layer_name}_layer", new_layer)
        setattr(self, f"{layer_name}_layer_weight_mask", nn.Parameter(new_mask, requires_grad=False))
        
        # Update hidden size if splitting hidden layer
        if layer_name == 'hidden':
            self._current_hidden_dim += 1
            self.hidden_norm = nn.LayerNorm(self._current_hidden_dim)
        
        # Update next layer's input dimension
        self._update_downstream_layer(layer_name)
        
        # Reset age for new neuron
        self.neuron_age = torch.cat([self.neuron_age, torch.tensor([0.0])])
    
    def merge_neurons(self, layer_name: str, neuron_indices: List[int]):
        """Merge multiple neurons in the specified layer"""
        layer = getattr(self, f"{layer_name}_layer")
        mask = getattr(self, f"{layer_name}_layer_weight_mask")
        
        # Create new layer dimensions
        out_features, in_features = layer.weight.shape
        new_out = out_features - len(neuron_indices) + 1
        
        # Initialize new layer
        new_layer = nn.Linear(in_features, new_out)
        new_mask = torch.ones_like(new_layer.weight.data)
        
        # Create merged weights
        valid_indices = [i for i in range(out_features) if i not in neuron_indices]
        new_layer.weight.data[:-1] = layer.weight.data[valid_indices]
        new_layer.weight.data[-1] = layer.weight.data[neuron_indices].mean(dim=0)
        
        if layer.bias is not None:
            new_layer.bias.data[:-1] = layer.bias.data[valid_indices]
            new_layer.bias.data[-1] = layer.bias.data[neuron_indices].mean()
        
        new_mask[:-1] = mask.data[valid_indices]
        new_mask[-1] = mask.data[neuron_indices].max(dim=0)[0]
        
        # Update layer and mask
        setattr(self, f"{layer_name}_layer", new_layer)
        setattr(self, f"{layer_name}_layer_weight_mask", nn.Parameter(new_mask, requires_grad=False))
        
        # Update hidden size if merging in hidden layer
        if layer_name == 'hidden':
            self._current_hidden_dim = new_out
            self.hidden_norm = nn.LayerNorm(self._current_hidden_dim)
        
        # Update downstream layer
        self._update_downstream_layer(layer_name)
        
        # Remove merged neuron ages
        self.neuron_age = torch.tensor([age for i, age in enumerate(self.neuron_age.tolist()) 
                                      if i not in neuron_indices] + [0.0])
    
    def _update_downstream_layer(self, layer_name: str):
        """Update subsequent layer dimensions after structural change"""
        if layer_name == 'input':
            new_size = self.input_layer.weight.shape[0]
            self._update_layer_dimensions('hidden', 'in_features', new_size)
        elif layer_name == 'hidden':
            new_size = self.hidden_layer.weight.shape[0]
            self._update_layer_dimensions('output', 'in_features', new_size)
    
    def _update_layer_dimensions(self, layer_name: str, dim_type: str, new_size: int):
        """Update layer dimensions while preserving existing functionality"""
        layer = getattr(self, f"{layer_name}_layer")
        original_size = getattr(layer, dim_type)
        
        if original_size == new_size:
            return
        
        # Create new layer
        if dim_type == 'in_features':
            new_layer = nn.Linear(new_size, layer.out_features)
        else:
            new_layer = nn.Linear(layer.in_features, new_size)
        
        # Initialize weights
        with torch.no_grad():
            if dim_type == 'in_features':
                # Copy existing weights for overlapping dimensions
                min_dim = min(original_size, new_size)
                new_layer.weight.data[:, :min_dim] = layer.weight.data[:, :min_dim]
                # Initialize new dimensions with small noise
                if new_size > original_size:
                    new_layer.weight.data[:, original_size:] = torch.randn_like(new_layer.weight.data[:, original_size:]) * 0.01
            else:
                min_dim = min(original_size, new_size)
                new_layer.weight.data[:min_dim] = layer.weight.data[:min_dim]
                if new_size > original_size:
                    new_layer.weight.data[original_size:] = torch.randn_like(new_layer.weight.data[original_size:]) * 0.01
        
        # Update mask
        mask = getattr(self, f"{layer_name}_layer_weight_mask")
        new_mask = torch.ones_like(new_layer.weight.data)
        if dim_type == 'in_features':
            new_mask[:, :mask.shape[1]] = mask[:, :min_dim]
        else:
            new_mask[:mask.shape[0]] = mask[:min_dim]
        
        setattr(self, f"{layer_name}_layer", new_layer)
        setattr(self, f"{layer_name}_layer_weight_mask", nn.Parameter(new_mask, requires_grad=False))

class GlobalLayerRegistry:
    """Enhanced registry maintaining explicit layer ordering and connectivity with device support"""
    
    def __init__(self):
        self.layers = OrderedDict()  # {layer_id: layer_metadata}
        self.connections = defaultdict(set)  # Adjacency list
        self.skip_connections = {}  # {(source_id, target_id): connection_info}
        self.performance_stats = defaultdict(deque)
        self.order = []  # Explicit ordering of layer IDs
        self.logger = logging.getLogger('LayerRegistry')
        self.device = None  # Track current device

    def add_layer(self, layer: nn.Module, layer_type: str, 
                 input_dim: int, output_dim: int,
                 parent_ids: List[str] = None,
                 position: Optional[int] = None) -> str:
        """Register new layer with ordering control"""
        layer_id = str(uuid.uuid4())
        
        # If we have a current device, move layer to it
        if self.device is not None:
            layer = layer.to(self.device)
        
        # Store warmup factor as tensor if device is known
        warmup_factor = 0.0
        if self.device is not None:
            warmup_factor = torch.tensor(0.0, device=self.device)
        
        self.layers[layer_id] = {
            'module': layer,
            'type': layer_type,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'active': True,
            'warmup_factor': warmup_factor,
            'metrics': {
                'gradient_mag': 0.0,
                'activation_sparsity': 1.0,
                'usage_frequency': 0.0
            }
        }
        
        # Update ordering
        if position is not None and 0 <= position <= len(self.order):
            self.order.insert(position, layer_id)
        else:
            self.order.append(layer_id)
        
        # Update connections
        if parent_ids:
            for pid in parent_ids:
                if pid in self.layers:  # Ensure parent exists
                    self.connections[pid].add(layer_id)
        
        self.logger.info(f"Added layer {layer_id} ({layer_type}) at position {position if position is not None else 'end'}")
        return layer_id

    def to(self, device):
        """Move all layers to specified device"""
        self.device = device
        for layer_id, layer_info in self.layers.items():
            if hasattr(layer_info['module'], 'to'):
                layer_info['module'] = layer_info['module'].to(device)
                
            # Convert warmup_factor to tensor on the correct device
            warmup_factor = layer_info['warmup_factor']
            if not isinstance(warmup_factor, torch.Tensor):
                layer_info['warmup_factor'] = torch.tensor(warmup_factor, device=device)
            elif warmup_factor.device != device:
                layer_info['warmup_factor'] = warmup_factor.to(device)
                
        return self
    
    def increment_warmup(self, warmup_steps=100, device=None):
        """Update warmup factors with device awareness"""
        for layer_id in self.layers:
            if isinstance(self.layers[layer_id]['warmup_factor'], torch.Tensor):
                current_device = self.layers[layer_id]['warmup_factor'].device
                current_value = self.layers[layer_id]['warmup_factor'].item()
                new_value = min(1.0, current_value + 1/warmup_steps)
                self.layers[layer_id]['warmup_factor'] = torch.tensor(new_value, device=current_device)
            else:
                current_value = self.layers[layer_id]['warmup_factor']
                new_value = min(1.0, current_value + 1/warmup_steps)
                
                # Convert to tensor if device is provided
                if device is not None:
                    self.layers[layer_id]['warmup_factor'] = torch.tensor(new_value, device=device)
                else:
                    self.layers[layer_id]['warmup_factor'] = new_value

    def remove_layer(self, layer_id: str):
        """Safely remove layer and update ordering"""
        if layer_id not in self.layers:
            return

        # Re-route connections
        parents = [pid for pid, children in self.connections.items()
                  if layer_id in children]
        children = list(self.connections[layer_id])
        
        # Connect parents to children
        for p in parents:
            self.connections[p].remove(layer_id)
            self.connections[p].update(children)
            
        # Update ordering
        if layer_id in self.order:
            self.order.remove(layer_id)
            
        del self.layers[layer_id]
        del self.connections[layer_id]
        
        self.logger.info(f"Removed layer {layer_id}")

    def get_layer_position(self, layer_id: str) -> Optional[int]:
        """Get position of layer in current ordering"""
        try:
            return self.order.index(layer_id)
        except ValueError:
            return None

    def get_active_forward_path(self) -> List[str]:
        """Get execution path using explicit ordering"""
        return [lid for lid in self.order 
                if lid in self.layers and self.layers[lid]['active']]
    def add_skip_connection(self, source_id: str, target_id: str,
                           weight_shape: tuple, device=None) -> str:
        """
        Register a new skip connection between two layers
        
        Args:
            source_id: ID of the source layer
            target_id: ID of the target layer
            weight_shape: Shape of the weight tensor (target_dim, source_dim)
            device: Device for the weight tensors
            
        Returns:
            connection_id: Unique identifier for the skip connection
        """
        connection_id = f"skip_{source_id}_{target_id}"
        
        # Skip if connection already exists
        if (source_id, target_id) in self.skip_connections:
            return self.skip_connections[(source_id, target_id)]['id']
        
        # Check if both layers exist
        if source_id not in self.layers or target_id not in self.layers:
            self.logger.warning(f"Cannot create skip connection: one or both layers not found - {source_id}, {target_id}")
            return None
        
        # Initialize weight and gate parameters
        if device is None and self.device is not None:
            device = self.device
            
        # Create connection info
        self.skip_connections[(source_id, target_id)] = {
            'id': connection_id,
            'source_id': source_id,
            'target_id': target_id,
            'weight': torch.zeros(weight_shape, device=device).normal_(0, 0.02),
            'weight_mask': torch.ones(weight_shape, device=device),
            'gate': torch.tensor(0.01, device=device),  # Start with low influence
            'warmup_factor': torch.tensor(0.0, device=device),
            'active': True,
            'metrics': {
                'correlation': 0.0,
                'gradient_flow': 0.0,
                'usage': 0.0
            }
        }
        
        self.logger.info(f"Added skip connection from {source_id} to {target_id}")
        return connection_id
    
    def remove_skip_connection(self, source_id: str, target_id: str):
        """
        Remove a skip connection
        
        Args:
            source_id: ID of the source layer
            target_id: ID of the target layer
        """
        if (source_id, target_id) in self.skip_connections:
            del self.skip_connections[(source_id, target_id)]
            self.logger.info(f"Removed skip connection from {source_id} to {target_id}")
            
    def update_skip_connection_metrics(self, source_id: str, target_id: str, metrics: Dict[str, float]):
        """
        Update metrics for a specific skip connection
        
        Args:
            source_id: ID of the source layer
            target_id: ID of the target layer
            metrics: Dictionary of metric names and values to update
        """
        if (source_id, target_id) in self.skip_connections:
            self.skip_connections[(source_id, target_id)]['metrics'].update(metrics)
        else:
            self.logger.warning(f"Skip connection {source_id} -> {target_id} not found for metrics update")
    
    def update_metrics(self, layer_id: str, metrics: Dict[str, float]):
        """
        Update metrics for a specific layer in the registry.
        
        Args:
            layer_id (str): The ID of the layer to update.
            metrics (Dict[str, float]): A dictionary of metric names and values to update.
        """
        if layer_id in self.layers:
            self.layers[layer_id]['metrics'].update(metrics)
        else:
            self.logger.warning(f"Layer {layer_id} not found in registry for metrics update.")
            self.logger.warning(f"Layer {layer_id} not found in registry for metrics update.")

class EnhancedAdaptiveNetwork(nn.Module):
    """Network with dynamic layer architecture and structural plasticity"""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 num_initial_layers: int = 3):
        super().__init__()
        # Initialize registry and meta-controller.
        self.registry = GlobalLayerRegistry()
        self.meta_controller = MetaController()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Create ModuleList for nodes.
        self.nodes = nn.ModuleList()
        # Initialize gating parameters container BEFORE adding layers.
        self.gating_params = nn.ParameterDict()

        # Root layer
        self.root_layer = EnhancedAdaptiveNode(input_dim, hidden_dim)
        self.nodes.append(self.root_layer)
        root_id = self.registry.add_layer(self.root_layer, 'input', input_dim, input_dim)  # output_dim=input_dim
        self.gating_params[root_id] = nn.Parameter(torch.tensor(0.0))

        # Initialize additional layers.
        for _ in range(num_initial_layers - 1):
            layer_id = self._add_layer(input_dim, hidden_dim)
            # _add_layer should append new gating parameters as well.

    def _add_layer(self, input_dim: int, hidden_dim: int, position: Optional[int] = None) -> str:
        new_layer = EnhancedAdaptiveNode(input_dim, hidden_dim)
        self.nodes.append(new_layer)
        
        if position is not None and position > 0:
            parent_ids = [self.registry.order[position-1]]
        else:
            parent_ids = [self.registry.order[-1]] if self.registry.order else None
        
        # Correct output_dim in registry to match node's output (input_dim)
        layer_id = self.registry.add_layer(
            new_layer, 'hidden', input_dim, input_dim,
            parent_ids=parent_ids, position=position
        )
        
        self.gating_params[layer_id] = nn.Parameter(torch.tensor(0.0))
        return layer_id

    def _remove_layer_by_id(self, layer_id: str):
        """Remove layer and update ModuleList"""
        try:
            if layer_id not in self.registry.layers:
                return
                
            # Get position in ModuleList
            layer_info = self.registry.layers[layer_id]
            module = layer_info['module']
            
            # Remove from ModuleList if present
            for i, node in enumerate(self.nodes):
                if node is module:
                    self.nodes = nn.ModuleList(
                        [n for j, n in enumerate(self.nodes) if j != i]
                    )
                    break
            
            # Remove from registry and gating
            self.registry.remove_layer(layer_id)
            if layer_id in self.gating_params:
                del self.gating_params[layer_id]
                
        except Exception as e:
            self.registry.logger.error(f"Error removing layer {layer_id}: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using registry ordering"""
        execution_path = self.registry.get_active_forward_path()
        
        if not execution_path:
            raise RuntimeError("No active layers in registry")
            
        # Process root layer
        root_id = execution_path[0]
        x = self.registry.layers[root_id]['module'](x)
        
        # Process subsequent layers with gating
        for layer_id in execution_path[1:]:
            layer_info = self.registry.layers[layer_id]
            if not layer_info['active']:
                continue
                
            # Apply gated residual connection
            gate = torch.sigmoid(self.gating_params[layer_id])
            node_output = layer_info['module'](x)
            x = x + gate * node_output * layer_info['warmup_factor']
            
            # Update usage metrics
            layer_info['metrics']['usage_frequency'] = (
                0.9 * layer_info['metrics'].get('usage_frequency', 0.0) + 0.1
            )
            
        return x

    def meta_step(self, val_loss: float):
        add_layer, remove_layer = self.meta_controller.analyze_performance(self.registry, val_loss)
        
        if add_layer:
            grad_mags = [
                (i, self.registry.layers[lid]['metrics']['gradient_mag'])
                for i, lid in enumerate(self.registry.order)
            ]
            position = max(grad_mags, key=lambda x: x[1])[0] + 1
            self._add_layer(
                self.input_dim, self.hidden_dim,  # Changed from (self.hidden_dim, self.hidden_dim)
                position=min(position, len(self.nodes))
            )
        
        if remove_layer:
            redundancy_scores = [
                (lid, l['metrics']['activation_sparsity'] * (1 - l['metrics']['usage_frequency']))
                for lid, l in self.registry.layers.items()
            ]
            if redundancy_scores:
                redundant_id = max(redundancy_scores, key=lambda x: x[1])[0]
                if redundancy_scores[0][1] > self.meta_controller.redundancy_threshold:
                    self._remove_layer_by_id(redundant_id)

    def update_metrics(self, gradients: Dict[str, float], 
                      activations: Dict[str, float]):
        """Update layer metrics from training"""
        for lid in self.registry.layers:
            if lid in gradients:
                self.registry.update_metrics(lid, {'gradient_mag': gradients[lid]})
            if lid in activations:
                self.registry.update_metrics(lid, {'activation_sparsity': activations[lid]})

    def increment_warmup(self):
        """Gradually activate new layers"""
        for lid in self.registry.layers:
            if self.registry.layers[lid]['warmup_factor'] < 1.0:
                self.registry.layers[lid]['warmup_factor'] = min(
                    1.0, 
                    self.registry.layers[lid]['warmup_factor'] + 
                    1/self.meta_controller.warmup_steps
                )

class MetaController:
    """Improved meta-controller with ModuleList integration"""

    def __init__(self,
                 complexity_threshold: float = 0.15,
                 redundancy_threshold: float = 0.25,
                 warmup_steps: int = 100,
                 max_layers: int = 10):
        self.complexity_threshold = complexity_threshold
        self.redundancy_threshold = redundancy_threshold
        self.warmup_steps = warmup_steps
        self.max_layers = max_layers
        self.performance_history = deque(maxlen=100)
        self.logger = logging.getLogger('MetaController')

    def analyze_performance(self, registry: GlobalLayerRegistry, 
                            current_loss: float) -> Tuple[bool, bool]:
        """Return (should_add_layer, should_remove_layer) based on loss trend and layer metrics."""
        self.performance_history.append(current_loss)
        
        if len(self.performance_history) >= 10:
            avg_old = np.mean(list(self.performance_history)[-10:-5])
            avg_new = np.mean(list(self.performance_history)[-5:])
            performance_plateau = abs(avg_old - avg_new) < self.complexity_threshold
            
            # Calculate redundancy scores from each layer's metrics.
            redundancy_scores = [
                (lid, registry.layers[lid]['metrics']['activation_sparsity'] * 
                      (1 - registry.layers[lid]['metrics']['usage_frequency']))
                for lid in registry.layers
            ]
            redundant_layers = sum(score > self.redundancy_threshold for _, score in redundancy_scores)
            
            add_layer = performance_plateau and (len(registry.layers) < self.max_layers)
            remove_layer = redundant_layers > 0
            
            return add_layer, remove_layer
        
        return False, False

    def execute_reorganization(self, registry: GlobalLayerRegistry,
                               network: EnhancedAdaptiveNetwork,
                               add_layer: bool, remove_layer: bool):
        """Modify network architecture based on performance analysis."""
        if add_layer:
            self._add_strategic_layer(registry, network)
        if remove_layer:
            self._remove_redundant_layer(registry, network)

    def _add_strategic_layer(self, registry: GlobalLayerRegistry, network: EnhancedAdaptiveNetwork):
        """Add a new layer after the layer with the highest gradient magnitude."""
        layer_ids = list(registry.layers.keys())
        grad_mags = [registry.layers[lid]['metrics']['gradient_mag'] for lid in layer_ids]
        parent_id = layer_ids[np.argmax(grad_mags)]
        
        # Use network's synchronized add method (assumed to be defined in EnhancedAdaptiveNetwork)
        new_id = network.add_layer(parent_id)
        
        # Re-route parent's children to be connected after the new layer.
        children = registry.connections[parent_id]
        registry.connections[new_id].update(children)
        registry.connections[parent_id] = {new_id}
        
        # Initialize warmup factor for the new layer.
        registry.layers[new_id]['warmup_factor'] = 0.0
        self.logger.info(f"Added new layer {new_id} after {parent_id}")

    def _remove_redundant_layer(self, registry: GlobalLayerRegistry, network: EnhancedAdaptiveNetwork):
        """Remove the layer with the highest redundancy score if above the threshold."""
        redundancy_scores = [
            (lid, registry.layers[lid]['metrics']['activation_sparsity'] * 
                  (1 - registry.layers[lid]['metrics']['usage_frequency']))
            for lid in registry.layers
        ]
        if redundancy_scores:
            redundant_tuple = max(redundancy_scores, key=lambda x: x[1])
            if redundant_tuple[1] > self.redundancy_threshold:
                # Use network's synchronized removal method.
                network._remove_layer_by_id(redundant_tuple[0])
                self.logger.info(f"Removed redundant layer {redundant_tuple[0]}")

class StructuralPlasticityController:
    """
    Enhanced implementation of structural plasticity with robust pruning and growth strategies.
    Handles edge cases and enforces constraints while maintaining network stability.
    """
    
    def __init__(self,
            prune_threshold: float = 0.005,
            growth_rate: float = 0.2,
            split_threshold: float = 0.5,
            merge_threshold: float = 0.4,
            prune_frequency: int = 10,
            structural_frequency: int = 20,
            min_connections: int = 1000,
            activation_growth_ratio: float = 0.7,
            growth_noise_scale: float = 0.01,
            split_noise_scale: float = 0.1,
            merge_similarity_threshold: float = 0.6,
            epsilon: float = 1e-8):
        
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.prune_frequency = prune_frequency
        self.structural_frequency = structural_frequency
        self.min_connections = min_connections
        self.activation_growth_ratio = activation_growth_ratio
        self.growth_noise_scale = growth_noise_scale
        self.split_noise_scale = split_noise_scale
        self.merge_similarity_threshold = merge_similarity_threshold
        self.epsilon = epsilon
        
        self.stats = defaultdict(list)
        self.epoch_counter = 0
        self.logger = logging.getLogger('StructuralPlasticity')

    def step(self, model: nn.Module):
        """Execute structural updates at configured intervals"""
        # Resolve circular import: adaptive_feedforward imports MovingStats from this module,
        # so the top-level import fails. Import lazily here and cache in the global.
        global AdaptiveFeedForward
        if AdaptiveFeedForward is None:
            try:
                from transformer_model.adaptive_feedforward import AdaptiveFeedForward as _AFF
                AdaptiveFeedForward = _AFF
            except ImportError:
                pass

        self.epoch_counter += 1
        
        if self.epoch_counter % self.prune_frequency == 0:
            self._prune_connections(model)
            self._grow_connections(model)
        
        if self.epoch_counter % self.structural_frequency == 0:
            self._split_neurons(model)
            self._merge_neurons(model)
            self._log_network_stats(model)
        
        self._log_detailed_stats()

    def _split_neurons(self, model: nn.Module):
            """Identify and split overloaded neurons in both EnhancedAdaptiveNode and AdaptiveFeedForward."""
            total_splits = 0
            for name, module in model.named_modules():
                # Determine layer types based on module type
                if isinstance(module, EnhancedAdaptiveNode):
                    layer_types = ['input', 'hidden', 'output']
                elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                    layer_types = ['input']  # AdaptiveFeedForward only supports splitting 'input'
                else:
                    continue

                for layer_type in layer_types:
                    try:
                        # AdaptiveFeedForward uses 'intermediate_activation_stats' for the FFN middle layer
                        if isinstance(module, AdaptiveFeedForward) and layer_type == 'input':
                            stats = module.intermediate_activation_stats
                        else:
                            stats = getattr(module, f"{layer_type}_activation_stats")

                        if stats.recent_activations is None:
                            continue

                        activations = stats.recent_activations
                        if activations.ndim != 2:
                            continue

                        # Calculate neuron utilization
                        mean_activations = activations.mean(dim=0)
                        neuron_age_np = module.neuron_age.numpy() if module.neuron_age.numel() > 0 else np.array([0])
                        overload_mask = (mean_activations > self.split_threshold) & \
                                    (module.neuron_age[:mean_activations.shape[0]] < np.percentile(neuron_age_np, 75))
                        overload_indices = overload_mask.nonzero().squeeze().tolist()

                        if isinstance(overload_indices, int):
                            overload_indices = [overload_indices]

                        for idx in overload_indices:
                            module.split_neuron(layer_type, idx)
                            total_splits += 1
                            self.logger.info(f"Split neuron {idx} in {layer_type} layer of {name}")

                    except Exception as e:
                        self.logger.error(f"Error splitting neurons in {name}.{layer_type}: {str(e)}")
                        continue

            self.stats['total_splits'].append(total_splits)


    def _merge_neurons(self, model: nn.Module):
            """Identify and merge underutilized neurons in both EnhancedAdaptiveNode and AdaptiveFeedForward."""
            total_merges = 0
            for name, module in model.named_modules():
                # Determine layer types based on module type
                if isinstance(module, EnhancedAdaptiveNode):
                    layer_types = ['input', 'hidden', 'output']
                elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                    layer_types = ['input']  # AdaptiveFeedForward only supports merging 'input'
                else:
                    continue

                for layer_type in layer_types:
                    try:
                        # Get the weight layer for similarity computation
                        layer = getattr(module, f"{layer_type}_layer")
                        weights = layer.weight.data.cpu().numpy()
                        norms = np.linalg.norm(weights, axis=1)

                        # Skip if any norm is zero (prevents division by zero)
                        if np.any(norms < 1e-10):
                            continue

                        normalized = weights / norms[:, np.newaxis]
                        similarity = np.dot(normalized, normalized.T)

                        # Find merge candidates
                        candidates = []
                        for i in range(similarity.shape[0]):
                            for j in range(i+1, similarity.shape[0]):
                                if similarity[i,j] > self.merge_similarity_threshold:
                                    candidates.append((i,j))

                        # Merge most similar pairs
                        merged = set()
                        for i,j in sorted(candidates, key=lambda x: similarity[x], reverse=True):
                            if i not in merged and j not in merged:
                                module.merge_neurons(layer_type, [i,j])
                                merged.update([i,j])
                                total_merges += 1
                                self.logger.info(f"Merged neurons {i},{j} in {layer_type} layer of {name}")

                    except Exception as e:
                        self.logger.error(f"Error merging neurons in {name}.{layer_type}: {str(e)}")
                        continue

            self.stats['total_merges'].append(total_merges)


    def _validate_tensor_dimensions(self, tensor: torch.Tensor, expected_dims: int) -> bool:
        """Validate tensor dimensions to ensure proper reshaping."""
        return len(tensor.shape) == expected_dims

    def _ensure_1d_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 1D while preserving indices."""
        if len(tensor.shape) == 0:
            return tensor.unsqueeze(0)
        elif len(tensor.shape) > 1:
            return tensor.reshape(-1)
        return tensor

    def _get_pruning_candidates(self,
                                weight: torch.Tensor, 
                                mask: torch.Tensor,
                                min_active: int) -> Tuple[torch.Tensor, int]:
            """
            Identify valid pruning candidates while strictly enforcing minimum connection constraints.
            
            Args:
                weight: Weight tensor for the layer
                mask: Binary mask indicating active connections
                min_active: Minimum number of active connections to maintain
                
            Returns:
                Tuple of (candidate mask, number of allowed pruning candidates)
            """
            with torch.no_grad():
                weight_mag = torch.abs(weight)
                current_active = torch.sum(mask).item()
                
                # Ensure we maintain minimum connections
                if current_active <= min_active:
                    self.logger.debug(f"Skipping pruning: active connections ({current_active}) at minimum ({min_active})")
                    return torch.zeros_like(mask), 0
                
                # Find candidates below threshold that are currently active
                candidates = (weight_mag < self.prune_threshold) & (mask == 1)
                potential_prune_count = torch.sum(candidates).item()
                
                if potential_prune_count == 0:
                    self.logger.debug("No pruning candidates found")
                    return torch.zeros_like(mask), 0
                
                # Calculate maximum allowed pruning while maintaining minimum connections
                max_allowed_prune = int(current_active - min_active)
                actual_prune_count = min(potential_prune_count, max_allowed_prune)
                
                if actual_prune_count <= 0:
                    self.logger.debug("Cannot prune without violating minimum connection constraint")
                    return torch.zeros_like(mask), 0
                
                # Get indices of candidates sorted by magnitude
                candidate_indices = torch.nonzero(candidates).squeeze()
                candidate_indices = self._ensure_1d_tensor(candidate_indices)
                
                if len(candidate_indices) == 0:
                    return torch.zeros_like(mask), 0
                
                candidate_weights = weight_mag[candidates]
                _, sort_indices = torch.sort(candidate_weights)
                
                # Select the smallest weights up to the allowed pruning count
                prune_indices = candidate_indices[sort_indices[:actual_prune_count]]
                
                # Create final pruning mask
                prune_mask = torch.zeros_like(mask, dtype=torch.bool)
                prune_mask.view(-1)[prune_indices] = True
                
                self.logger.debug(f"Selected {actual_prune_count} connections for pruning")
                return prune_mask, actual_prune_count

    def _calculate_growth_scores(self,
                                module: EnhancedAdaptiveNode,
                                layer: nn.Linear,
                                mask: torch.Tensor,
                                activation_stats: 'MovingStats',
                                layer_name: str) -> torch.Tensor:
        """
        Calculate growth scores using per-neuron activation statistics when available.
        """
        with torch.no_grad():
            # Get inactive positions
            inactive = (mask == 0)
            if not torch.any(inactive):
                self.logger.debug(f"No inactive connections in {layer_name}")
                return torch.zeros_like(mask, dtype=torch.float32)
            
            # Determine growth strategy
            activation_level = activation_stats.mean
            use_activation_based = (activation_level > 0.1 and 
                                np.random.rand() < self.activation_growth_ratio)
            
            # Determine layer type once at the start
            is_input_layer = layer_name.endswith("input_layer")
            
            self.logger.debug(
                f"Layer {layer_name}: Using {'activation-based' if use_activation_based else 'random'} growth"
            )
            
            if use_activation_based:
                # Get input and output activations from same batch
                if is_input_layer:
                    # Handle input layer specially
                    input_act = module.recent_raw_input
                    output_act = activation_stats.recent_activations
                    
                    # Explicit reshape using raw input dimensions
                    batch_size = input_act.size(0)
                    input_dim = module._current_input_dim
                    input_act = input_act.reshape(batch_size, input_dim)
                    output_act = output_act.reshape(batch_size, -1)
                    
                    self.logger.debug(
                        f"Input layer activations - Raw: {module.recent_raw_input.shape}, "
                        f"Reshaped input: {input_act.shape}, Output: {output_act.shape}"
                    )
                else:
                    # For hidden/output layers
                    input_act = activation_stats.recent_activations
                    output_act = activation_stats.recent_outputs
                    
                    # Standard reshaping
                    batch_size = input_act.size(0)
                    input_act = input_act.reshape(batch_size, -1)
                    output_act = output_act.reshape(batch_size, -1)

                # Add debug logging
                self.logger.debug(
                    f"After reshaping ({layer_name}): "
                    f"Input: {input_act.shape}, Output: {output_act.shape}"
                )

                # Validate activation data
                if input_act is None or output_act is None:
                    self.logger.warning(f"Missing activation data in {layer_name}, using random scores")
                    return torch.rand_like(layer.weight.data) * inactive.float()
                    
                try:
                    # Ensure matching batch dimensions
                    batch_size = input_act.size(0)
                    output_act = output_act[:batch_size]  # Truncate to match input batch size
                    
                    # Reshape tensors
                    input_act = input_act.reshape(batch_size, -1)
                    output_act = output_act.reshape(batch_size, -1)

                    # Log shapes for debugging
                    self.logger.debug(
                        f"Growth scoring - {layer_name}: "
                        f"Input shape: {input_act.shape}, Output shape: {output_act.shape}"
                    )

                    # Compute correlation matrix
                    correlation = torch.mm(input_act.t(), output_act)
                    
                    # Handle input layer transpose
                    if is_input_layer:
                        correlation = correlation.t()  # [784, 256] -> [256, 784]
                    
                    # Normalize and calculate scores
                    correlation = correlation / (batch_size + self.epsilon)
                    scores = torch.abs(correlation)

                    # Verify score dimensions
                    if scores.shape != layer.weight.shape:
                        self.logger.warning(
                            f"Score shape mismatch in {layer_name}. "
                            f"Expected {layer.weight.shape}, got {scores.shape}. "
                            "Falling back to random scores."
                        )
                        scores = torch.rand_like(layer.weight.data)
                        
                except (RuntimeError, ValueError) as e:
                    self.logger.warning(
                        f"Error computing correlation scores for {layer_name}: {str(e)}. "
                        "Falling back to random scores."
                    )
                    scores = torch.rand_like(layer.weight.data)
            else:
                # Generate random scores for non-activation-based growth
                scores = torch.rand_like(layer.weight.data)
            
            # Apply inactive mask and normalize
            scores = scores * inactive.float()
            total_score = torch.sum(scores) + self.epsilon
            scores = scores / total_score
            
            return scores

    def _grow_connections(self, model: nn.Module):
        """Implement connection growth with detailed activation-based prioritization."""
        total_grown = 0
        for name, module in model.named_modules():
            # Determine layer types based on module type
            if isinstance(module, EnhancedAdaptiveNode):
                layer_types = ['input', 'hidden', 'output']
            elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                layer_types = ['input', 'output']
            else:
                continue

            for layer_type in layer_types:
                layer = getattr(module, f"{layer_type}_layer")
                mask = getattr(module, f"{layer_type}_layer_weight_mask")

                # Get the right activation stats for this layer type
                if isinstance(module, AdaptiveFeedForward) and layer_type == 'input':
                    stats = module.intermediate_activation_stats
                elif isinstance(module, AdaptiveFeedForward) and layer_type == 'output':
                    stats = module.output_activation_stats
                else:
                    stats = getattr(module, f"{layer_type}_activation_stats")

                layer_name = f"{name}.{layer_type}_layer"

                with torch.no_grad():
                    # Skip if no inactive connections
                    inactive = (mask == 0)
                    num_inactive = torch.sum(inactive).item()

                    if num_inactive == 0:
                        self.logger.debug(f"No inactive connections in {layer_name}")
                        continue

                    # Calculate number of connections to grow
                    num_to_grow = max(1, int(num_inactive * self.growth_rate))
                    self.logger.debug(f"Growing {num_to_grow} connections in {layer_name}")

                    # For AdaptiveFeedForward, use random growth scores (no recent_raw_input)
                    if AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                        scores = torch.rand_like(layer.weight.data) * inactive.float()
                        total_score = torch.sum(scores) + self.epsilon
                        scores = scores / total_score
                    else:
                        scores = self._calculate_growth_scores(module, layer, mask, stats, layer_name)

                    if torch.sum(scores) == 0:
                        self.logger.debug(f"No valid growth scores for {layer_name}")
                        continue

                    # Sample connections to grow
                    flat_scores = scores.flatten()
                    try:
                        flat_indices = torch.multinomial(flat_scores,
                                                    num_to_grow,
                                                    replacement=False)
                    except RuntimeError as e:
                        self.logger.warning(
                            f"Error sampling growth indices for {layer_name}: {str(e)}"
                        )
                        continue

                    # Initialize new connections
                    rows = flat_indices // layer.weight.shape[1]
                    cols = flat_indices % layer.weight.shape[1]

                    new_weights = torch.randn(num_to_grow,
                                            device=layer.weight.device) * self.growth_noise_scale

                    layer.weight.data[rows, cols] = new_weights
                    mask[rows, cols] = 1.0
                    total_grown += num_to_grow

                    self.logger.debug(
                        f"Successfully grew {num_to_grow} connections in {layer_name}"
                    )

        self.stats['total_grown'].append(total_grown)
        self.logger.info(f"Grew {total_grown} new connections across all layers")

    def _prune_connections(self, model: nn.Module):
            """Implement connection pruning with strict minimum connection enforcement."""
            total_pruned = 0
            for name, module in model.named_modules():
                # Determine layer types based on module type
                if isinstance(module, EnhancedAdaptiveNode):
                    layer_types = ['input', 'hidden', 'output']
                elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                    layer_types = ['input', 'output']
                else:
                    continue

                for layer_type in layer_types:
                    layer = getattr(module, f"{layer_type}_layer")
                    mask = getattr(module, f"{layer_type}_layer_weight_mask")

                    layer_name = f"{name}.{layer_type}_layer"

                    # Get pruning candidates
                    prune_mask, num_to_prune = self._get_pruning_candidates(
                        layer.weight.data,
                        mask,
                        self.min_connections
                    )

                    if num_to_prune > 0:
                        # Apply pruning
                        with torch.no_grad():
                            layer.weight.data[prune_mask] = 0.0
                            mask[prune_mask] = 0.0
                            total_pruned += num_to_prune

                            self.logger.debug(
                                f"Pruned {num_to_prune} connections in {layer_name}"
                            )

            self.stats['total_pruned'].append(total_pruned)
            self.logger.info(f"Pruned {total_pruned} connections across all layers")

    def _log_network_stats(self, model: nn.Module):
            """Log comprehensive per-layer network statistics."""
            total_active = 0
            total_params = 0
            layer_stats = {}

            for name, module in model.named_modules():
                # Determine layer types based on module type
                if isinstance(module, EnhancedAdaptiveNode):
                    layer_types = ['input', 'hidden', 'output']
                elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                    layer_types = ['input', 'output']
                else:
                    continue

                for layer_type in layer_types:
                    layer_name = f"{name}.{layer_type}_layer"
                    mask = getattr(module, f"{layer_type}_layer_weight_mask")
                    active = torch.sum(mask).item()
                    total = mask.numel()

                    layer_stats[layer_name] = {
                        'active': active,
                        'total': total,
                        'sparsity': 1 - (active / total) if total > 0 else 0.0
                    }
                    total_active += active
                    total_params += total

                    self.logger.debug(
                        f"Layer {layer_name}: {active}/{total} active connections "
                        f"({layer_stats[layer_name]['sparsity']:.2%} sparse)"
                    )

            self.stats['total_active'].append(total_active)
            self.stats['total_params'].append(total_params)
            self.stats['global_sparsity'].append(1 - (total_active / total_params) if total_params > 0 else 0.0)
            self.stats['layer_stats'].append(layer_stats)

    def _log_detailed_stats(self):
            """Log detailed structural plasticity statistics with proper formatting."""
            if not self.stats.get('layer_stats'):
                return

            stats_str = "\n".join([
                f"{k}: {v[-1]}" for k, v in self.stats.items()
                if k != 'layer_stats' and v
            ])

            layer_stats = self.stats['layer_stats'][-1]
            layer_details = "\n".join([
                f"  {name}: {stats['active']}/{stats['total']} connections "
                f"({stats['sparsity']:.2%} sparse)"
                for name, stats in layer_stats.items()
            ])
            
            self.logger.info(
                f"Structural plasticity stats:\n{stats_str}\n"
                f"Per-layer statistics:\n{layer_details}"
            )

    def get_global_sparsity(self, model: nn.Module) -> float:
        global AdaptiveFeedForward
        if AdaptiveFeedForward is None:
            try:
                from transformer_model.adaptive_feedforward import AdaptiveFeedForward as _AFF
                AdaptiveFeedForward = _AFF
            except ImportError:
                pass
        total_active = 0
        total_params = 0
        for name, module in model.named_modules():
            # Determine layer types based on module type
            if isinstance(module, EnhancedAdaptiveNode):
                layer_types = ['input', 'hidden', 'output']
            elif AdaptiveFeedForward is not None and isinstance(module, AdaptiveFeedForward):
                layer_types = ['input', 'output']
            else:
                continue

            for layer_type in layer_types:
                mask = getattr(module, f"{layer_type}_layer_weight_mask")
                active = torch.sum(mask).item()
                total = mask.numel()
                total_active += active
                total_params += total
        return 1 - (total_active / total_params) if total_params > 0 else 0.0

class TrainingManager:
    def __init__(self, network: EnhancedAdaptiveNetwork, 
                 structural_plasticity: StructuralPlasticityController, 
                 lr: float = 0.001, batch_size: int = 32):
        self.network = network
        self.structural_plasticity = structural_plasticity
        self.optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.network(x)
        loss = self.criterion(output, x)
        
        # Backward pass
        loss.backward()
        
        # Collect metrics before optimization
        gradients = self._collect_gradient_metrics()
        activations = self._collect_activation_metrics()
        
        self.optimizer.step()
        
        # Update network metrics
        self.network.update_metrics(gradients, activations)
        self.network.increment_warmup()
        
        return {'loss': loss.item()}

    def train_epoch(self, train_data: Iterable[torch.Tensor],
                val_data: Optional[Iterable[torch.Tensor]] = None
                ) -> Dict[str, float]:
        self.network.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_data:
            metrics = self.train_step(batch)
            train_loss += metrics['loss']
            num_batches += 1
            
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        # Apply structural plasticity update per epoch
        self.structural_plasticity.step(self.network)
        
        # Compute adaptation reward
        current_global_sparsity = self.structural_plasticity.get_global_sparsity(self.network)
        adaptation_reward = -ADAPT_REWARD_SCALE * abs(current_global_sparsity - TARGET_SPARSITY)
        
        if val_data is not None:
            val_metrics = self.validate(val_data)
            self.network.meta_step(val_metrics['val_loss'])
            return {
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['val_loss'],
                'adaptation_reward': adaptation_reward
            }
        else:
            return {
                'train_loss': avg_train_loss,
                'adaptation_reward': adaptation_reward
        }
    
    def validate(self, val_data: Iterable[torch.Tensor]) -> Dict[str, float]:
        """Validate the network on the given validation data."""
        self.network.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_data:
                output = self.network(batch)
                loss = self.criterion(output, batch)
                val_loss += loss.item()
                num_batches += 1
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_val_loss}

    def _collect_gradient_metrics(self) -> Dict[str, float]:
        """Collect gradient magnitudes per layer"""
        gradients = {}
        for lid, layer in self.network.registry.layers.items():
            grads = []
            for param in layer['module'].parameters():
                if param.grad is not None:
                    grads.append(param.grad.abs().mean().item())
            gradients[lid] = np.mean(grads) if grads else 0.0
        return gradients

    def _collect_activation_metrics(self) -> Dict[str, float]:
        """Collect activation sparsity per layer"""
        activations = {}
        for lid, layer in self.network.registry.layers.items():
            if hasattr(layer['module'], 'hidden_activation_stats'):
                acts = layer['module'].hidden_activation_stats.recent_activations
                if acts is not None:
                    sparsity = (torch.abs(acts) < 0.01).float().mean().item()
                    activations[lid] = sparsity
        return activations
    
    def meta_update(self, val_loss: float):
        """Trigger architectural updates (layer addition/removal) based on validation loss."""
        add_layer, remove_layer = self.network.meta_controller.analyze_performance(
            self.network.registry, val_loss
        )
        self.network.meta_controller.execute_reorganization(
            self.network.registry, self.network, add_layer, remove_layer
        )

class MetaLearner(nn.Module):
    """Complete meta-learning implementation with optimization memory"""
    
    def __init__(self, state_size: int, action_size: int,
                 hidden_dim: int = 256,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        super().__init__()
        
        # Network components
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Experience replay
        self.memory = ExperienceMemory(memory_size)
        self.batch_size = batch_size
        
        # Optimization components
        self.optimizer = torch.optim.Adam([
            {'params': self.state_encoder.parameters(), 'lr': 0.001},
            {'params': self.policy_head.parameters(), 'lr': 0.001},
            {'params': self.value_head.parameters(), 'lr': 0.001}
        ])
        
        # Hyperparameters
        self.gamma = 0.99
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Statistics
        self.training_stats = MovingStats(window_size=100)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through meta-learner"""
        encoded_state = self.state_encoder(state)
        action_logits = self.policy_head(encoded_state)
        value = self.value_head(encoded_state)
        return action_logits, value
        
    def update_from_experience(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return {}
            
        # Sample experience batch
        batch = self.memory.sample(batch_size)
        states = torch.stack([x['state'] for x in batch])
        actions = torch.stack([x['action'] for x in batch])
        rewards = torch.tensor([x['reward'] for x in batch])
        adaptation_rewards = torch.tensor([x['adaptation_reward'] for x in batch])

        # Retrieve old log probs (stored at collection time)
        old_log_probs = torch.stack([x['log_prob'] for x in batch]).detach()

        LAMBDA_ADAPT = 5.0  # Hyperparameter to weight adaptation reward

        # Compute advantages with adaptation reward
        _, values = self(states)
        total_rewards = rewards + LAMBDA_ADAPT * adaptation_rewards
        advantages = total_rewards - values.squeeze()

        # Compute policy loss with correct importance sampling ratio
        action_logits, new_values = self(states)
        policy_distribution = Normal(action_logits, torch.ones_like(action_logits))

        new_log_probs = policy_distribution.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clip objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values.squeeze(), total_rewards)
        
        # Entropy bonus
        entropy = policy_distribution.entropy().mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Track statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
        self.training_stats.update_dict(stats)
        
        return stats
    
class ExperienceMemory:
    """Efficient experience replay memory with prioritization"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros(capacity)
        
    def push(self, experience: Dict[str, torch.Tensor]) -> None:
        """Add experience to memory with prioritization"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience

        # Use absolute reward as initial priority (higher reward = more likely sampled)
        reward = experience.get('reward', 0)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self.priorities[self.position] = max(abs(reward), 1e-6)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample batch with prioritization"""
        if len(self.memory) < batch_size:
            return self.memory
            
        probs = self.priorities[:len(self.memory)] / self.priorities[:len(self.memory)].sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[idx] for idx in indices]
        
    def __len__(self) -> int:
        return len(self.memory)
    
class MovingStats:
    def __init__(self, window_size: int = 100, alpha: float = 0.1):
        self.window = deque(maxlen=window_size)
        self.ema = None
        self.alpha = alpha
        self.recent_activations = None  # Store single batch
        self.recent_outputs = None      # Store single batch

    # Add this property definition
    @property
    def mean(self) -> float:
        """Calculate mean of stored values."""
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def update(self, value: float, activations: Optional[torch.Tensor] = None, 
              batch_size: Optional[int] = None):
        self.window.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
            
        if activations is not None:
            try:
                # Store only the most recent batch
                self.recent_activations = activations.detach().cpu()
                if batch_size is not None:
                    features = activations.numel() // batch_size
                    self.recent_activations = self.recent_activations.view(batch_size, features)
            except Exception as e:
                logging.warning(f"Error updating activations: {str(e)}")
                self.recent_activations = None

    def update_outputs(self, outputs: torch.Tensor, batch_size: Optional[int] = None):
        try:
            # Store only the most recent output batch
            self.recent_outputs = outputs.detach().cpu()
            if batch_size is not None:
                features = outputs.numel() // batch_size
                self.recent_outputs = self.recent_outputs.view(batch_size, features)
        except Exception as e:
            logging.warning(f"Error updating outputs: {str(e)}")
            self.recent_outputs = None

    def update_dict(self, stats_dict: dict) -> None:
        """Update with dictionary of named statistics."""
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)):
                self.update(value)

class AdaptiveOptimizer:
    """Optimizer with adaptive learning rates based on plasticity"""
    
    def __init__(self, parameters, lr: float = 0.001):
        self.opt = torch.optim.Adam(parameters, lr=lr)
        self.stats = MovingStats()
        
    def step(self, loss: torch.Tensor) -> None:
        """Take optimization step with adaptive learning rate"""
        self.stats.update(loss.item())
        
        # Adjust learning rate based on loss statistics
        if self.stats.std > 0:
            for param_group in self.opt.param_groups:
                param_group['lr'] *= torch.clamp(
                    torch.tensor(1.0 / self.stats.std), 0.1, 10.0
                )
        
        self.opt.step()

def create_production_network(
    input_dim: int = 784,
    hidden_dim: int = 256,
    num_initial_layers: int = 3
) -> Tuple[EnhancedAdaptiveNetwork]:
    """Create production-ready network with all components"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create network components
    network = EnhancedAdaptiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_initial_layers=num_initial_layers
    )
    
    return network

def train_production_network(
    network: EnhancedAdaptiveNetwork,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    num_epochs: int = 100,
    early_stopping_patience: int = 15
) -> Dict[str, List[float]]:
    """Production training loop with monitoring and early stopping"""
    
    training_manager = TrainingManager(network)
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'adaptation_reward': [],
        'meta_learner_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_metrics = training_manager.train_epoch(train_data)
        
        # Validation phase
        val_metrics = training_manager.validate(val_data)
        
        # Update history
        for key in history:
            if key in train_metrics:
                history[key].append(train_metrics[key])
            elif key in val_metrics:
                history[key].append(val_metrics[key])
        
        # Early stopping check
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            # Save best model
            torch.save(network.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Log progress
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.4f} - "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"Adaptation Reward: {train_metrics['adaptation_reward']:.4f}"
        )
    
    return history

if __name__ == "__main__":
    # Create dummy training and validation data for demonstration.
    dummy_train = [torch.randn(32, 784) for _ in range(50)]
    dummy_val = [torch.randn(32, 784) for _ in range(10)]
    
    # Create a StructuralPlasticityController instance with desired parameters.
    sp_controller = StructuralPlasticityController(
        prune_threshold=0.05,
        growth_rate=0.4,
        split_threshold=0.2,
        merge_threshold=0.4,
        prune_frequency=10,
        structural_frequency=20,
        min_connections=100,
        activation_growth_ratio=0.7,
        growth_noise_scale=0.01,
        split_noise_scale=0.1,
        merge_similarity_threshold=0.6,
        epsilon=1e-8
    )

    # Create the production network.
    model = create_production_network(input_dim=784, hidden_dim=256, num_initial_layers=3)
    
    # Create the training manager with the structural plasticity controller.
    trainer = TrainingManager(model, sp_controller, lr=0.001, batch_size=32)
    
    num_epochs = 100
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'adaptation_reward': []}

    meta_learner = MetaLearner(state_size=784, action_size=2)
    
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(dummy_train, val_data=dummy_val)
        val_metrics = trainer.validate(dummy_val)
        
        # Example state (e.g., flattened input or network stats)
        state = dummy_train[0].flatten()  # Placeholder; adjust based on your state definition
        action = torch.tensor([0])  # Placeholder; e.g., 0 for no change, 1 for structural update
        reward = -val_metrics['val_loss']  # Environment reward (negative validation loss)
        adaptation_reward = train_metrics['adaptation_reward']
        
        # Push experience to memory — compute log_prob at collection time for PPO
        with torch.no_grad():
            action_logits, _ = meta_learner(state.unsqueeze(0))
            policy_dist = Normal(action_logits, torch.ones_like(action_logits))
            log_prob = policy_dist.log_prob(action.unsqueeze(0).float()).sum(dim=-1).squeeze()
        meta_learner.memory.push({
            'state': state,
            'action': action,
            'reward': reward,
            'adaptation_reward': adaptation_reward,
            'log_prob': log_prob,
        })
        
        # Periodically update MetaLearner (e.g., every 10 epochs)
        if epoch % 10 == 0 and len(meta_learner.memory) >= meta_learner.batch_size:
            meta_stats = meta_learner.update_from_experience()
            print(f"Meta-Learner Update: {meta_stats}")
        
        # Record history
        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['adaptation_reward'].append(train_metrics['adaptation_reward'])
        
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_metrics['train_loss']:.4f} - "
              f"Val Loss: {val_metrics['val_loss']:.4f} - "
              f"Adaptation Reward: {train_metrics['adaptation_reward']:.4f}")
        
        # Early stopping check
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # After training, print out structural plasticity statistics.
    print("\nStructural Plasticity Stats:")
    for key, value in sp_controller.stats.items():
        print(f"{key}: {value[-1] if value else 'N/A'}")
    
    # Optionally, print the final network architecture.
    print("\nFinal Network Structure:")
    print(model)
    
    print("\nTraining complete. History:", history)
