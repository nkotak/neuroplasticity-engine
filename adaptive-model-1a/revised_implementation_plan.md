# Revised Implementation Plan for Enhanced Neuroplasticity

After reviewing the codebase thoroughly, including `adaptive_transformer.py`, `adaptive_feedforward.py`, `main.py`, and `inference.py`, I've updated the implementation plan to ensure seamless integration with the existing architecture.

## 1. Evolving Adaptive Connectivity

### 1.1 Skip Connection Registry and Management

The implementation will build on the existing `GlobalLayerRegistry` to support skip connections between non-adjacent layers:

```python
# Within GlobalLayerRegistry class:
def __init__(self):
    # Existing initialization code...
    
    # Add skip connections registry
    self.skip_connections = {}  # Format: {(source_id, target_id): connection_info}
    
def add_skip_connection(self, source_id: str, target_id: str, 
                       weight_shape: tuple, device=None) -> str:
    """Register a new skip connection between two layers"""
    connection_id = f"skip_{source_id}_{target_id}"
    
    # Skip if connection already exists
    if (source_id, target_id) in self.skip_connections:
        return self.skip_connections[(source_id, target_id)]['id']
    
    # Initialize weight and gate parameters
    if device is None and self.device is not None:
        device = self.device
        
    # Create connection info
    self.skip_connections[(source_id, target_id)] = {
        'id': connection_id,
        'source_id': source_id,
        'target_id': target_id,
        'weight': torch.zeros(weight_shape, device=device).normal_(0, 0.02),
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
    """Remove a skip connection"""
    if (source_id, target_id) in self.skip_connections:
        del self.skip_connections[(source_id, target_id)]
        self.logger.info(f"Removed skip connection from {source_id} to {target_id}")
```

### 1.2 Skip Connection Analysis and Decision Engine

Create a new `SkipConnectionController` class that will analyze layer activations and determine when to add or remove skip connections:

```python
class SkipConnectionController:
    """Controller that analyzes layer activations and manages skip connections"""
    
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
        
        self.logger = logging.getLogger('SkipConnectionController')
        
    def update_activations(self, layer_id: str, activations: torch.Tensor):
        """Track layer activations for correlation analysis"""
        if layer_id not in self.layer_activations:
            self.layer_activations[layer_id] = deque(maxlen=self.activation_window)
            
        # Store activation statistics (mean across batch)
        activation_stats = activations.detach().mean(dim=0).cpu()
        self.layer_activations[layer_id].append(activation_stats)
    
    def analyze_layer_correlations(self, registry: GlobalLayerRegistry, network) -> List[Tuple[str, str]]:
        """Analyze correlations between layers and find candidates for skip connections"""
        # Only analyze if we have enough activation data
        candidates = []
        layer_ids = registry.order
        
        # Check correlations between all pairs of non-adjacent layers
        for i, source_id in enumerate(layer_ids):
            if source_id not in self.layer_activations:
                continue
                
            source_activations = torch.stack(list(self.layer_activations[source_id]))
            
            # Look for potential target layers (skipping adjacent layers)
            for j, target_id in enumerate(layer_ids):
                # Skip if target doesn't have activations or is adjacent
                if (target_id not in self.layer_activations or 
                    abs(i - j) <= 1 or  # Skip adjacent layers
                    (source_id, target_id) in registry.skip_connections):  # Skip existing
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
                    correlation = F.cosine_similarity(
                        source_acts.view(min_len, -1), 
                        target_acts.view(min_len, -1)
                    ).mean().item()
                    
                    # Store correlation
                    self.layer_correlations[(source_id, target_id)] = correlation
                    
                    # If correlation exceeds threshold, add as candidate
                    if correlation > self.correlation_threshold:
                        candidates.append((source_id, target_id))
                        self.logger.info(f"Found skip connection candidate: {source_id} -> {target_id} (correlation: {correlation:.4f})")
                except:
                    self.logger.warning(f"Could not compute correlation for {source_id} -> {target_id}")
                
        return candidates
    
    def step(self, registry: GlobalLayerRegistry, network, gradients: Dict[str, float]) -> int:
        """Analyze network and add/remove skip connections as needed"""
        # Analyze correlation between layers
        candidates = self.analyze_layer_correlations(registry, network)
        
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
        
        return added_connections
        
    def _update_skip_connection_metrics(self, registry: GlobalLayerRegistry, network, gradients: Dict[str, float]):
        """Update performance metrics for skip connections"""
        for (source_id, target_id), connection in registry.skip_connections.items():
            # Update correlation metric
            correlation = self.layer_correlations.get((source_id, target_id), 0.0)
            connection['metrics']['correlation'] = correlation
            
            # Update gradient flow metric (if available)
            if source_id in gradients and target_id in gradients:
                gradient_flow = min(gradients[source_id], gradients[target_id])
                connection['metrics']['gradient_flow'] = gradient_flow
            
            # Increment warmup factor
            connection['warmup_factor'] = torch.min(
                torch.tensor(1.0, device=connection['warmup_factor'].device),
                connection['warmup_factor'] + 1.0/self.warmup_steps
            )
        
    def _prune_skip_connections(self, registry: GlobalLayerRegistry, network):
        """Remove skip connections that aren't providing benefit"""
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
```

### 1.3 Integration with Adaptive Transformer

Update the `AdaptiveTransformer` forward pass to utilize skip connections:

```python
def forward(self, x: Tensor) -> Tensor:
    """Forward pass with skip connections and proper device handling."""
    # Get device from input and ensure model is on same device
    device = x.device
    
    # Initialize weights on first pass (existing code)
    # ...
    
    # Embedding and positional encoding (existing code)
    # ...
    
    # Execute active layers with skip connections
    execution_path = self.registry.get_active_forward_path()
    
    # Track layer outputs for skip connections
    layer_outputs = {}
    
    for layer_id in execution_path:
        layer_info = self.registry.layers[layer_id]
        if not layer_info['active']:
            continue
            
        # Ensure gate parameter is on correct device (existing code)
        # ...
            
        gate = torch.sigmoid(self.gating_params[layer_id])
        
        # Make sure module is on correct device (existing code)
        # ...
            
        layer_output = layer_info['module'](x)
        
        # Process skip connections targeting this layer
        skip_contribution = torch.zeros_like(x)
        for (source_id, target_id), connection in self.registry.skip_connections.items():
            if target_id == layer_id and source_id in layer_outputs and connection['active']:
                # Move connection parameters to current device if needed
                if connection['gate'].device != device:
                    connection['gate'] = connection['gate'].to(device)
                    connection['weight'] = connection['weight'].to(device)
                    connection['warmup_factor'] = connection['warmup_factor'].to(device)
                
                # Get skip connection parameters
                skip_gate = torch.sigmoid(connection['gate'])
                skip_weight = connection['weight']
                warmup = connection['warmup_factor']
                
                # Apply skip connection
                source_output = layer_outputs[source_id]
                skip_signal = F.linear(source_output, skip_weight)
                skip_contribution = skip_contribution + skip_gate * skip_signal * warmup
                
                # Update usage metrics
                connection['metrics']['usage'] = 0.9 * connection['metrics'].get('usage', 0.0) + 0.1
        
        # Add layer output and skip contribution
        x = x + gate * layer_output * layer_info['warmup_factor'] + skip_contribution
        
        # Store output for potential skip connections
        layer_outputs[layer_id] = x
        
        # Update usage metrics (existing code)
        # ...
    
    # Output projection (existing code)
    # ...
    return x
```

### 1.4 Integration with Training Process

Modify the `LMTrainingManager` in `main.py` to measure and update skip connection gradients:

```python
def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Training step with skip connection handling"""
    # Existing training step code...
    
    # After collecting gradients but before optimization
    gradients = self._collect_gradient_metrics()
    activations = self._collect_activation_metrics()
    
    # Update skip connections if controller exists
    if hasattr(self.network, 'skip_connection_controller'):
        skip_connections_added = self.network.skip_connection_controller.step(
            self.network.registry, self.network, gradients
        )
        if skip_connections_added > 0:
            logging.info(f"Added {skip_connections_added} new skip connections")
    
    # Continue with existing implementation...
```

## 2. Cross-Domain / Multi-Modal Handling

### 2.1 Multi-Modal Embedder Integration

Create a new `MultiModalEmbedder` class that can process different input modalities:

```python
class MultiModalEmbedder(nn.Module):
    """Embedder module that can process different types of inputs (text, image, audio)"""
    
    def __init__(self, modality_config: Dict[str, Dict], fusion_dim: int = 512, device=None):
        """
        Initialize embedder with modality-specific configurations
        
        Args:
            modality_config: Dictionary mapping modality names to their config dicts
                Example: {
                    'text': {'vocab_size': 50000, 'max_seq_len': 512, 'd_model': 256},
                    'image': {'image_size': 256, 'patch_size': 16, 'd_model': 256},
                    'audio': {'sample_rate': 16000, 'segment_ms': 1000, 'd_model': 256}
                }
            fusion_dim: Common dimension for all modality embeddings
            device: Target device
        """
        super().__init__()
        self.modality_config = modality_config
        self.fusion_dim = fusion_dim
        self.device = device
        
        # Create embedders for each modality
        self.embedders = nn.ModuleDict()
        for modality, config in modality_config.items():
            if modality == 'text':
                self.embedders[modality] = self._create_text_embedder(config)
            elif modality == 'image':
                self.embedders[modality] = self._create_image_embedder(config)
            elif modality == 'audio':
                self.embedders[modality] = self._create_audio_embedder(config)
        
        # Create projection layers for each modality
        self.projectors = nn.ModuleDict()
        for modality, config in modality_config.items():
            input_dim = config.get('d_model', 256)
            self.projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim)
            )
            
        # Attention fusion mechanism
        self.attention_layer = nn.MultiheadAttention(fusion_dim, 8)
        
        # Modality gates for dynamic weighting
        self.modality_gates = nn.ParameterDict({
            modality: nn.Parameter(torch.tensor(1.0))
            for modality in modality_config.keys()
        })
        
        if device is not None:
            self.to(device)
        
    def _create_text_embedder(self, config: Dict) -> nn.Module:
        """Create text embedding module"""
        vocab_size = config.get('vocab_size', 50000)
        d_model = config.get('d_model', 256)
        max_seq_len = config.get('max_seq_len', 512)
        
        embedding = nn.Embedding(vocab_size, d_model)
        pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Initialize weights
        with torch.no_grad():
            nn.init.normal_(embedding.weight, std=0.02)
            nn.init.normal_(pos_encoder, std=0.01)
        
        return nn.ModuleDict({
            'embedding': embedding,
            'pos_encoder': pos_encoder
        })
    
    def _create_image_embedder(self, config: Dict) -> nn.Module:
        """Create image embedding module (Vision Transformer style)"""
        image_size = config.get('image_size', 256)
        patch_size = config.get('patch_size', 16)
        d_model = config.get('d_model', 256)
        
        # Simple ViT-style embedder
        num_patches = (image_size // patch_size) ** 2
        
        return nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Transpose(1, 2),  # (B, num_patches, d_model)
            nn.LayerNorm(d_model)
        )
    
    def _create_audio_embedder(self, config: Dict) -> nn.Module:
        """Create audio embedding module"""
        d_model = config.get('d_model', 256)
        
        # Use 1D convolutions for audio spectrogram
        return nn.Sequential(
            nn.Conv1d(40, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),  # Fixed sequence length
            nn.Flatten(1, -1),
            nn.Linear(32 * d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def embed_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Embed text tokens"""
        embedder = self.embedders['text']
        seq_len = text_tokens.size(1)
        
        # Get embeddings and add positional encoding
        x = embedder['embedding'](text_tokens) * (self.modality_config['text']['d_model'] ** 0.5)
        x = x + embedder['pos_encoder'][:, :seq_len, :]
        
        return x
    
    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        """Embed images"""
        return self.embedders['image'](images)
    
    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Embed audio"""
        return self.embedders['audio'](audio)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multi-modal inputs
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
                Example: {
                    'text': tensor of shape [batch_size, seq_len] (token ids),
                    'image': tensor of shape [batch_size, 3, H, W] (images),
                    'audio': tensor of shape [batch_size, 40, T] (mel spectrograms)
                }
        
        Returns:
            Fused representation tensor of shape [batch_size, seq_len, fusion_dim]
        """
        # Process each modality
        embedded_modalities = {}
        for modality, input_tensor in inputs.items():
            if modality not in self.embedders:
                continue
                
            # Move tensor to device if needed
            if input_tensor.device != self.device and self.device is not None:
                input_tensor = input_tensor.to(self.device)
                
            # Embed using modality-specific embedder
            if modality == 'text':
                embedded = self.embed_text(input_tensor)
            elif modality == 'image':
                embedded = self.embed_image(input_tensor)
            elif modality == 'audio':
                embedded = self.embed_audio(input_tensor)
            else:
                continue
                
            # Project to common fusion dimension
            projected = self.projectors[modality](embedded)
            embedded_modalities[modality] = projected
            
        if not embedded_modalities:
            raise ValueError("No valid modalities provided in input")
            
        # Apply modality gating
        gated_embeddings = []
        for modality, embedding in embedded_modalities.items():
            gate = torch.sigmoid(self.modality_gates[modality])
            gated_embeddings.append(embedding * gate)
            
        # Fuse modalities with attention
        if len(gated_embeddings) > 1:
            # Create sequence from all modality embeddings
            stacked = []
            
            # Handle varying sequence lengths by padding
            max_seq_len = max(emb.size(1) for emb in gated_embeddings)
            for emb in gated_embeddings:
                # Pad if necessary
                if emb.size(1) < max_seq_len:
                    padding = torch.zeros(
                        emb.size(0), max_seq_len - emb.size(1), emb.size(2),
                        device=emb.device
                    )
                    emb = torch.cat([emb, padding], dim=1)
                stacked.append(emb)
                
            # Concatenate along sequence dimension
            seq = torch.cat(stacked, dim=1)
            
            # Self-attention for cross-modal fusion
            attn_output, _ = self.attention_layer(seq, seq, seq)
            
            # Restore to single sequence representation (average or first modality)
            if 'text' in embedded_modalities:
                # Use text sequence length if available
                text_seq_len = embedded_modalities['text'].size(1)
                fused = torch.cat([
                    embedded_modalities['text'],
                    attn_output[:, text_seq_len:, :]
                ], dim=1)
            else:
                # Use first modality's sequence length
                first_modality = list(embedded_modalities.keys())[0]
                first_seq_len = embedded_modalities[first_modality].size(1)
                fused = attn_output[:, :first_seq_len, :]
        else:
            # Only one modality, use it directly
            fused = list(embedded_modalities.values())[0]
            
        return fused
```

### 2.2 Abstract Concept Embedding Implementation

Create a new `ConceptEmbeddingLayer` class that maps inputs to a shared concept space:

```python
class ConceptEmbeddingLayer(nn.Module):
    """Layer that maintains an abstract concept space for cross-domain knowledge"""
    
    def __init__(self, input_dim: int, concept_dim: int = 256, 
                num_concepts: int = 100, update_rate: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.update_rate = update_rate
        
        # Initialize concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, concept_dim) * 0.02
        )
        
        # Projection layers
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.LayerNorm(concept_dim)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(concept_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Attention mechanism for concept retrieval
        self.concept_attention = nn.Linear(concept_dim, num_concepts)
        
        # Tracking usage and domains
        self.register_buffer('concept_usage', torch.zeros(num_concepts))
        self.register_buffer('concept_last_domain', torch.zeros(num_concepts, dtype=torch.long))
        
        # Residual connection gate
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, domain_id: int = 0, 
               update_concepts: bool = True) -> torch.Tensor:
        """
        Map input through concept space
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            domain_id: ID of the current domain
            update_concepts: Whether to update concepts (during training)
            
        Returns:
            Output with same shape as input, enriched with concept knowledge
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to concept dimension
        projected = self.input_projection(x)  # [B, S, concept_dim]
        
        # Calculate attention scores for each concept
        attention_logits = self.concept_attention(projected)  # [B, S, num_concepts]
        attention_weights = F.softmax(attention_logits, dim=-1)  # [B, S, num_concepts]
        
        # Retrieve weighted concepts
        # Reshape for matrix multiplication
        attention_weights_reshaped = attention_weights.view(batch_size * seq_len, self.num_concepts)
        concepts_expanded = self.concept_embeddings.unsqueeze(0).expand(
            batch_size * seq_len, self.num_concepts, self.concept_dim
        )
        
        weighted_concepts = torch.bmm(
            attention_weights_reshaped.unsqueeze(1),
            concepts_expanded
        ).squeeze(1)  # [B*S, concept_dim]
        
        # Reshape back to sequence format
        weighted_concepts = weighted_concepts.view(batch_size, seq_len, self.concept_dim)
        
        # Project back to input dimension
        output = self.output_projection(weighted_concepts)  # [B, S, input_dim]
        
        # Apply residual connection
        residual_scale = torch.sigmoid(self.residual_gate)
        output = (1 - residual_scale) * x + residual_scale * output
        
        # Update concepts during training
        if self.training and update_concepts:
            with torch.no_grad():
                # Get mean attention weights across batch and sequence
                mean_attn = attention_weights.mean(dim=[0, 1])  # [num_concepts]
                
                # Track usage
                self.concept_usage = 0.9 * self.concept_usage + 0.1 * mean_attn
                
                # Find top concepts for this input
                batch_top_concepts = attention_weights.topk(3, dim=-1).indices  # [B, S, 3]
                
                # Update concept embeddings
                for b in range(min(batch_size, 4)):  # Limit updates to first 4 batch items
                    for s in range(min(seq_len, 4)):  # And first 4 sequence positions
                        input_embedding = projected[b, s]
                        for c in range(3):  # Update top 3 concepts
                            concept_idx = batch_top_concepts[b, s, c].item()
                            
                            # Check when this concept was last updated
                            if self.concept_last_domain[concept_idx] != domain_id:
                                # Stronger update for new domains
                                update_strength = self.update_rate * 2.0
                            else:
                                update_strength = self.update_rate
                                
                            # Move concept closer to this input embedding
                            self.concept_embeddings.data[concept_idx] += (
                                update_strength * 
                                (input_embedding - self.concept_embeddings.data[concept_idx])
                            )
                            
                            # Record domain
                            self.concept_last_domain[concept_idx] = domain_id
        
        return output
    
    def prune_unused_concepts(self, threshold: float = 0.01):
        """Replace rarely used concepts with new ones"""
        if not self.training:
            return 0
            
        with torch.no_grad():
            # Normalize usage
            usage_ratio = self.concept_usage / (self.concept_usage.sum() + 1e-12)
            
            # Find unused concepts
            unused_mask = (usage_ratio < threshold)
            num_unused = unused_mask.sum().item()
            
            if num_unused > 0:
                # Reset unused concepts
                self.concept_embeddings.data[unused_mask] = torch.randn(
                    num_unused, self.concept_dim, 
                    device=self.concept_embeddings.device
                ) * 0.02
                
                # Reset tracking
                self.concept_usage[unused_mask] = torch.ones_like(
                    self.concept_usage[unused_mask]
                ) * 0.1  # Initialize with small usage to promote exploration
                
                self.concept_last_domain[unused_mask] = 0
                
            return num_unused
```

### 2.3 Domain Expert Controller

Create a `DomainExpertController` that can route inputs to specialized subnetworks:

```python
class DomainExpertController(nn.Module):
    """Controller that routes inputs to specialized expert subnetworks"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                num_experts: int = 3, expert_capacity: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Create expert networks
        self.experts = nn.ModuleList([
            self._create_expert(input_dim, output_dim, expert_capacity)
            for _ in range(num_experts)
        ])
        
        # Router network for determining which expert to use
        self.router = nn.Sequential(
            nn.Linear(input_dim, expert_capacity),
            nn.LayerNorm(expert_capacity),
            nn.GELU(),
            nn.Linear(expert_capacity, num_experts)
        )
        
        # Expert usage tracking
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('expert_capacity_factor', torch.ones(num_experts))
        
        # Storage for expert gradients
        self.expert_grads = [0.0] * num_experts
        
        # Auxiliary loss weight
        self.load_balancing_loss_weight = 0.01
        
    def _create_expert(self, input_dim: int, output_dim: int, capacity: int) -> nn.Module:
        """Create a new expert module"""
        return nn.Sequential(
            nn.Linear(input_dim, capacity),
            nn.LayerNorm(capacity),
            nn.GELU(),
            nn.Linear(capacity, capacity),
            nn.LayerNorm(capacity),
            nn.GELU(),
            nn.Linear(capacity, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor, 
               return_auxiliary_loss: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process input through expert routing
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_auxiliary_loss: Whether to return the load balancing loss
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
            Optional auxiliary load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for routing
        x_flat = x.view(-1, self.input_dim)  # [batch_size * seq_len, input_dim]
        
        # Get routing probabilities
        routing_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Get dispatching weights with capacity constraints
        # We select the top-1 expert per token
        expert_indices = torch.argmax(routing_probs, dim=-1)  # [batch_size * seq_len]
        
        # Create one-hot mask based on expert assignment
        mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [BS*S, num_experts]
        
        # Process through experts
        final_output = torch.zeros(
            batch_size * seq_len, self.output_dim,
            device=x.device
        )
        
        # Process each expert separately
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Select tokens routed to this expert
            expert_mask = (expert_indices == i)
            if expert_mask.sum() == 0:
                # No tokens for this expert
                expert_outputs.append(None)
                continue
                
            # Get expert inputs
            expert_inputs = x_flat[expert_mask]
            
            # Process through expert
            expert_out = expert(expert_inputs)
            expert_outputs.append(expert_out)
            
            # Add to final output
            final_output[expert_mask] = expert_out
            
            # Track expert usage
            if self.training:
                usage = expert_mask.float().mean().item()
                self.expert_usage[i] = 0.9 * self.expert_usage[i] + 0.1 * usage
        
        # Reshape output back to [batch_size, seq_len, output_dim]
        final_output = final_output.view(batch_size, seq_len, self.output_dim)
        
        # Calculate load balancing auxiliary loss
        if return_auxiliary_loss and self.training:
            # We want experts to be used equally
            # 1. Calculate fraction of tokens routed to each expert
            route_frac = mask.mean(dim=0)  # [num_experts]
            
            # 2. Calculate fraction of router probability assigned to each expert
            prob_frac = routing_probs.mean(dim=0)  # [num_experts]
            
            # 3. Calculate auxiliary loss
            aux_loss = self.load_balancing_loss_weight * (route_frac * prob_frac).sum() * self.num_experts
            
            return final_output, aux_loss
        
        return final_output, None
        
    def update_expert_gradients(self):
        """Track expert parameter gradients during training"""
        if not self.training:
            return
            
        for i, expert in enumerate(self.experts):
            # Calculate average gradient magnitude
            grad_sum = 0.0
            grad_count = 0
            for param in expert.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().mean().item()
                    grad_count += 1
                    
            if grad_count > 0:
                self.expert_grads[i] = grad_sum / grad_count
        
    def spawn_new_expert(self):
        """Add a new expert if needed"""
        if not self.training or self.num_experts >= 10:
            return False
            
        # Check if we need a new expert based on usage imbalance
        usage = self.expert_usage.tolist()
        max_usage = max(usage)
        min_usage = min(usage)
        
        if max_usage > 0.4 and max_usage / (min_usage + 1e-10) > 3.0:
            # Significant imbalance, spawn a new expert
            
            # 1. Create new expert
            new_expert = self._create_expert(self.input_dim, self.output_dim, self.expert_capacity)
            
            # 2. Add to experts list
            self.experts.append(new_expert)
            
            # 3. Expand tracking buffers
            device = self.expert_usage.device
            self.expert_usage = torch.cat([
                self.expert_usage,
                torch.ones(1, device=device) * 0.1  # Initialize with small usage
            ])
            self.expert_capacity_factor = torch.cat([
                self.expert_capacity_factor,
                torch.ones(1, device=device)
            ])
            self.expert_grads.append(0.0)
            
            # 4. Update router output size
            old_router_final = self.router[-1]
            new_router_final = nn.Linear(
                old_router_final.in_features,
                self.num_experts + 1,
                device=device
            )
            
            # Copy existing weights
            with torch.no_grad():
                new_router_final.weight.data[:-1] = old_router_final.weight.data
                new_router_final.bias.data[:-1] = old_router_final.bias.data
                
                # Initialize new weights
                nn.init.normal_(new_router_final.weight.data[-1:], std=0.02)
                nn.init.zeros_(new_router_final.bias.data[-1:])
                
            # Replace router final layer
            self.router[-1] = new_router_final
            
            # 5. Update expert count
            self.num_experts += 1
            
            return True
            
        return False
        
    def remove_underutilized_expert(self):
        """Remove an expert if it's not being used"""
        if not self.training or self.num_experts <= 2:
            return False
            
        # Find least used expert
        usage = self.expert_usage.tolist()
        min_idx = usage.index(min(usage))
        
        # Check if it's significantly underutilized
        if usage[min_idx] < 0.01:
            # Very low usage, remove this expert
            
            # 1. Remove from experts list
            self.experts = nn.ModuleList([
                expert for i, expert in enumerate(self.experts)
                if i != min_idx
            ])
            
            # 2. Shrink tracking buffers
            mask = torch.ones_like(self.expert_usage, dtype=torch.bool)
            mask[min_idx] = False
            self.expert_usage = self.expert_usage[mask]
            self.expert_capacity_factor = self.expert_capacity_factor[mask]
            self.expert_grads = [
                grad for i, grad in enumerate(self.expert_grads)
                if i != min_idx
            ]
            
            # 3. Update router output size
            old_router_final = self.router[-1]
            new_router_final = nn.Linear(
                old_router_final.in_features,
                self.num_experts - 1,
                device=self.expert_usage.device
            )
            
            # Copy weights, skipping the removed expert
            with torch.no_grad():
                new_weights = []
                new_biases = []
                for i in range(self.num_experts):
                    if i != min_idx:
                        new_weights.append(old_router_final.weight.data[i])
                        new_biases.append(old_router_final.bias.data[i])
                        
                new_router_final.weight.data = torch.stack(new_weights)
                new_router_final.bias.data = torch.stack(new_biases)
                
            # Replace router final layer
            self.router[-1] = new_router_final
            
            # 4. Update expert count
            self.num_experts -= 1
            
            return True
            
        return False
```

### 2.4 Integration with Adaptive Transformer

Modify the `AdaptiveTransformer` class to support multi-modal inputs:

```python
class AdaptiveTransformer(nn.Module):
    """Enhanced transformer model with multi-modal and cross-domain capabilities"""
    
    def __init__(self, vocab_size: int = 50000, d_model: int = 1024, 
                nhead: int = 16, num_layers: int = 24, d_ff: int = 4096, 
                max_seq_len: int = 512, dropout: float = 0.1, 
                modality_config: Optional[Dict] = None,
                device=None):
        super().__init__()
        
        self.d_model = d_model
        self.registry = GlobalLayerRegistry()
        self.meta_controller = MetaController(max_layers=30)
        self.device = device or torch.device('cpu')
        self.max_seq_len = max_seq_len
        
        # Handle multi-modal inputs if config provided
        self.is_multimodal = modality_config is not None
        if self.is_multimodal:
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
                self.embedding.weight.data.uniform_(-0.1, 0.1)
                
            self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            
        # Skip connection controller
        self.skip_connection_controller = SkipConnectionController()
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Gating parameters
        self.gating_params = nn.ParameterDict()
        
        # Initialize transformer layers
        for i in range(num_layers):
            layer = AdaptiveTransformerLayer(d_model, nhead, d_ff, dropout)
            self.add_module(f"layer_{i}", layer)
            layer_id = self.registry.add_layer(
                layer, 'transformer', d_model, d_model, position=i
            )
            self.gating_params[layer_id] = nn.Parameter(torch.tensor(0.0).to(device))
            
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        if device is not None:
            self.to(device)
            
    def forward(self, x, modality_inputs=None, domain_id=0):
        """
        Enhanced forward pass with multi-modal support
        
        Args:
            x: For text-only mode, token IDs [batch_size, seq_len]
               For multi-modal mode, can be None if modality_inputs is provided
            modality_inputs: Optional dict of modality inputs for multi-modal mode
                Example: {'text': text_tokens, 'image': images}
            domain_id: Domain identifier for concept updates
            
        Returns:
            Output tensor with vocabulary logits
        """
        # Handle multi-modal inputs
        if self.is_multimodal and modality_inputs is not None:
            # Process through modality embedder
            embeddings = self.modality_embedder(modality_inputs)
            
            # Process through concept layer
            embeddings = self.concept_embedding(embeddings, domain_id)
            
            # Process through domain experts
            embeddings, aux_loss = self.domain_controller(embeddings)
            
            # Apply dropout
            x = self.dropout(embeddings)
        else:
            # Standard text embedding
            device = x.device if hasattr(x, 'device') else self.device
            
            # Ensure model parameters are on the same device
            if next(self.parameters()).device != device:
                self.to(device)
                
            seq_len = x.size(1)
            
            # Text embedding with positional encoding
            x = self.embedding(x) * (self.d_model ** 0.5) + self.pos_encoder[:, :seq_len, :]
            x = self.dropout(x)
            
        # Process through transformer layers with skip connections
        # (rest of the forward pass implementation follows the skip connection implementation
        # described in the previous section)
        
        execution_path = self.registry.get_active_forward_path()
        layer_outputs = {}  # Track outputs for skip connections
        
        for layer_id in execution_path:
            layer_info = self.registry.layers[layer_id]
            if not layer_info['active']:
                continue
                
            # (Device transfer code and core layer execution)
            
            # Process skip connections (as in previous section)
            # ...
            
            # Track intermediate activations for skip connection analysis
            if self.training and hasattr(self, 'skip_connection_controller'):
                device_x = x.detach().to(self.device)
                self.skip_connection_controller.update_activations(layer_id, device_x)
        
        # Output projection
        x = self.output_layer(x)
        
        return x
        
    def meta_step(self, val_loss: float):
        """Perform meta-learning steps for all adaptive components"""
        # Original meta-controller step
        add_layer, remove_layer = self.meta_controller.analyze_performance(self.registry, val_loss)
        
        # (Layer addition/removal logic)
        
        # Additional expert management for multi-modal mode
        if self.is_multimodal and self.training:
            # Expert spawn/remove logic
            if val_loss < 3.0:  # Only after some training
                if np.random.random() < 0.1:  # 10% chance to try
                    self.domain_controller.spawn_new_expert()
                    
            if val_loss < 2.0:  # Only after more training
                if np.random.random() < 0.05:  # 5% chance to try
                    self.domain_controller.remove_underutilized_expert()
                    
            # Concept pruning logic
            if np.random.random() < 0.2:  # 20% chance each meta-step
                self.concept_embedding.prune_unused_concepts(threshold=0.01)
```

### 2.5 Training Extensions for Multi-Modal Support

Update the training manager to handle multi-modal inputs:

```python
# Within the LMTrainingManager class:

def train_step(self, batch: Union[dict, Tuple]):
    """Enhanced training step with multi-modal support"""
    self.network.train()
    self.optimizer.zero_grad(set_to_none=True)
    
    # Handle multi-modal batches
    modality_inputs = None
    targets = None
    domain_id = 0
    
    if isinstance(batch, dict):
        if 'input_ids' in batch:
            # Standard text-only batch
            input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
            
            # Use autoregressive language modeling targets
            targets = input_ids[:, 1:]
            
        elif any(k in {'text', 'image', 'audio'} for k in batch.keys()):
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
    else:
        # Assume text-only tensor input
        input_ids = batch.to(self.device, dtype=torch.long)
        targets = input_ids[:, 1:]
    
    # Forward pass
    if modality_inputs is not None:
        outputs = self.network(None, modality_inputs, domain_id)
        
        # For multi-modal, we might have an auxiliary loss
        auxiliary_loss = 0.0
        if hasattr(self.network, 'domain_controller'):
            _, aux_loss = self.network.domain_controller(
                self.network.modality_embedder(modality_inputs)
            )
            if aux_loss is not None:
                auxiliary_loss = aux_loss
    else:
        # Text-only
        outputs = self.network(input_ids)
    
    # Compute loss (same as before, just ensuring we handle targets correctly)
    logits = outputs[:, :-1, :].contiguous() if outputs.size(1) > 1 else outputs
    
    if targets is None:
        # If no targets (e.g., image-only input), create synthetic targets
        targets = torch.zeros(
            outputs.size(0), outputs.size(1) - 1, dtype=torch.long,
            device=outputs.device
        )
    
    # Compute loss
    batch_size, seq_len, vocab_size = logits.size()
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)
    
    loss = self.criterion(logits_flat, targets_flat)
    
    # Add auxiliary loss if present
    if 'auxiliary_loss' in locals() and auxiliary_loss != 0.0:
        loss = loss + auxiliary_loss
    
    # Backward and optimizer steps
    loss.backward()
    
    # Collect metrics
    gradients = self._collect_gradient_metrics()
    activations = self._collect_activation_metrics()
    
    # Update domain controller gradients if present
    if hasattr(self.network, 'domain_controller'):
        self.network.domain_controller.update_expert_gradients()
    
    # Update skip connections if controller exists
    if hasattr(self.network, 'skip_connection_controller'):
        skip_connections_added = self.network.skip_connection_controller.step(
            self.network.registry, self.network, gradients
        )
    
    # Apply gradient clipping and optimization
    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
    self.optimizer.step()
    
    # Update network metrics
    self.network.update_metrics(gradients, activations)
    self.network.increment_warmup()
    
    # Return metrics
    return {
        'loss': loss.item(),
        'auxiliary_loss': auxiliary_loss if 'auxiliary_loss' in locals() else 0.0,
    }
```

## 3. Comprehensive Implementation Strategy

### 3.1 Phased Implementation Approach

To implement these changes safely while maintaining full compatibility with the existing codebase, we'll use a phased approach:

1. **Phase 1: Skip Connection Architecture**
   - Extend GlobalLayerRegistry with skip connection support
   - Implement SkipConnectionController
   - Update AdaptiveTransformer forward pass
   - Add skip connection tracking to TrainingManager
   - Test with text-only inputs

2. **Phase 2: Multi-Modal Integration**
   - Implement MultiModalEmbedder
   - Add ConceptEmbeddingLayer
   - Implement device handling for multi-modal inputs
   - Update inference code to handle multi-modal inputs
   - Test with basic multi-modal inputs

3. **Phase 3: Domain Expert Integration**
   - Implement DomainExpertController
   - Add domain-specific routing
   - Update training code for domain experts
   - Test with mixed domain inputs

### 3.2 Compatibility with Existing Features

Special attention needed for:

1. **Device Handling**: All new components must check tensor device placement and ensure consistent device usage across different hardware (CPU, MPS, CUDA)

2. **Memory Efficiency**: For multi-modal inputs, use gradient checkpointing and optimize tensor operations to avoid OOM errors

3. **Backward Compatibility**: All modified classes should support the existing API to maintain compatibility with existing code

4. **Training Stability**: Ensure warmup mechanics are applied to new components (skip connections, experts, etc.) to prevent instability

### 3.3 Testing and Validation Strategy

1. **Component Tests**: Individual tests for each new class to verify expected behavior
2. **Integration Tests**: Tests for interactions between components
3. **Regression Tests**: Ensure existing functionality continues to work
4. **Multi-Modal Tests**: Verify handling of different input modalities
5. **Device Tests**: Test across different hardware configurations
6. **End-to-End Tests**: Validate the complete pipeline

This implementation will provide a comprehensive solution for evolving adaptive connectivity and cross-domain handling while maintaining compatibility with all existing features in the adaptive network architecture.