# transformer_model/multi_modal_embedder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging

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
        
        # Create a custom text embedder module
        class TextEmbedder(nn.Module):
            def __init__(self, vocab_size, d_model, max_seq_len):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
                
                # Initialize weights
                with torch.no_grad():
                    nn.init.normal_(self.embedding.weight, std=0.02)
                    nn.init.normal_(self.pos_encoder, std=0.01)
                    
            def forward(self, x):
                seq_len = x.size(1)
                embedded = self.embedding(x) * (d_model ** 0.5)
                embedded = embedded + self.pos_encoder[:, :seq_len, :]
                return embedded
                
        return TextEmbedder(vocab_size, d_model, max_seq_len)
    
    def _create_image_embedder(self, config: Dict) -> nn.Module:
        """Create image embedding module (Vision Transformer style)"""
        image_size = config.get('image_size', 256)
        patch_size = config.get('patch_size', 16)
        d_model = config.get('d_model', 256)
        
        # Simple ViT-style embedder
        num_patches = (image_size // patch_size) ** 2
        
        class Permute(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims
                
            def forward(self, x):
                return x.permute(*self.dims)
                
        return nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Permute((0, 2, 1)),  # (B, num_patches, d_model)
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
        # Our TextEmbedder now handles the embedding and positional encoding
        return embedder(text_tokens)
    
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