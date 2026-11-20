# transformer_model/concept_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging

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