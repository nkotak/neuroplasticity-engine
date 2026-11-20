# transformer_model/domain_expert_controller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging

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