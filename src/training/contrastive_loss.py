"""
Contrastive Alignment Loss for Cross-Event Generalization
Pulls same-veracity claims from different events together
Pushes different-veracity claims apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive loss that aligns representations across different events
    while preserving veracity label information.
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        same_event_weight: Weight for same-event pairs (default: 0.0)
            (we want to ignore same-event pairs, focus on cross-event)
    """
    
    def __init__(self, temperature=0.07, same_event_weight=0.0):
        super().__init__()
        self.temperature = temperature
        self.same_event_weight = same_event_weight
    
    def forward(self, representations, labels, event_ids):
        """
        Args:
            representations: [batch_size, hidden_dim] - model embeddings
            labels: [batch_size] - veracity labels (0,1,2)
            event_ids: [batch_size] - event identifiers (strings)
        
        Returns:
            contrastive_loss: scalar tensor
        """
        # Normalize representations
        reps = F.normalize(representations, dim=1)
        
        # Compute similarity matrix
        sim_matrix = reps @ reps.T / self.temperature
        
        batch_size = len(labels)
        device = representations.device
        
        # Convert event_ids to indices for tensor operations
        unique_events = {e: i for i, e in enumerate(set(event_ids))}
        event_indices = torch.tensor([unique_events[e] for e in event_ids]).to(device)
        
        # Create masks
        # Positive: same label, DIFFERENT event
        same_label = labels.unsqueeze(1) == labels.unsqueeze(0)
        different_event = event_indices.unsqueeze(1) != event_indices.unsqueeze(0)
        positive_mask = same_label & different_event
        
        # Negative: different label
        negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)
        
        # Zero out diagonal (no self-positive)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~eye_mask
        
        # Compute loss
        total_loss = 0.0
        num_positives = 0
        
        for i in range(batch_size):
            # Positive pairs for sample i
            pos_indices = positive_mask[i]
            if pos_indices.sum() == 0:
                continue
            
            # Negative pairs for sample i
            neg_indices = negative_mask[i]
            
            # Positive similarities
            pos_sim = sim_matrix[i, pos_indices]
            
            # Negative similarities
            neg_sim = sim_matrix[i, neg_indices]
            
            # Contrastive loss (InfoNCE style)
            # L = -log( sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))) )
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)
            
            loss_i = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))
            total_loss += loss_i
            num_positives += 1
        
        if num_positives > 0:
            return total_loss / num_positives
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class SimpleContrastiveLoss(nn.Module):
    """
    Simplified version - assumes we have anchor, positive, negative triplets
    Useful when we pre-compute positive/negative pairs.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Triplet loss: pull anchor and positive together,
        push anchor and negative apart.
        """
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(pos_distance - neg_distance + self.margin))
        return loss


# For testing
if __name__ == "__main__":
    # Test the contrastive loss
    batch_size = 8
    hidden_dim = 768
    
    reps = torch.randn(batch_size, hidden_dim)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 2, 2])
    event_ids = ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B']
    
    loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    loss = loss_fn(reps, labels, event_ids)
    
    print(f"Test contrastive loss: {loss.item():.4f}")