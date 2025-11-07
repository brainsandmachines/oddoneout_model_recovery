"""
Loss Functions Module for Triplet-Based Models

This module provides custom loss functions specialized for triplet-based learning, 
particularly focusing on cross-entropy losses for multi-class classification
from similarity scores.

These loss functions are designed to be used with models that compute similarity
between elements in triplets.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Custom implementation of cross-entropy loss for triplet-based models.
    
    This class implements a cross-entropy loss that works directly with
    similarity scores, handling the softmax and log operations manually
    for enhanced numerical stability.
    """
    def __init__(self) -> None:
        """Initialize the TripletLoss."""
        super(TripletLoss, self).__init__()

    def logsumexp(self, similarity_scores: Tensor) -> Tensor:
        """
        Compute log(sum(exp(x))) in a numerically stable way.
        
        Uses the logsumexp trick: log(sum(exp(x))) = log(sum(exp(x-c))) + c
        where c is the max of x to prevent overflow.
        
        Args:
            similarity_scores (Tensor): Similarity scores for triplets
            
        Returns:
            Tensor: Computed logsumexp values
        """
        #the logsumexp trick for numerical stability:
        #log(sum(exp(x))) = log(sum(exp(x-c))) + c
        #where c is the max of the x
        c = similarity_scores.max(dim=1, keepdim=True).values
        return torch.log(torch.sum(torch.exp(similarity_scores-c), dim=1)) + c.squeeze()
    
    def log_softmax(self, similarity_scores: Tensor, correct_similarity_scores: Tensor) -> Tensor:
        """
        Compute log softmax for similarity scores.
        
        Args:
            similarity_scores (Tensor): All similarity scores
            correct_similarity_scores (Tensor): Scores for correct choices
            
        Returns:
            Tensor: Log softmax values
        """
        return correct_similarity_scores - self.logsumexp(similarity_scores)

    def cross_entropy_loss(self, similarity_scores: Tensor, correct_similarity_scores: Tensor) -> Tensor:
        """
        Compute cross entropy loss using log softmax.
        
        Args:
            similarity_scores (Tensor): All similarity scores
            correct_similarity_scores (Tensor): Scores for correct choices
            
        Returns:
            Tensor: Cross entropy loss value
        """
        return torch.mean(-self.log_softmax(similarity_scores, correct_similarity_scores))

    def forward(self, similarity_scores: Tensor, correct_similarity_scores: Tensor): 
        """
        Forward pass for the loss function.
        
        Args:
            similarity_scores (Tensor): All similarity scores
            correct_similarity_scores (Tensor): Scores for correct choices
            
        Returns:
            Tensor: Computed loss value
        """
        loss = self.cross_entropy_loss(similarity_scores, correct_similarity_scores)
        return loss

class Triplets_cross_entropy_loss(nn.Module):
    """
    Cross-entropy loss for triplet models supporting both index-based and one-hot encoded targets.
    
    This loss function handles two different target formats:
    1. Index-based: where the target is the index of the correct choice
    2. One-hot encoded: where the target is a one-hot vector or a distribution of choices
    """
    def __init__(self) -> None:
        """Initialize the Triplets_cross_entropy_loss."""
        super(Triplets_cross_entropy_loss, self).__init__()

    def forward(self, similarity_scores: Tensor, y: Tensor):
        """
        Forward pass for the loss function.
        
        Args:
            similarity_scores (Tensor): Similarity scores for triplets (BxC tensor)
            y (Tensor): Target values, either indices (1D tensor) or one-hot encoded (2D tensor)
            
        Returns:
            Tensor: Computed loss value
            
        Raises:
            ValueError: If y has invalid shape
        """
        #if y is not hot encoded we use torch.nn.CrossEntropyLoss
        if len(y.shape) == 1:
            loss = torch.nn.CrossEntropyLoss()(similarity_scores, y)
        #if y is hot encoded we use the log_softmax
        elif len(y.shape) == 2:
            log_probs = F.log_softmax(similarity_scores, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
        else:
            raise ValueError(f"y must be a positions(1D tensor) or one hot encoded of choices or choices counts(2D tensor), got {y.shape}")
        return loss