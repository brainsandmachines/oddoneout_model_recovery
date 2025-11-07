"""
Similarity Functions Module

This module provides various similarity functions for measuring the relationship between feature vectors.
These functions are used in triplet-based models to compute similarity scores between elements.

Functions include:
- dot_product: Standard dot product similarity
- cosine_similarity: Cosine similarity (normalized dot product)
- euclidean_distance: Euclidean distance between vectors
"""

import torch
import math
from utils import calculate_batch_size

def dot_product(x, y = None):
    """
    Compute dot product between two sets of vectors.
    
    This function handles different input shapes and automatically determines
    the appropriate way to perform the dot product operation.
    
    Args:
        x (torch.Tensor): First tensor of vectors
        y (torch.Tensor, optional): Second tensor of vectors. If None, dot product is computed
                                    between x and itself. Defaults to None.
    
    Returns:
        torch.Tensor: Dot product result
        
    Raises:
        ValueError: If tensor dimensions don't align properly for dot product
    """
    #check if the dimensions fit by
    if y is None:
        y = x
    if len(x.shape) > 2:
        return torch.matmul(x, y.transpose(1, 2))
    elif x.shape[-1] == y.shape[0]:
        return torch.matmul(x, y)
    elif x.shape[-1] == y.shape[-1]:
        return torch.matmul(x, y.T)
    
    raise ValueError(
            f"Dimensions do not align: "
            f"x has shape {x.shape}, y has shape {y.shape}."
        )

def cosine_similarity(x, y = None):
    """
    Compute cosine similarity between two sets of vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    which is equivalent to the normalized dot product.
    
    This function handles both 2D tensors (matrices) and 3D tensors (batched matrices),
    automatically detecting the appropriate computation method.
    
    Args:
        x (torch.Tensor): First tensor of vectors
        y (torch.Tensor, optional): Second tensor of vectors. If None, cosine similarity
                                    is computed between x and itself. Defaults to None.
    
    Returns:
        torch.Tensor: Cosine similarity result
    """
    # Handle batched triplet case (3D tensors)
    if len(x.shape) == 3:
        # x shape: (batch_size, 3, feature_dim)
        # y shape: (batch_size, 3, feature_dim)
        dot_sim = torch.bmm(x, y.transpose(1, 2))  # (batch_size, 3, 3)
        x_norm = x.norm(dim=2, keepdim=True)  # (batch_size, 3, 1)
        y_norm = y.norm(dim=2, keepdim=True)  # (batch_size, 3, 1)
        return dot_sim / (x_norm @ y_norm.transpose(1, 2))
    
    dot_similarity = dot_product(x, y)
    if y is None:
        y = x  
    if x.shape[1] == y.shape[1]:
        return dot_similarity / (x.norm(dim=1)[:, None] * y.norm(dim=1)[None, :])
    else:
        return dot_similarity / (x.norm(dim=1)[:, None] * y.T.norm(dim=1)[None, :])

def euclidean_distance(x, y = None):
    """
    Compute Euclidean distance between two sets of vectors.
    
    Args:
        x (torch.Tensor): First 2D tensor of vectors
        y (torch.Tensor, optional): Second 2D tensor of vectors. If None, euclidean distance
                                    is computed between x and itself. Defaults to None.
    
    Returns:
        torch.Tensor: Euclidean distance result
        
    Raises:
        AssertionError: If x or y are not 2D tensors
    """
    assert len(x.shape) == 2, "x must be a 2D tensor"
    if y is None:
        y = x
    else:
        assert len(y.shape) == 2, "y must be a 2D tensor"
    return torch.norm(x - y, dim=1)

