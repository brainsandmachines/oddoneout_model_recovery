"""
Constraints Module for PyTorch Neural Networks

This module provides tools for applying various constraints to neural network parameters, 
including orthogonality, diagonal matrices, and others. It uses the GeoTorch library to enforce 
manifold constraints on parameters and PyTorch's parametrization utilities for other constraints.

The module contains:
1. Custom layer implementations with built-in constraints
2. Matrix manipulation utilities
3. Parametrization classes for different constraint types
4. A unified wrapper function for applying constraints
"""

import geotorch
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from typing import Optional
import torch.nn.functional as F
from torch.nn import init

class OrthogonalScaledLinear(nn.Module):
    """
    Linear layer with orthogonal weight matrix multiplied by a positive diagonal scaling matrix.
    
    This implements a transformation of the form W = A @ D where:
    - A is constrained to be an orthogonal (orthonormal) matrix
    - D is constrained to be a positive diagonal matrix
    
    This parameterization allows the model to scale features differently while preserving
    orthogonality properties.
    
    Attributes:
        A (nn.Parameter): Orthogonal matrix parameter
        D (nn.Parameter): Diagonal scaling parameter (constrained to be positive)
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Initialize the OrthogonalScaledLinear layer.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            device (torch.device, optional): Device to place tensors on
            dtype (torch.dtype, optional): Data type of the parameters
        
        Raises:
            AssertionError: If in_features != out_features (layer requires square matrices)
        """
        super().__init__()
        assert in_features == out_features, "in_features must be equal to out_features for the OrthogonalScaledLinear layer"
        self.A = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        init.eye_(self.A)
        # orthnormal\isometric matrix 
        geotorch.orthogonal(self, 'A')
        # Diagonal scaling Factors constraint to be positive and diagonal.
        self.D = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        P.register_parametrization(self, 'D', PositiveDiagonalParametrization())

    @property
    def weight(self):
        """
        Computes the effective weight matrix as the product of A and D.
        
        Returns:
            torch.Tensor: The effective weight matrix
        """
        # Compute weight dynamically each time it's accessed.
        return self.A @ self.D
    
    def forward(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying the transformation
        """
        return F.linear(x, self.weight)

class PositiveDiagonalParametrization(nn.Module):
    """
    Parametrization that constrains a matrix to be diagonal with positive entries.
    
    Uses the exponential function to ensure positivity of diagonal elements.
    """
    def forward(self, X):
        """
        Transform an unconstrained diagonal matrix to have positive diagonal entries.
        
        Args:
            X (torch.Tensor): Input diagonal matrix
            
        Returns:
            torch.Tensor: Diagonal matrix with positive entries
        """
        # X is an unconstrained diagonal matrix. We assume X is diagonal.
        diag = torch.diag(X)
        positive_diag = torch.exp(diag)
        return torch.diag(positive_diag)


def check_orthogonality(W_mat):
    """
    Check how close a matrix is to being orthogonal.
    
    For an orthogonal matrix W, W^T @ W should be the identity matrix.
    This function measures the deviation from this property.
    
    Args:
        W_mat (torch.Tensor): Matrix to check
        
    Returns:
        None: Prints the orthogonality error
    """
    off_diag = W_mat.t() @ W_mat - torch.diag(torch.diag(W_mat.t() @ W_mat))
    error = torch.norm(off_diag)
    return print(f"Orthogonality error: {error}")

def polar_decomposition(matrix):
    """
    Perform polar decomposition of a matrix.
    
    The polar decomposition of a matrix A is A = U_Polar @ P, where:
    - U_Polar is an orthogonal matrix
    - P is a positive semidefinite matrix
    
    Args:
        matrix (torch.Tensor): Input matrix
        
    Returns:
        tuple: (U_polar, P) - orthogonal and positive semidefinite parts
    """
    # Polar decomposition of a matrix into an orthogonal part and a positive semidefinite part
    # The polar decomposition of a matrix A is A = U_Polar @ P, where U_Polar is an orthogonal matrix and P is a positive semidefinite matrix.
    # In SVD A = U * S * Vh, where U is the orthonormal base and S is the diagonal matrix of the singular values and Vh is the conjugate transpose of V.
    # connection to the polar decomposition: P = Vh^T @ S @ Vh, which is sqrt(matrix^T matrix)
    # and U_polar = U @ Vh 
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False, driver='gesvd')
    U_polar = U @ Vh # Orthogonal (or isometric) part
    # Compute P = Vh^T @ S @ Vh, which is sqrt(matrix^T matrix)
    P = Vh.T @ torch.diag(S) @ Vh # Symmetric positive semi definite part
    return U_polar, P


class DiagonalParametrization(nn.Module):
    """
    Parametrization that constrains a matrix to be diagonal.
    """
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform any matrix to be diagonal by extracting its diagonal elements.
        
        Args:
            X (torch.Tensor): Input matrix
            
        Returns:
            torch.Tensor: Diagonal matrix with the same diagonal as X
        """
        return torch.diag_embed(torch.diagonal(X))

class LowerTriangularParametrization(nn.Module):
    """
    Parametrization that constrains a matrix to be lower triangular.
    """    
    def forward(self, X):
        """
        Transform any matrix to be lower triangular.
        
        Args:
            X (torch.Tensor): Input matrix
            
        Returns:
            torch.Tensor: Lower triangular matrix
        """
        return torch.tril(X)
    
class UpperTriangularParametrization(nn.Module):
    """
    Parametrization that constrains a matrix to be upper triangular.
    """    
    def forward(self, X):
        """
        Transform any matrix to be upper triangular.
        
        Args:
            X (torch.Tensor): Input matrix
            
        Returns:
            torch.Tensor: Upper triangular matrix
        """
        return torch.triu(X)    

class IdentityParametrization(nn.Module):
    """
    Parametrization that forces a matrix to be the identity matrix.
    """
    def forward(self, X):
        """
        Transform any matrix to be the identity matrix.
        
        Args:
            X (torch.Tensor): Input matrix (shape determines size of identity)
            
        Returns:
            torch.Tensor: Identity matrix with the same device and dtype as X
        """
        # We assume X is a square matrix; its shape determines the size.
        n = X.shape[0]
        # Return an identity matrix with the same device and dtype as X.
        return torch.eye(n, device=X.device, dtype=X.dtype)

def constraints_wrapper(constraint_type: str, 
                        layer: nn.Module, 
                        ):
    """
    Apply a specified constraint to a parameter of a neural network layer.

    Parameters:
    - constraint_type (str): The type of constraint to apply ('symmetric', 'orthogonal', 'invertible', 'diagonal').
    - layer (nn.Module): The neural network layer containing the parameter to be constrained.
    - param_name (str): The name of the parameter within the layer to which the constraint will be applied.
    """
    if constraint_type == 'symmetric':
        geotorch.symmetric(layer,'weight')
    elif constraint_type == 'orthonormal':
        geotorch.orthogonal(layer,'weight')
    elif constraint_type == 'diagonal':
        P.register_parametrization(layer, 'weight', DiagonalParametrization())
    elif constraint_type == 'invertible':
        geotorch.invertible(layer,'weight')
    elif constraint_type == 'lower_triangular':
        P.register_parametrization(layer, 'weight', LowerTriangularParametrization())
    elif constraint_type == 'upper_triangular':
        P.register_parametrization(layer, 'weight', UpperTriangularParametrization())
    elif constraint_type == 'zero_shot':
        P.register_parametrization(layer, 'weight', IdentityParametrization())
    elif constraint_type == 'full_W':
        return
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")


# if __name__ == "__main__":
#     print("Testing constraints wrapper...")
    
#     # Test symmetric constraint
#     print("\nTesting symmetric constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     param_name = 'weight'
#     constraints_wrapper('symmetric', test_layer, param_name)
#     assert torch.allclose(test_layer.weight, test_layer.weight.T), "Symmetric constraint failed"
#     print("Symmetric constraint test passed")
    
#     # Test orthogonal constraint 
#     print("\nTesting orthogonal constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     constraints_wrapper('orthogonal', test_layer, param_name)
#     product = test_layer.weight @ test_layer.weight.T
#     assert torch.allclose(product, torch.eye(10), atol=1e-6), "Orthogonal constraint failed"
#     print("Orthogonal constraint test passed")
    
#     # Test invertible constraint
#     print("\nTesting invertible constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     constraints_wrapper('invertible', test_layer, param_name)
#     # Check if determinant is non-zero (matrix is invertible)
#     det = torch.det(test_layer.weight)
#     assert det != 0, "Invertible constraint failed: determinant is zero"
#     print("Invertible constraint test passed")
    
#     # Test diagonal constraint
#     print("\nTesting diagonal constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     constraints_wrapper('diagonal', test_layer, param_name)
#     # Check if off-diagonal elements are zero
#     mask = ~torch.eye(10, dtype=torch.bool)
#     assert torch.allclose(test_layer.weight[mask], torch.zeros_like(test_layer.weight[mask])), "Diagonal constraint failed"
#     print("Diagonal constraint test passed")

#     # Test lower triangular constraint
#     print("\nTesting lower triangular constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     constraints_wrapper('lower_triangular', test_layer, param_name)
#     # Check if upper triangular part (excluding diagonal) is zero
#     upper_mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()
#     assert torch.allclose(test_layer.weight[upper_mask], torch.zeros_like(test_layer.weight[upper_mask])), "Lower triangular constraint failed"
#     print("Lower triangular constraint test passed")

#     # Test upper triangular constraint
#     print("\nTesting upper triangular constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     constraints_wrapper('upper_triangular', test_layer, param_name)
#     # Check if lower triangular part (excluding diagonal) is zero
#     lower_mask = torch.tril(torch.ones(10, 10), diagonal=-1).bool()
#     assert torch.allclose(test_layer.weight[lower_mask], torch.zeros_like(test_layer.weight[lower_mask])), "Upper triangular constraint failed"
#     print("Upper triangular constraint test passed")
    
#     # Test invalid constraint
#     print("\nTesting invalid constraint...")
#     test_layer = nn.Linear(10, 10)  # Fresh layer
#     try:
#         constraints_wrapper('invalid', test_layer, param_name)
#         raise AssertionError("Invalid constraint type should raise ValueError")
#     except ValueError:
#         print("Invalid constraint test passed")
    
#     print("\nAll tests passed!")