"""
Regularization Functions Module

This module provides various regularization functions for neural network optimization.
These functions apply penalties to model parameters to prevent overfitting and enforce
specific structural properties.

Currently supported regularization methods:
- L2: Standard L2 regularization (weight decay)
- eye_distance: Penalizes deviation from scaled identity matrix
"""

import torch
import torch.nn as nn
from typing import Optional

class RegularizationFunction:
    """
    Class implementing different regularization functions for neural network optimization.
    
    This class supports different regularization techniques that can be applied during 
    model training to penalize certain parameter configurations.
    
    Attributes:
        reg_func_name (str): Name of the regularization function to use
        lambda_reg (float): Strength of the regularization
        device (torch.device): Device on which to perform calculations
    """
    def __init__(self,
            reg_func_name: str,
            lambda_reg: float,
            device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
            ):
        """
        Initialize the regularization function.
        
        Args:
            reg_func_name (str): Name of the regularization function ('L2', 'eye_distance')
            lambda_reg (float): Strength of the regularization penalty
            device (torch.device, optional): Device to use for computation.
                Defaults to CUDA if available, otherwise CPU.
        """
        super(RegularizationFunction, self).__init__()
        self.device = device
        self.reg_func_name = reg_func_name
        self.lambda_reg = torch.tensor(lambda_reg).to(self.device)
        assert self.reg_func_name in ["L1","L2","eye_distance"], f"Regularization function {self.reg_func_name} not recognized!"
        
    def __call__(self, model_parameters):
        """
        Apply the regularization function to model parameters.
        
        Args:
            model_parameters (torch.Tensor): Parameters to regularize
            
        Returns:
            torch.Tensor: Regularization penalty value
            
        Raises:
            ValueError: If the regularization function name is not recognized
        """
        if self.reg_func_name == "L1":
            penalty = self.lambda_reg * torch.sum(torch.abs(model_parameters))
            return penalty
        
        if self.reg_func_name == "L2":
            penalty = self.lambda_reg * torch.sum(torch.square(model_parameters))
            return penalty
        
        elif self.reg_func_name == "eye_distance":
            # Use the optimal solution to minimize ||W - cI||^2
            # c* = Trace(W) / N assuming both W and I are NxN matrices
            if model_parameters.shape[0] == model_parameters.shape[1]:
                constant = torch.trace(model_parameters) / model_parameters.shape[0]
                penalty = self.lambda_reg * torch.sum(torch.square(model_parameters - (constant * torch.eye(model_parameters.shape[0]).to(self.device))))
            else:  # Handle the rectangular matrix case
                W = model_parameters
                nin = W.shape[1]  # Number of input features 
                # Ensure the transpose is contiguous.
                W_t = W.T
                # Compute W^T W and the constant
                reg_matrix = torch.matmul(W_t, W)  # Shape: (input_dim, input_dim)
                constant = torch.trace(reg_matrix) / nin
                # Create an identity matrix of the appropriate size
                I = torch.eye(nin, device=self.device)  # Identity matrix with matching dimensions

                # Compute the regularization penalty
                penalty = self.lambda_reg * torch.sum(torch.square(reg_matrix - (constant * I)))

            return penalty
        
        else:
            raise ValueError(f"Regularization function {self.reg_func_name} not recognized!")
        
        
