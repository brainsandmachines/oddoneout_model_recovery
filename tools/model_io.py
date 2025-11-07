"""
Model Input/Output Module

This module provides utility functions for saving and loading CogModel instances to/from disk.
It handles serialization of model parameters, configurations, and state in a device-agnostic manner.

Key functions:
- save_cog_model: Saves a CogModel instance to a .pt file
- load_cog_model: Loads a CogModel instance from a saved .pt file

Part of the model recovery pipeline - used by:
- create_data_generating_models.py (saves fitted models)
- run_simulations.py (loads data-generating models)
"""

import torch
import os
import sys
from typing import Optional

#add parent directory to path and all subdirectories possible
for dir in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), dir))
from tools.Cog_model import CogModel

def save_cog_model(model: CogModel, save_folder_path: str, file_name: Optional[str] = None):
    """
    Save a CogModel to disk, handling device and dtype appropriately.
    
    This function serializes a CogModel instance to a file, preserving its parameters and configuration.
    The model state is moved to CPU before saving to ensure compatibility when loading on different devices.
    The function creates the target directory if it doesn't already exist.
    
    Parameters:
        model (CogModel): The cognitive model instance to save
        save_folder_path (str): Directory path where to save the model
        file_name (Optional[str]): Optional name for the saved file. If None, the model's name 
                                   with .pt extension is used
    
    Returns:
        str: The full path to the saved model file
        
    Notes:
        - The model's weights are moved to CPU before saving for better compatibility
        - Configuration parameters are stored in a separate dictionary for easier loading
        - Original device information is preserved for potential restoration
    """
    # Create folder if it doesn't exist
    os.makedirs(save_folder_path, exist_ok=True)
    
    # Use model name as filename if none provided
    if file_name is None:
        file_name = f"{model.model_name}.pt"
    
    # Ensure filename has .pt extension
    if not file_name.endswith('.pt'):
        file_name += '.pt'
    
    # Combine folder path and filename
    save_path = os.path.join(save_folder_path, file_name)
    
    # Store original device for metadata
    original_device = str(model.device)
    
    config = {
        'model_name': model.model_name,
        'images_features_dim': model.images_features_dim,
        'work_with_S_matrix': model.work_with_S_matrix,
        'similarity_function': model.similarity_function,
        'max_memory_gb': model.max_memory_gb,
        'dtype': str(model.dtype),
        'original_device': original_device,
        'constraints': model.constraint
    }
    
    # Move model state to CPU before saving
    model_state = {
        key: value.cpu() for key, value in model.state_dict().items()
    }
    
    torch.save({
        'model_state_dict': model_state,
        'model_config': config
    }, save_path)
    
    return save_path  # Return the full save path for reference

def is_diagonal(matrix, tol=1e-8):
    """
    Check if a square matrix is diagonal within a given tolerance.
    
    A diagonal matrix has non-zero entries only on the main diagonal. This function
    checks if all off-diagonal elements are close to zero within the specified tolerance.
    
    Parameters:
        matrix (torch.Tensor): The matrix to check, must be 2D and square
        tol (float): The tolerance threshold for considering an element to be zero
    
    Returns:
        bool: True if the matrix is diagonal within the given tolerance, False otherwise
        
    Raises:
        ValueError: If the input is not a square matrix
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    off_diagonal = matrix - torch.diag(torch.diagonal(matrix))
    return torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=tol)

def load_cog_model(save_folder_path: str, file_name: Optional[str] = None, device=None, dtype=None) -> CogModel:
    """
    Load a CogModel from disk with specified device and dtype.
    
    This function deserializes a previously saved CogModel instance from a file. It provides
    flexibility to load the model to a different device or with a different data type than
    the original. The function automatically detects model constraints based on the weight matrix structure.
    
    Parameters:
        save_folder_path (str): Directory path where the model is saved
        file_name (Optional[str]): Name of the saved file. If None, the function looks for a single .pt file
                                  in the directory
        device (str, optional): Target device to load model to (default: CUDA if available, else CPU)
        dtype (torch.dtype, optional): Target dtype to load model with (default: saved dtype)
        
    Returns:
        CogModel: The loaded model instance, moved to the specified device
        
    Raises:
        FileNotFoundError: If no .pt files are found in the directory
        ValueError: If multiple .pt files are found and no specific file_name is provided
        
    Notes:
        - The function automatically detects constraint types (rectangular, diagonal) based on W matrix shape
        - The model is instantiated with its original configuration but can be placed on a different device
        - All tensors are converted to the specified device and dtype during loading
    """
    # If filename not provided, look for .pt file in directory
    if file_name is None:
        pt_files = [f for f in os.listdir(save_folder_path) if f.endswith('.pt')]
        if len(pt_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {save_folder_path}")
        if len(pt_files) > 1:
            raise ValueError(f"Multiple .pt files found in {save_folder_path}. Please specify file_name.")
        file_name = pt_files[0]
    
    # Ensure filename has .pt extension
    if not file_name.endswith('.pt'):
        file_name += '.pt'
    
    # Combine folder path and filename
    load_path = os.path.join(save_folder_path, file_name)
    
    # Rest of the loading logic remains the same
    checkpoint = torch.load(load_path)
    config = checkpoint['model_config']
    try:
        if checkpoint["model_state_dict"]["W_mat.weight"].shape[0] != config["images_features_dim"]:
            config["constraints"] = ("rectangular",checkpoint["model_state_dict"]["W_mat.weight"].shape[0])
    except:    
        if is_diagonal(checkpoint["model_state_dict"]["W_mat.parametrizations.weight.original"]):
            config["constraints"] = "diagonal"
        else:
            config["constraints"] = None
            
    if device is None:
        original_device = config.get('original_device', 'cuda')
        device = original_device if torch.cuda.is_available() else 'cpu'
    
    if dtype is None:
        dtype = getattr(torch, config['dtype'].split('.')[-1])
    
    config['device'] = device
    config['dtype'] = dtype
    config.pop('original_device', None)
    
    model = CogModel(**config)
    
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict({
        key: value.to(device=device, dtype=dtype) 
        for key, value in state_dict.items()
    })
    
    model.to(device)
    
    return model

# if __name__ == "__main__":
#     print("Testing model IO functionality...")
#     
#     # Create a dummy model
#     model = CogModel(
#         model_name="test_model",
#         images_features_dim=10,
#         work_with_S_matrix=True,
#         temp=2.0,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )
#     
#     # Print initial state
#     print(f"\nInitial model state:")
#     print(f"Device: {model.device}")
#     print(f"W_mat device: {model.W_mat.device}")
#     print(f"Temperature: {model.temp.item()}")
#     
#     # Create models directory in work directory
#     models_dir = os.path.join(os.getcwd(), "models")
#     
#     # Save the model with default name (test_model.pt)
#     print(f"\nSaving model to {models_dir}")
#     save_path = save_cog_model(model, models_dir)
#     print(f"Model saved to: {save_path}")
#     
#     # Load the model
#     print("\nLoading model...")
#     loaded_model = load_cog_model(models_dir)
#     
#     # Print loaded state
#     print(f"\nLoaded model state:")
#     print(f"Device: {loaded_model.device}")
#     print(f"W_mat device: {loaded_model.W_mat.device}")
#     print(f"Temperature: {loaded_model.temp.item()}")
#     
#     # Test if states match
#     print("\nChecking if states match:")
#     w_mat_match = torch.allclose(model.W_mat, loaded_model.W_mat)
#     temp_match = torch.allclose(model.temp, loaded_model.temp)
#     print(f"W_mat match: {w_mat_match}")
#     print(f"Temperature match: {temp_match}")
#     
#     # Clean up (optional - comment out if you want to keep the saved model)
#     if os.path.exists(save_path):
#         os.remove(save_path)
#         print(f"\nCleaned up {save_path}")
#         # Remove models directory if empty
#         if not os.listdir(models_dir):
#             os.rmdir(models_dir)
#             print(f"Removed empty directory: {models_dir}")