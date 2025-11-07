"""
Save Transformed Features Script
==================================

This script applies learned W transformation matrices from fitted CogModels to raw 
neural network features, producing transformed feature representations.

Purpose:
--------
After fitting CogModels to behavioral data (via create_data_generating_models.py),
this script extracts and saves the transformed features that result from applying
the learned W matrix to the original model features.

Workflow:
---------
1. Load raw neural network features for each model (e.g., ResNet50.pt)
2. Load corresponding fitted CogModel with learned W matrix
3. Apply transformation: features_transformed = W_mat(features)
4. Save transformed features for downstream analysis (e.g., RSA, visualization)

Use Cases:
----------
- Representational Similarity Analysis (RSA) on transformed spaces
- Visualization of learned transformations (e.g., MDS, t-SNE)
- Comparing raw vs. transformed feature geometries
- Feature extraction for other downstream tasks

Configuration:
--------------
Uses Hydra config from scripts_configurations/save_transformed_features.yaml

Required config keys:
    models: List of model names to process
    device: "cuda" or "cpu"
    paths:
        raw_features_dir: Directory containing raw features (.pt files)
        models_dir: Directory containing fitted CogModels (.pt files)
        transformed_out: Output directory for transformed features

Example Usage:
--------------
    # Default configuration
    python analysis_scripts/save_transformed_features.py
    
    # Override device
    python analysis_scripts/save_transformed_features.py device=cpu
    
    # Override output directory
    python analysis_scripts/save_transformed_features.py \
        paths.transformed_out=Data/transformed_features_custom/

Output:
-------
Saves transformed features to paths.transformed_out/{model_name}.pt
Each file contains a [N_concepts, feature_dim] tensor (same shape as input features)

Author: Itamar Avitan & Tal Golan
Institution: Ben-Gurion University of the Negev
Paper: NeurIPS 2025 - Model-Behavior Alignment under Flexible Evaluation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from tools.Cog_model import CogModel
from tools.model_io import load_cog_model


def _ensure_dir(path: str):
    """
    Create directory and all parent directories if they don't exist.
    
    Args:
        path: Directory path to create
        
    Note:
        Uses exist_ok=True to avoid errors if directory already exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_tensor(path: str) -> torch.Tensor:
    """
    Load a PyTorch tensor from disk.
    
    Args:
        path: Path to .pt file containing tensor
        
    Returns:
        Loaded tensor
        
    Note:
        Assumes file was saved with torch.save()
    """
    return torch.load(path)


def _save_tensor(tensor: torch.Tensor, path: str):
    """
    Save a PyTorch tensor to disk.
    
    Args:
        tensor: Tensor to save
        path: Destination path for .pt file
        
    Note:
        Automatically creates parent directories if they don't exist
    """
    _ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)


def _device() -> str:
    """
    Determine available device (CUDA GPU or CPU).
    
    Returns:
        "cuda" if GPU available, otherwise "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(models_dir: str, model_name: str, device: Optional[str] = None) -> CogModel:
    """
    Load a fitted CogModel from disk.
    
    Args:
        models_dir: Directory containing saved CogModel .pt files
        model_name: Name of model (without .pt extension)
        device: Device to load model to ("cuda" or "cpu"). If None, auto-detects.
        
    Returns:
        Loaded CogModel instance with learned W matrix
        
    Note:
        Uses model_io.load_cog_model() for robust loading with proper dtype/device handling
    """
    device = device or _device()
    # Prefer robust loader via model_io to preserve dtype/device handling
    return load_cog_model(models_dir, f"{model_name}.pt", device=device)


def transform_and_save(model_name: str, cfg_paths: Dict[str, str], device: str):
    """
    Apply learned W transformation to model features and save result.
    
    This function:
    1. Loads raw neural network features for a specific model
    2. Loads the corresponding fitted CogModel with learned W matrix
    3. Applies transformation: features_transformed = W_mat(features)
    4. Saves transformed features to disk
    
    Args:
        model_name: Name of the model (e.g., "ResNet50", "ViT_Large")
        cfg_paths: Dictionary with paths:
            - raw_features_dir: Directory containing raw features (.pt files)
            - models_dir: Directory containing fitted CogModels (.pt files)
            - transformed_out: Output directory for transformed features
        device: Device for computation ("cuda" or "cpu")
        
    Skips:
        - Models without raw features file
        - Models without fitted CogModel file
        
    Output:
        Saves transformed features to: {transformed_out}/{model_name}.pt
        
    Example:
        >>> paths = {
        ...     "raw_features_dir": "Data/models_features/full/",
        ...     "models_dir": "Data/models_data_generating/full_W/",
        ...     "transformed_out": "Data/transformed_features/"
        ... }
        >>> transform_and_save("ResNet50", paths, "cuda")
        [ok] Saved transformed features for ResNet50 -> Data/transformed_features/ResNet50.pt
    """
    raw_features_dir = Path(cfg_paths["raw_features_dir"])  # folder of raw features .pt (one per model)
    models_dir = Path(cfg_paths["models_dir"])              # folder with saved CogModel .pt (one per model)
    out_dir = Path(cfg_paths["transformed_out"])           # destination for transformed features

    _ensure_dir(str(out_dir))

    # Construct file paths
    features_path = raw_features_dir / f"{model_name}.pt"
    model_path = models_dir / f"{model_name}.pt"
    out_path = out_dir / f"{model_name}.pt"

    # Check if required files exist
    if not features_path.exists():
        print(f"[skip] Missing features for {model_name}: {features_path}")
        return
    if not model_path.exists():
        print(f"[skip] Missing model for {model_name}: {model_path}")
        return

    # Load raw features
    features: torch.Tensor = _load_tensor(str(features_path))
    # Ensure 2D tensor (N_concepts × feature_dim)
    assert features.ndim == 2, f"Features must be 2D (N x D). Got shape {tuple(features.shape)} for {model_name}"

    # Load fitted CogModel
    cog: CogModel = _load_model(str(models_dir), model_name, device=device)
    cog.eval()  # Set to evaluation mode (disables dropout, etc.)

    # Align dtype and device between features and model
    if features.device != cog.device:
        features = features.to(cog.device)
    if hasattr(cog, 'dtype') and features.dtype != cog.dtype:
        features = features.to(cog.dtype)

    # Apply W transformation without gradient computation
    with torch.no_grad():
        transformed = cog.apply_W_matrix(features)

    # Save transformed features
    _save_tensor(transformed, str(out_path))
    print(f"[ok] Saved transformed features for {model_name} -> {out_path}")


@hydra.main(version_base=None, config_path="scripts_configurations", config_name="save_transformed_features.yaml")
def main(cfg: DictConfig):
    """
    Main function to transform and save features for multiple models.
    
    Loads configuration via Hydra and processes each model in the list:
    1. Applies learned W transformation from fitted CogModel
    2. Saves transformed features for downstream analysis
    
    Args:
        cfg: Hydra configuration object with keys:
            - models: List of model names to process
            - device: "cuda" or "cpu" (auto-detects if invalid)
            - paths: Dictionary with raw_features_dir, models_dir, transformed_out
            
    Configuration:
        Loaded from scripts_configurations/save_transformed_features.yaml
        
    Example config:
        models: [ResNet50, ViT_Large, CLIP_RN50, ...]
        device: cuda
        paths:
          raw_features_dir: Data/models_features/full/
          models_dir: Data/models_data_generating/full_W/
          transformed_out: Data/transformed_features/
          
    Usage:
        python analysis_scripts/save_transformed_features.py
        python analysis_scripts/save_transformed_features.py device=cpu
        python analysis_scripts/save_transformed_features.py \
            paths.transformed_out=Data/custom_transformed/
    """
    # Determine device (validate cfg.device or auto-detect)
    device = cfg.device if cfg.device in ("cuda", "cpu") else _device()
    models: List[str] = list(cfg.models)

    # Ensure output directory exists
    _ensure_dir(cfg.paths["transformed_out"])

    # Process each model with progress bar
    for model_name in tqdm(models, desc="Transform", position=0):
        transform_and_save(model_name, cfg.paths, device)

    print("All transformed features saved.")


if __name__ == "__main__":
    main()


