"""
================================================================================
RSA Similarity Matrix Builder
================================================================================

This script computes Representational Similarity Analysis (RSA) metrics for
comparing model representations before and after linear transformations.

Purpose:
--------
Generates pairwise similarity matrices between model representations using RSA
to understand how different vision models organize their representational spaces.
This analysis reveals which models have similar internal representations and how
linear transformations affect representational geometry.

Key Concepts:
-------------
1. **RDM (Representational Dissimilarity Matrix)**:
   - For each model, computes pairwise distances between all stimulus representations
   - Shows how the model organizes the stimulus space
   - Calculated using Euclidean distance between feature vectors

2. **Similarity Matrix**:
   - Compares RDMs between different models using correlation
   - High correlation = similar representational geometry
   - Used to cluster models with similar representations

3. **Raw vs. Transformed Features**:
   - Raw: Original model activations (before alignment)
   - Transformed: After applying learned linear transformations (W)
   - Comparison reveals how alignment changes representational structure

Usage:
------
    # Default configuration (full features, full_W transformation)
    python analysis_scripts/build_rsa_sim_matrix.py
    
    # Use PCA-reduced features
    python analysis_scripts/build_rsa_sim_matrix.py features_dim="PCA_500"
    
    # Different transformation constraint
    python analysis_scripts/build_rsa_sim_matrix.py transformed_features_constraints="diagonal"
    
    # Include only base 20 models
    python analysis_scripts/build_rsa_sim_matrix.py include_new_models=False

Configuration:
--------------
All parameters are specified in: scripts_configurations/build_rsa_sim_matrix.yaml

Inputs:
-------
- Model features (raw): Data/models_features/models_original_features/{features_dim}/
- Model features (transformed): Data/models_features/models_transformed_features/{features_dim}/{constraints}/
- VICE features: Data/models_features/VICE_features/VICE.pt

Outputs:
--------
- RDMs: Results/rsa/rdms_{features_dim}_dim_{constraints}.hdf5
- Similarity matrix: Results/rsa/sim_{features_dim}_dim_{constraints}.nc

Dependencies:
-------------
- rsatoolbox: For RSA computations
- xarray: For saving/loading similarity matrices
- torch: For loading feature tensors
- hydra: For configuration management

Paper Context:
--------------
Used in Figure S4 to show how model representations cluster based on
training objective and architecture, and how linear transformations
affect these representational relationships.

Author: Itamar Avitan and Tal Golan
Institution: Department of Cognitive and Brain Sciences, Ben-Gurion University
Paper: NeurIPS 2025

================================================================================
"""

import os
import sys
from pathlib import Path

# Set up paths to allow relative imports
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import rsatoolbox as rsa
import hydra
from omegaconf import DictConfig
import xarray as xr


# =============================================================================
# Feature Loading and Validation
# =============================================================================

# =============================================================================
# Feature Loading and Validation
# =============================================================================

def build_validated_models_dict(
    features_paths_dict: Dict[str, str], 
    models_list: List[str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Validate feature file paths and build dictionaries mapping models to their features.
    
    This function ensures data integrity by:
    1. Verifying all required directories exist
    2. Checking that each model has features in BOTH raw and transformed folders
    3. Filtering out models with missing data
    4. Creating properly formatted path dictionaries for downstream analysis
    
    The function is defensive - it validates all paths upfront to prevent failures
    during expensive RSA computations. VICE (human behavioral features) is always
    included as a reference point for model-human alignment.
    
    Args:
        features_paths_dict: Dictionary containing three required keys:
            - 'raw_features_path': Directory with original model activations
            - 'post_transformation_features_path': Directory with transformed features
            - 'vice_features_path': Directory containing VICE.pt (human features)
        models_list: List of model names (without .pt extension) to include in analysis
        
    Returns:
        Tuple of two dictionaries:
        - raw_models_dict: Maps "{model_name}_raw" -> absolute path to .pt file
        - post_transformation_models_dict: Maps "{model_name}_post" -> absolute path
        Both dictionaries include VICE with "_raw" and "_post" suffixes.
        
    Raises:
        FileNotFoundError: If any directory doesn't exist or VICE.pt is missing
        
    Example:
        >>> features_paths = {
        ...     "raw_features_path": "Data/models_features/models_original_features/full",
        ...     "post_transformation_features_path": "Data/models_features/models_transformed_features/full/full_W",
        ...     "vice_features_path": "Data/models_features/VICE_features"
        ... }
        >>> models = ["ResNet50", "VGG19", "DINOv2_ViT_Large"]
        >>> raw_dict, post_dict = build_validated_models_dict(features_paths, models)
        >>> print(raw_dict.keys())
        dict_keys(['DINOv2_ViT_Large_raw', 'ResNet50_raw', 'VICE_raw', 'VGG19_raw'])
        
    Notes:
        - Models are sorted alphabetically for consistent ordering
        - Missing models trigger warnings but don't stop execution
        - VICE is treated specially and always included if file exists
        - The "_raw" and "_post" suffixes distinguish feature types in analysis
    """
    # Extract and validate paths
    raw_path = Path(features_paths_dict["raw_features_path"])
    post_path = Path(features_paths_dict["post_transformation_features_path"])
    vice_path = Path(features_paths_dict["vice_features_path"])
    
    # 1. Check that all directories exist
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw features path not found: {raw_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"Post-transformation features path not found: {post_path}")
    if not vice_path.exists():
        raise FileNotFoundError(f"VICE features path not found: {vice_path}")
    
    # Check VICE.pt exists
    vice_file = vice_path / "VICE.pt"
    if not vice_file.exists():
        raise FileNotFoundError(f"VICE.pt not found at: {vice_file}")
    
    # 2. Get available models in each folder
    raw_files = {f.stem for f in raw_path.glob("*.pt")}
    post_files = {f.stem for f in post_path.glob("*.pt")}
    
    # 3. Filter models_list to only include models present in BOTH folders
    valid_models = []
    missing_models = []
    
    for model_name in models_list:
        if model_name in raw_files and model_name in post_files:
            valid_models.append(model_name)
        else:
            missing_info = []
            if model_name not in raw_files:
                missing_info.append(f"raw ({raw_path / f'{model_name}.pt'})")
            if model_name not in post_files:
                missing_info.append(f"post ({post_path / f'{model_name}.pt'})")
            missing_models.append(f"{model_name}: missing in {', '.join(missing_info)}")
    
    # Print validation summary
    print(f"Found {len(valid_models)}/{len(models_list)} models in both raw and post-transformation folders")
    if missing_models:
        print(f"Warning: {len(missing_models)} models not found in all locations:")
        for missing in missing_models:
            print(f"  - {missing}")
    print(f"VICE.pt found at: {vice_file}")
    
    # 4. Build the models dictionaries
    raw_models_dict = {}
    post_transformation_models_dict = {}
    
    # Add validated models
    for model_name in valid_models:
        model_file = f"{model_name}.pt"
        raw_models_dict[f"{model_name}_raw"] = os.path.join(raw_path, model_file)
        post_transformation_models_dict[f"{model_name}_post"] = os.path.join(post_path, model_file)
    
    # Add VICE (always included)
    raw_models_dict["VICE_raw"] = str(vice_file)
    post_transformation_models_dict["VICE_post"] = str(vice_file)
    
    # Sort by model name for consistent ordering
    raw_models_dict = dict(sorted(raw_models_dict.items(), key=lambda item: item[0]))
    post_transformation_models_dict = dict(sorted(post_transformation_models_dict.items(), key=lambda item: item[0]))
    
    print(f"Built models dict with {len(valid_models)} models + VICE = {len(raw_models_dict)} total entries")
    
    return raw_models_dict, post_transformation_models_dict


def load_datasets(
    features_paths_dict: Dict[str, str], 
    models_list: List[str]
) -> List[rsa.data.Dataset]:
    """
    Load model features and convert to rsatoolbox Dataset format for RSA analysis.
    
    This function orchestrates the loading process:
    1. Validates and builds model path dictionaries
    2. Loads feature tensors from .pt files
    3. Wraps features in rsatoolbox Dataset objects with proper metadata
    
    The rsatoolbox Dataset format is required for RSA computations and includes:
    - measurements: The actual feature vectors (n_stimuli × n_features)
    - descriptors: Model-level metadata (model name)
    - obs_descriptors: Stimulus-level metadata (image indices)
    
    Args:
        features_paths_dict: Dictionary with paths to raw, transformed, and VICE features
        models_list: List of model names to load (will be validated internally)
        
    Returns:
        List of rsatoolbox.data.Dataset objects, one per model configuration:
        - Each model appears twice: once for raw features, once for transformed
        - VICE appears twice as well (same features, different labels)
        - Total datasets = 2 × (number of valid models + 1 for VICE)
        
    Example:
        >>> datasets = load_datasets(features_paths, ["ResNet50", "VGG19"])
        >>> print(len(datasets))  # 2 models + VICE, each with raw and post
        6
        >>> print(datasets[0].descriptors['model'])
        'ResNet50_raw'
        >>> print(datasets[0].measurements.shape)
        (1854, 2048)  # n_images × n_features
        
    Notes:
        - Features are converted to float32 for memory efficiency
        - Features are moved to CPU to avoid GPU memory issues
        - Image indices are used as observation descriptors for tracking stimuli
        - Model names include "_raw" or "_post" suffix to distinguish feature types
    """
    raw_models_dict, post_transformation_models_dict = build_validated_models_dict(features_paths_dict, models_list)
    
    # Combine raw and post-transformation dictionaries
    models_dict = {**raw_models_dict, **post_transformation_models_dict}
    
    # Create rsatoolbox Dataset objects for each model
    datasets = []
    for model_name, model_path in models_dict.items():
        # Load feature tensor and convert to numpy array
        arr = torch.load(model_path).detach().cpu().numpy().astype(np.float32)
        
        # Wrap in rsatoolbox Dataset with metadata
        datasets.append(
            rsa.data.Dataset(   
                measurements=arr,  # Shape: (n_stimuli, n_features)
                descriptors={'model': model_name},  # Model identifier
                obs_descriptors={'image': np.arange(arr.shape[0])},  # Image indices
            )
        )
    return datasets


# =============================================================================
# RSA Computation
# =============================================================================

def create_models_RDMs_and_sim_matrix(
    features_paths_dict: Dict[str, Any], 
    models_list: List[str]
) -> Tuple[np.ndarray, rsa.rdm.RDMs]:
    """
    Compute RDMs and pairwise similarity matrix for all models.
    
    This is the core RSA computation function that:
    1. Loads all model features as rsatoolbox Datasets
    2. Computes RDMs (Representational Dissimilarity Matrices) for each model
       - Uses Euclidean distance between feature vectors
       - Results in one RDM per model (n_stimuli × n_stimuli matrix)
    3. Compares RDMs pairwise to create similarity matrix
       - Uses correlation of covariances (corr_cov)
       - Results in (n_models × n_models) similarity matrix
    
    The similarity matrix reveals which models have similar representational
    geometries. High values indicate models organize stimuli similarly, even
    if their raw feature values differ.
    
    Args:
        features_paths_dict: Dictionary with paths to all feature directories
        models_list: List of model names to include in analysis
        
    Returns:
        Tuple of (similarity_matrix, rdms):
        - similarity_matrix: (n_models × n_models) array of RDM correlations
          Range: [-1, 1] where 1 = identical geometry, 0 = uncorrelated, -1 = opposite
        - rdms: rsatoolbox.rdm.RDMs object containing all individual model RDMs
          Each RDM is (n_stimuli × n_stimuli) dissimilarity matrix
          
    Example:
        >>> sim, rdms = create_models_RDMs_and_sim_matrix(features_paths, models)
        >>> print(sim.shape)
        (44, 44)  # 22 models × 2 (raw + post)
        >>> print(rdms.n_rdm)
        44
        >>> print(rdms.rdm_descriptors['model'][0])
        'AlexNet_raw'
        
    Notes:
        - Euclidean distance is standard for RSA with neural network features
        - corr_cov (correlation of covariances) is more robust than simple correlation
        - All computations are done in float32 for memory efficiency
        - Diagonal of similarity matrix should be 1 (model compared to itself)
    """
    # Load all model features as rsatoolbox Datasets
    models_datasets = load_datasets(features_paths_dict, models_list)
    
    # Compute RDM for each model using Euclidean distance
    rdms = rsa.rdm.calc_rdm(models_datasets, method='euclidean')
    
    # Compare RDMs pairwise using correlation of covariances
    sim = rsa.rdm.compare(rdms, rdms, method='corr_cov')
    
    return sim, rdms


# =============================================================================
# Output Saving
# =============================================================================

def save_RDMs_and_sim_matrix(
    rdms: rsa.rdm.RDMs, 
    sim: np.ndarray, 
    save_path: str, 
    features_name: str
) -> None:
    """
    Save RSA results to disk in standard formats for later analysis.
    
    Saves two files:
    1. Similarity matrix as NetCDF (xarray format)
       - Includes model names as coordinates for easy indexing
       - Can be loaded with xarray.open_dataarray()
       - Format: sim_{features_name}.nc
       
    2. RDMs as HDF5 (rsatoolbox format)
       - Preserves all metadata and descriptors
       - Can be loaded with rsatoolbox.rdm.load_rdm()
       - Format: rdms_{features_name}.hdf5
    
    Args:
        rdms: rsatoolbox RDMs object containing all model RDMs
        sim: Similarity matrix (n_models × n_models numpy array)
        save_path: Directory where files will be saved
        features_name: Descriptive name for output files (e.g., "full_dim_full_W")
        
    Output Files:
        - {save_path}/sim_{features_name}.nc: xarray NetCDF similarity matrix
        - {save_path}/rdms_{features_name}.hdf5: rsatoolbox HDF5 RDMs
        
    Example:
        >>> save_RDMs_and_sim_matrix(rdms, sim, "Results/rsa", "full_dim_full_W")
        # Creates:
        # - Results/rsa/sim_full_dim_full_W.nc
        # - Results/rsa/rdms_full_dim_full_W.hdf5
        
    Notes:
        - NetCDF format preserves labeled dimensions for easy plotting
        - HDF5 format is efficient for large RDM collections
        - Both formats are standard in computational neuroscience
        - Files can be loaded for visualization in Figure S4
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Extract model labels from RDMs for xarray coordinates
    labels = rdms.rdm_descriptors['model']
    
    # Convert similarity matrix to xarray with labeled dimensions
    sim_xr = xr.DataArray(
        sim, 
        dims=["model1", "model2"], 
        coords={"model1": labels, "model2": labels}
    )
    
    # Save similarity matrix as NetCDF
    sim_xr.to_netcdf(os.path.join(save_path, f"sim_{features_name}.nc"))
    
    # Save RDMs as HDF5 (rsatoolbox native format)
    rdms.save(os.path.join(save_path, f"rdms_{features_name}"), file_type="hdf5")


# =============================================================================
# Main Execution
# =============================================================================

@hydra.main(config_path="../scripts_configurations", config_name="build_rsa_sim_matrix.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function orchestrating the RSA analysis pipeline.
    
    Workflow:
    ---------
    1. Extract configuration parameters from Hydra config
    2. Construct paths to raw and transformed feature directories
    3. Load and validate all model features
    4. Compute RDMs and similarity matrix
    5. Save results to disk
    6. Print summary of saved files
    
    Configuration Parameters (from build_rsa_sim_matrix.yaml):
    ----------------------------------------------------------
    - features_dim: Dimensionality type ("full" or "PCA_500")
    - transformed_features_constraints: Transformation type ("zero_shot", "diagonal", 
      "Rectangular_N", or "full_W")
    - save_path: Output directory for results
    - models_features_path: Base directory for all model features
    - vice_features_path: Directory containing VICE.pt
    - include_new_models: Whether to include additional 10 models beyond base 20
    
    Args:
        cfg: Hydra configuration object loaded from build_rsa_sim_matrix.yaml
        
    Output Files:
    ------------
    - {save_path}/rdms_{features_dim}_dim_{constraints}.hdf5
    - {save_path}/sim_{features_dim}_dim_{constraints}.nc
    
    Example Run:
    -----------
    >>> python analysis_scripts/build_rsa_sim_matrix.py
    features_dim: full
    transformed_features_constraints: full_W
    save_path: Results/rsa
    Found 32/32 models in both raw and post-transformation folders
    VICE.pt found at: Data/models_features/VICE_features/VICE.pt
    Built models dict with 32 models + VICE = 33 total entries
    Saved RDMs as: Results/rsa/rdms_full_dim_full_W.hdf5
    Saved similarity matrix as: Results/rsa/sim_full_dim_full_W.nc
    
    Notes:
    ------
    - Script automatically creates save_path directory if it doesn't exist
    - Missing models are reported but don't stop execution
    - VICE is always included if the file exists
    - File paths are constructed to match the standard project structure
    """
    # Extract configuration parameters
    features_dim = cfg.features_dim
    transformed_features_constraints = cfg.transformed_features_constraints
    save_path = cfg.save_path
    models_features_path = cfg.models_features_path
    
    # Print configuration for verification
    print(f"features_dim: {features_dim}")
    print(f"transformed_features_constraints: {transformed_features_constraints}")
    print(f"save_path: {save_path}")
    
    # Construct paths to feature directories
    transformed_features_path = Path(models_features_path) / f"models_transformed_features/{features_dim}/{transformed_features_constraints}"
    original_features_path = Path(models_features_path) / f"models_original_features/{features_dim}"
    
    # Build features path dictionary
    features_path_dict = {
        "raw_features_path": str(original_features_path),
        "post_transformation_features_path": str(transformed_features_path),
        "vice_features_path": str(Path(cfg.vice_features_path))
    }
    
    # Create descriptive name for output files
    if cfg.include_new_models:
        features_name = f"{features_dim}_dim_{transformed_features_constraints}_additional_models"
    else:
        features_name = f"{features_dim}_dim_{transformed_features_constraints}"

    # Determine which models to include
    models_list = cfg.models if not cfg.include_new_models else cfg.models + cfg.new_models
    
    # Compute RSA metrics
    sim, rdms = create_models_RDMs_and_sim_matrix(features_path_dict, models_list)
    
    # Save results
    save_RDMs_and_sim_matrix(rdms, sim, save_path, features_name)
    
    # Print paths to saved files
    rdms_path = os.path.join(save_path, f"rdms_{features_name}.hdf5")
    sim_path = os.path.join(save_path, f"sim_{features_name}.nc")
    print(f"Saved RDMs as: {rdms_path}, and similarity matrix as: {sim_path}")


if __name__ == "__main__":
    main()







