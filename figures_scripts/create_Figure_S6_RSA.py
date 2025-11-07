"""
Figure S4: Representational Similarity Analysis (RSA) Heatmap

This script generates Figure S4 of the supplementary materials, which shows
pairwise representational similarity between different neural network models
before and after transformation during the model recovery process.

The figure visualizes:
- A heatmap showing whitened Pearson correlation between model representations
- Original representations (in red) vs. transformed representations (in green)
- Similarity patterns revealing clustering by architecture and training paradigm

Key Features:
- Supports both 20-model (paper version) and 30-model (extended) configurations
- Uses whitened Pearson correlation for similarity metric
- Applies custom red-yellow-green colormap for intuitive interpretation
- Shows both raw (original) and post-transformation (aligned) representations

Note on Paper Version:
The version appearing in the paper includes minor aesthetic edits (colors, object 
positions, label adjustments) for improved clarity. These edits do not affect the 
underlying data or scientific results - only the visual presentation was refined 
for publication quality.
"""
# =============================================================================
# Imports and Environment Setup
# =============================================================================

import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(
    PARENT_DIR
)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import numpy as np
import torch
import rsatoolbox as rsa
from tqdm import tqdm
import matplotlib.pyplot as plt 
from typing import Dict
import xarray as xr
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hydra
from omegaconf import DictConfig

# =============================================================================
# Helper Functions
# =============================================================================
def load_datasets(folders: Dict[str,str], model_names, emb):
    """
    Load model representations as rsatoolbox Dataset objects.
    
    This function loads the PyTorch feature vectors for each specified model from
    the given folders and converts them to rsatoolbox Dataset objects for RSA analysis.
    
    Parameters:
        folders (Dict[str,str]): Dictionary mapping version names to data folders
        model_names (list): List of model names to load
        emb (str): Embedding type or subfolder containing the feature vectors
        
    Returns:
        list: List of rsatoolbox Dataset objects, one per model-version combination
        
    Notes:
        - Each dataset includes descriptors with the model name and version
        - Measurements are converted to float32 numpy arrays for memory efficiency
        - The observation descriptors are set to image indices
    """
    datasets = []
    for version_name, data_folder in folders.items():
        folder = os.path.join(data_folder, emb)
        for name in tqdm(model_names, desc="Loading models", unit=f"model_{version_name}"):
            arr = torch.load(os.path.join(folder, f"{name}.pt")
                         ).detach().cpu().numpy().astype(np.float32)
            datasets.append(
                rsa.data.Dataset(
                    measurements=arr,
                    descriptors={'model': f"{name}_{version_name}"},
                    obs_descriptors={'image': np.arange(arr.shape[0])},
                )
            )
    return datasets

def reorder_raw_then_post(da, models_order, raw_sfx='_raw', post_sfx='_post'):
    """
    Reorder a square similarity DataArray for optimal visualization.
    
    This function reorganizes the similarity matrix to group raw (original) and
    post (transformed) representations, following a specified model order. This
    produces a visually clearer heatmap with distinct blocks for raw and post-
    transformation similarities.
    
    The ordering strategy:
    1. All raw representations in models_order (maintaining order)
    2. All post representations in models_order (maintaining order)
    3. Any remaining labels (raw-like first, then post-like)
    
    Parameters:
        da : xarray.DataArray
            Square similarity matrix with coords 'model1' and 'model2'
        models_order : list
            Desired ordering of base model names (without suffixes)
        raw_sfx : str, default='_raw'
            Suffix identifying original representations
        post_sfx : str, default='_post'
            Suffix identifying transformed representations
            
    Returns:
        xarray.DataArray
            Reordered similarity matrix with both dimensions sorted
            
    Notes:
        - Preserves the original relative order of models not in models_order
        - Handles models without suffixes by treating them as raw
        - Both axes (model1 and model2) are reordered identically
    """
    labels = list(da.coords['model1'].values)

    def base(lbl):
        return lbl.replace(raw_sfx, '').replace(post_sfx, '')

    # Group original labels by base name preserving original order
    grouped = {}
    for lbl in labels:
        grouped.setdefault(base(lbl), []).append(lbl)

    raw_order, post_order = [], []

    # Pass 1: follow models_order
    for m in models_order:
        if m in grouped:
            items = grouped.pop(m)
            for lbl in items:
                if lbl.endswith(post_sfx):
                    post_order.append(lbl)
                else:
                    raw_order.append(lbl)  # treat no-suffix as raw

    # Pass 2: remaining bases (preserve original relative order)
    seen = set(raw_order) | set(post_order)
    for lbl in labels:
        if lbl in seen:
            continue
        if lbl.endswith(post_sfx):
            post_order.append(lbl)
        else:
            raw_order.append(lbl)

    ordered = raw_order + post_order
    return da.sel(model1=ordered, model2=ordered)


# =============================================================================
# Main Visualization Function
# =============================================================================
@hydra.main(config_path="../scripts_configurations", config_name="RSA_MDS_figures", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to generate Figure S4 RSA heatmap visualization.
    
    This function creates a representational similarity analysis (RSA) heatmap
    showing pairwise similarities between neural network model representations.
    The visualization helps understand:
    1. How similar different models' representations are to each other
    2. How representations change after transformation (raw vs. post)
    3. Clustering patterns by architecture and training objective
    
    The heatmap uses:
    - Red labels: Original (raw) representations
    - Green labels: Transformed (post) representations  
    - Red-Yellow-Green colormap: Low to high similarity
    - Whitened Pearson correlation as the similarity metric
    
    Configuration Options (via Hydra):
    -----------------------------------
    The script can generate two versions of the figure:
    
    1. Paper version (20 models) - DEFAULT:
       python Figure_S4_RSA.py
       
    2. Extended version (30 models):
       python Figure_S4_RSA.py include_additional_models=True
    
    Parameters:
        cfg : DictConfig
            Hydra configuration containing:
            - include_additional_models: Boolean flag for model count
            - features_dim: Feature dimensionality ('full' or 'PCA_500')
            - transformed_features_constraints: Transformation type
            - models: Base 20 models list
            - new_models: Additional 10 models list
            - formal_names_dict: Mapping of model names to display names
            
    Outputs:
        Saves a PDF file to: Plots/{features_dim}_dim_{constraints}/Figure_S4_RSA.pdf
        With additional models: Plots/{features_dim}_dim_{constraints}_additional_models/Figure_S4_RSA.pdf
        
    Notes:
        - The paper version uses 20 base models for clarity
        - The extended version includes 10 additional models for comprehensive analysis
        - Minor aesthetic edits were applied to the published version (colors, positions)
          but the underlying data and similarity values are unchanged
        - VICE model is included for reference but treated specially in the visualization
        - MUST run build_rsa_sim_matrix.py first to generate similarity data
    """
    
    # =============================================================================
    # Configuration and Model Selection
    # =============================================================================
    
    # Build model list based on configuration
    models_list = (
        cfg.models if not cfg.include_additional_models 
        else cfg.models + cfg.new_models
    )
    
    # Filter formal names to only include selected models
    formal_names_dict = {
        key: value for key, value in cfg.formal_names_dict.items() 
        if key in models_list
    }
    
    # Determine version name for reporting
    version_name = "EXTENDED (30 models)" if cfg.include_additional_models else "PAPER (20 models)"
    
    print(f"\n{'='*70}")
    print(f"Generating Figure S4 RSA - {version_name}")
    print(f"{'='*70}\n")
    
    # =============================================================================
    # Construct Similarity Matrix Path
    # =============================================================================
    
    # Build filename following the convention from build_rsa_sim_matrix.py
    features_dim = cfg.features_dim
    constraints = cfg.transformed_features_constraints
    
    # Construct base filename
    sim_filename = f"sim_{features_dim}_dim_{constraints}"
    
    # Add suffix if using additional models
    if cfg.include_additional_models:
        sim_filename += "_additional_models"
    
    sim_filename += ".nc"
    
    # Full path to similarity matrix
    similarity_path = os.path.join(cfg.rsa_results_dir, sim_filename)
    
    print(f"Loading similarity data from: {similarity_path}")
    
    # Verify file exists
    if not os.path.exists(similarity_path):
        error_msg = f"""
        ERROR: Similarity matrix file not found!
        
        Expected: {similarity_path}
        
        You must first generate the similarity matrix by running:
        python analysis_scripts/build_rsa_sim_matrix.py \\
            features_dim={features_dim} \\
            transformed_features_constraints={constraints} \\
            include_new_models={str(cfg.include_additional_models)}
        
        This will create the required .nc file with RSA similarity data.
        """
        raise FileNotFoundError(error_msg)
    
    # =============================================================================
    # Load and Process Similarity Data
    # =============================================================================
    
    # Load similarity matrix dataarray
    sim_array = xr.open_dataarray(similarity_path)
    sim_array = reorder_raw_then_post(sim_array, models_list)
    sim = sim_array.values
    labels = sim_array.coords['model1'].values
    
    print(f"Similarity matrix shape: {sim.shape}")
    print(f"Number of models: {len(models_list)}")
    print(f"Total representations (raw + post): {len(labels)}")
    
    # =============================================================================
    # Figure Setup and Styling Parameters
    # =============================================================================
    
    # Construct output directory name
    output_subdir = f"{features_dim}_dim_{constraints}"
    if cfg.include_additional_models:
        output_subdir += "_additional_models"
    
    out_figs = os.path.join(cfg.output_dir, output_subdir)
    os.makedirs(out_figs, exist_ok=True)
    
    # Visualization parameters
    ticks_font_size = 4
    line_thickness = 0.5   # Global line thickness for all plot elements
    
    # Figure dimensions (from configuration or defaults)
    fig_width_cm = cfg.get('figure_width_cm', 8)
    fig_height_cm = cfg.get('figure_height_cm', 10)
    

    
    # =============================================================================
    # Create RSA Heatmap
    # =============================================================================
    
    print("\n" + "="*70)
    print("Creating RSA heatmap visualization...")
    print("="*70 + "\n")
    
    # Create figure with maximized plotting area
    fig_rsa = plt.figure(figsize=(fig_width_cm/2.54, fig_height_cm/2.54), 
                         constrained_layout=False)
    ax_rsa = fig_rsa.add_subplot(111)
    ax_rsa.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]
    
    # Create custom red-yellow-green colormap for similarity
    from matplotlib.colors import LinearSegmentedColormap
    cmap_colors = [(0.8392, 0.15294, 0.15686),  # Red (low similarity)
                    (0.95, 0.90, 0.25),         # Yellow (medium)
                    (0.17647, 0.6275, 0.17647)]  # Green (high similarity)
    red_to_green = LinearSegmentedColormap.from_list('red_to_green', cmap_colors)
    
    # Plot similarity matrix as heatmap
    im = ax_rsa.imshow(sim,
                       cmap=red_to_green,
                       vmin=0, vmax=1)
    
    # =============================================================================
    # Label Preparation and Color Coding
    # =============================================================================
    
    # Define colors for raw (original) and post (transformed) representations
    raw_color = '#d62728'  # Red for original representations
    post_color = '#2ca02c'  # Green for transformed representations
    
    # Extract and categorize labels
    raw_indices = []
    post_indices = []
    formal_labels = []
    
    for i, label in enumerate(labels):
        # Extract the base name (without _raw or _post suffix)
        if "_raw" in label:
            base_name = label.replace("_raw", "")
            raw_indices.append(i)
        elif "_post" in label:
            base_name = label.replace("_post", "")
            post_indices.append(i)
        else:
            base_name = label
            raw_indices.append(i)  # Default to raw for models without suffix
        
        # Use formal name if available, otherwise use original
        if base_name in formal_names_dict:
            formal_labels.append(formal_names_dict[base_name])
        else:
            formal_labels.append(base_name)
    
    # =============================================================================
    # Axis Labels and Tick Styling
    # =============================================================================
    
    # Set tick labels on both axes
    ax_rsa.set_xticks(np.arange(len(labels)))
    ax_rsa.set_yticks(np.arange(len(labels)))
    ax_rsa.set_xticklabels(formal_labels, rotation=90, ha='right', 
                           fontsize=ticks_font_size)
    ax_rsa.set_yticklabels(formal_labels, fontsize=ticks_font_size)
    
    # Color-code tick labels (red for raw, green for post)
    for i in raw_indices:
        ax_rsa.get_xticklabels()[i].set_color(raw_color)
        ax_rsa.get_yticklabels()[i].set_color(raw_color)
    
    for i in post_indices:
        ax_rsa.get_xticklabels()[i].set_color(post_color)
        ax_rsa.get_yticklabels()[i].set_color(post_color)
    
    # Apply line thickness to tick marks
    ax_rsa.tick_params(which='both', width=line_thickness)
    
    # =============================================================================
    # Colorbar Configuration
    # =============================================================================
    
    # Add colorbar horizontally above the RSA matrix
    divider = make_axes_locatable(ax_rsa)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=ticks_font_size, width=line_thickness)
    
    # Position colorbar ticks and label at top
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel('Whitened pearson correlation', 
                       fontsize=ticks_font_size+1, labelpad=1)
    
    # =============================================================================
    # Save Figure
    # =============================================================================
    
    # Adjust layout to maximize space usage
    fig_rsa.tight_layout(pad=0.5)
    
    # Save as high-resolution PDF
    fig_path_rsa_pdf = os.path.join(out_figs, f"Figure_S4_RSA.pdf")
    fig_rsa.savefig(fig_path_rsa_pdf, format='pdf', dpi=600)
    plt.close(fig_rsa)
    
    # =============================================================================
    # Summary Report
    # =============================================================================
    
    print("\n" + "="*70)
    print("Figure S4 RSA - Generation Complete")
    print("="*70)
    print(f"Version: {version_name}")
    print(f"Output file: {fig_path_rsa_pdf}")
    print(f"Models included: {len(models_list)}")
    print(f"Total representations (raw + post): {len(labels)}")
    print(f"Figure size: {fig_width_cm:.1f} × {fig_height_cm:.1f} cm")
    print(f"Resolution: 600 DPI")
    print(f"Colormap: Red (low) → Yellow (medium) → Green (high similarity)")
    print("="*70 + "\n")
    
if __name__ == "__main__":
    main()
