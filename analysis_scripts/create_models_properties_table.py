import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import warnings
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from dadapy.data import Data
import hydra 
from omegaconf import DictConfig
import xarray as xr



"""
create_models_properties_table.py

Unified script to compute model property metrics controlled by a YAML configuration file.

IMPORTANT LIMITATIONS:
    This script is NOT dynamic and currently works ONLY for:
    - Full-dimension features (no PCA/dimensionality reduction applied)
    - Full-dimension features with a full_W learned transformation matrix
    
    The script expects specific file naming conventions and transformation types.
    Modifications would be required to support other feature transformations or
    dimensionality reduction schemes.

PREREQUISITES:
    Before running this script, you must have:
    1. RSA (Representational Similarity Analysis) matrix generated for the models
       - Generated from full-dimension features
       - Should contain similarity between _raw and _post versions of each model
    2. Metadata CSV file with model parameters (n_params, embed_dim)
       - Generated using an in-house package used for extracting THINGS features
       - Must include a 'model_name' column
    3. Raw and transformed feature files for all models
       - Stored as .pt (PyTorch tensor) files
       - Named as <model_name>.pt in respective directories

PURPOSE:
    Combines functionality from multiple analysis scripts:
    - represetations_measuerse.py (Two-NN and Gride via DADApy)
    - effective_dimensionality_score.py (Effective Dimensionality / participation ratio)
    - RSA-based alignment shift computation
    
    Merges optional metadata to output a single comprehensive CSV consumed by
    downstream analysis scripts (e.g., Table_S3_regression.py).

OUTPUT CSV COLUMNS (superset; some may be NaN if not provided in metadata):
    - model_name: Name of the model
    - n_params: Number of parameters in the model (from metadata)
    - embed_dim: Embedding dimension (inferred from features)
    - ED_pre: Effective dimensionality of original features
    - ED_post: Effective dimensionality of transformed features
    - delta_ed(post-pre): Change in effective dimensionality
    - pre_twonn: Two-nearest-neighbors ID estimate (original features)
    - post_twonn: Two-nearest-neighbors ID estimate (transformed features)
    - delta_twonn: Change in Two-NN estimate
    - pre_gride: Gride ID measure (original features)
    - post_gride: Gride ID measure (transformed features)
    - delta_gride: Change in Gride ID
    - alignment_induced_shift: Cosine distance between raw and transformed embeddings

USAGE:
    python analysis_scripts/create_models_properties_table.py
    
    Configuration is loaded from: scripts_configurations/models_properties_table.yaml
"""


# Quiet torch load future warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def effective_dimensionality(features: np.ndarray, center: bool = True) -> float:
    """
    Compute effective dimensionality (participation ratio) of feature representations.
    
    The effective dimensionality is calculated as:
        ED = (sum(λ))² / sum(λ²)
    
    where λ are the eigenvalues (explained variance) from PCA.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        center: Whether to mean-center the features before PCA (default: True)
        
    Returns:
        Effective dimensionality (float). Returns np.nan if PCA fails or no variance.
        
    Note:
        This metric quantifies how many principal components are needed to capture
        the variance in the data. Higher values indicate more distributed variance
        across dimensions (higher intrinsic dimensionality).
    """
    if center:
        features = features - np.mean(features, axis=0, keepdims=True)
    pca = PCA(svd_solver="full").fit(features)
    lambdas = pca.explained_variance_
    if lambdas is None or len(lambdas) == 0:
        return np.nan
    return (lambdas.sum() ** 2) / np.sum(lambdas ** 2)


def load_tensor_as_numpy(path: str) -> np.ndarray:
    """
    Load a PyTorch tensor file and convert it to a NumPy array.
    
    Args:
        path: Path to the .pt file containing the tensor
        
    Returns:
        NumPy array representation of the tensor
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file does not exist
        
    Note:
        Supports both PyTorch tensors and NumPy arrays saved as .pt files.
        Tensors are automatically moved to CPU and detached from computation graph.
    """
    t = torch.load(path, map_location='cpu')
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    # fallback if saved as numpy array
    try:
        return np.array(t)
    except Exception:
        raise ValueError(f"Unsupported tensor format at {path}")


def cosine_distance_mean(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compute the mean cosine distance between corresponding rows of two matrices.
    
    This measures how much the embedding vectors have changed between the raw (pre)
    and transformed (post) representations on average.
    
    Args:
        pre: Original features matrix of shape (n_samples, n_features)
        post: Transformed features matrix of shape (n_samples, n_features)
        
    Returns:
        Mean cosine distance across all samples (float in range [0, 2])
        
    Raises:
        ValueError: If pre and post have different shapes
        
    Note:
        Cosine distance = 1 - cosine_similarity
        - Distance of 0 means vectors are identical in direction
        - Distance of 1 means vectors are orthogonal
        - Distance of 2 means vectors point in opposite directions
    """
    if pre.shape != post.shape:
        raise ValueError(f"Shape mismatch pre {pre.shape} vs post {post.shape}")
    # Normalize rows
    def _row_norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n
    pre_n = _row_norm(pre)
    post_n = _row_norm(post)
    sims = np.sum(pre_n * post_n, axis=1)
    dists = 1.0 - sims
    return float(np.mean(dists))


def get_models_pre_post_embed_dist(model_name: str, data: xr.DataArray) -> float:
    """
    Get the traveled distance (alignment-induced shift) from RSA similarity matrix.
    
    Computes the distance between the raw and post-transformation versions of a model
    based on their representational similarity.
    
    Args:
        model_name: Name of the model (without _raw/_post suffix)
        data: xarray DataArray containing the similarity matrix with dimensions
              (model1, model2). Expected to have entries like "<model>_raw" and
              "<model>_post" in the model coordinates.
        
    Returns:
        Distance (1 - similarity) between raw and post versions of the model
        
    Note:
        This metric quantifies how much the learned transformation (e.g., full_W matrix)
        has shifted the representational geometry of the model, as measured by the change
        in similarity to other models or itself.
        
        LIMITATION: This function assumes the RSA matrix was built with _raw and _post
        suffixes for model names, which is specific to full-dimension transformations.
    """
    # LIMITATION: Hardcoded naming convention for RSA matrix entries
    # Expects entries like "modelname_raw" and "modelname_post"
    # This is specific to full_W transformation workflow
    sim = data.sel(model1=f"{model_name}_raw", model2=f"{model_name}_post").values.item()
    dist = 1 - sim
    return dist


def compute_twonn(features: np.ndarray) -> tuple:
    """
    Compute intrinsic dimensionality using the Two-Nearest-Neighbors (Two-NN) method.
    
    The Two-NN estimator leverages the ratio of distances to the first and second
    nearest neighbors to estimate the intrinsic dimensionality of the data manifold.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        
    Returns:
        Tuple of (estimate, error, distance):
            - estimate: Estimated intrinsic dimensionality (float)
            - error: Estimation error/uncertainty (float)
            - distance: Characteristic distance scale (float)
        Returns (np.nan, np.nan, np.nan) if computation fails.
        
    Note:
        Uses the DADApy library's implementation. Requires computing distances
        up to k=100 neighbors for robust estimation.
        
    References:
        Facco et al. (2017). "Estimating the intrinsic dimension of datasets
        by a minimal neighborhood information." Scientific Reports.
    """
    try:
        data = Data(features)
        data.compute_distances(maxk=100)
        est, err, dist = data.compute_id_2NN()
        return est, err, dist
    except Exception as e:
        logger.error(f"Two-NN failed: {e}")
        return np.nan, np.nan, np.nan


def compute_gride(features: np.ndarray, range_max: int) -> float:
    """
    Compute intrinsic dimensionality using the Gride (Gradient-based ID Estimator) method.
    
    Gride estimates intrinsic dimensionality by analyzing how the number of neighbors
    scales with distance across different spatial scales.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        range_max: Maximum neighborhood size for scaling analysis (typically 64)
        
    Returns:
        Median intrinsic dimensionality estimate across scales (float).
        Returns np.nan if computation fails.
        
    Note:
        Uses the DADApy library's implementation. The median across scales provides
        a robust estimate that is less sensitive to local variations in density.
        
        The range_max parameter should be chosen based on dataset size:
        - Larger datasets can support larger range_max values
        - Typical value: 64 for medium-sized datasets
        
    References:
        Rozza et al. (2012). "Novel high intrinsic dimensionality estimators."
        Machine Learning.
    """
    try:
        data = Data(features)
        data.compute_distances(maxk=range_max)
        id_list, id_err_list, id_distance_list = data.return_id_scaling_gride(range_max=range_max)
        if len(id_list) == 0:
            return np.nan
        return float(np.median(id_list))
    except Exception as e:
        logger.error(f"Gride failed: {e}")
        return np.nan


def build_properties_for_model(model: str, cfg) -> dict:
    """
    Compute all configured property metrics for a single model.
    
    Loads both raw (pre-transformation) and transformed (post-transformation) features
    for the model and computes various dimensionality and geometry metrics.
    
    Args:
        model: Model name (without .pt extension)
        cfg: Hydra configuration object with paths and metric settings
        
    Returns:
        Dictionary containing computed metrics for the model. Only includes metrics
        that are enabled in the configuration (via cfg.metrics flags).
        Always includes:
            - model_name: Name of the model
            - embed_dim: Embedding dimension (inferred from feature shape)
        Optionally includes (based on cfg.metrics flags):
            - ED_pre, ED_post, delta_ed(post-pre): Effective dimensionality metrics
            - pre_twonn, post_twonn, delta_twonn: Two-NN ID estimates
            - pre_gride, post_gride, delta_gride: Gride ID estimates
            - alignment_induced_shift: Cosine distance between raw and transformed
            
    Raises:
        FileNotFoundError: If raw or post feature files are missing
        
    Note:
        LIMITATION: This function expects feature files in specific locations:
        - Raw features: cfg.paths.features_raw_path/<model>.pt
        - Transformed features: cfg.paths.features_post_path/<model>.pt
        
        These paths are currently hardcoded for full-dimension features with
        full_W transformation matrices only.
    """
    # LIMITATION: Path structure assumes full-dimension features only
    # For PCA-reduced or other transformations, paths would need to be adjusted
    raw_p = os.path.join(cfg.paths.features_raw_path, f"{model}.pt")
    post_p = os.path.join(cfg.paths.features_post_path, f"{model}.pt")
    if not os.path.exists(raw_p):
        raise FileNotFoundError(f"Missing raw features for {model}: {raw_p}")
    if not os.path.exists(post_p):
        raise FileNotFoundError(f"Missing post features for {model}: {post_p}")

    # Load features
    pre = load_tensor_as_numpy(raw_p)
    post = load_tensor_as_numpy(post_p)

    # Initialize result dictionary - only add metrics that are computed
    result = {
        'model_name': model,
        'embed_dim': int(pre.shape[1])  # Always include embed_dim (inferred from features)
    }

    # Effective dimensionality
    if cfg.metrics.compute_effective_dimensionality:
        ED_pre = effective_dimensionality(pre)
        ED_post = effective_dimensionality(post)
        result['ED_pre'] = ED_pre
        result['ED_post'] = ED_post
        result['delta_ed(post-pre)'] = ED_post - ED_pre

    # Two-NN
    if cfg.metrics.get('compute_two_nn', False):
        pre_twonn, pre_twonn_err, _ = compute_twonn(pre)
        post_twonn, post_twonn_err, _ = compute_twonn(post)
        result['pre_twonn'] = pre_twonn
        result['post_twonn'] = post_twonn
        result['delta_twonn'] = (post_twonn - pre_twonn) if (not np.isnan(pre_twonn) and not np.isnan(post_twonn)) else np.nan

    # Gride
    if cfg.metrics.compute_gride:
        pre_gride = compute_gride(pre, cfg.metrics.gride_range_max)
        post_gride = compute_gride(post, cfg.metrics.gride_range_max)
        result['pre_gride'] = pre_gride
        result['post_gride'] = post_gride
        result['delta_gride'] = (post_gride - pre_gride) if (not np.isnan(pre_gride) and not np.isnan(post_gride)) else np.nan

    # Embedding distance (cosine)
    if cfg.metrics.compute_embedding_pre_post_distance:
        result['alignment_induced_shift'] = cosine_distance_mean(pre, post)

    return result

@hydra.main(config_path="../scripts_configurations", config_name="models_properties_table.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to build a comprehensive models properties table.
    
    This function orchestrates the entire pipeline:
    1. Loads model list from configuration
    2. Computes properties for each model (dimensionality, geometry metrics)
    3. Optionally loads and merges alignment shift from RSA similarity matrix
    4. Optionally merges metadata (n_params, embed_dim) from external CSV
    5. Saves final comprehensive table as CSV
    
    Args:
        cfg: Hydra configuration loaded from models_properties_table.yaml
        
    Configuration Structure:
        - cfg.models: List of base model names
        - cfg.new_models: List of additional model names (if include_new_models=True)
        - cfg.include_new_models: Whether to include new_models list
        - cfg.paths.features_raw_path: Directory with raw feature .pt files
        - cfg.paths.features_post_path: Directory with transformed feature .pt files
        - cfg.paths.metadata_csv_path: Optional CSV with model metadata (n_params, etc.)
        - cfg.paths.rsa_xarray_path: Optional RSA similarity matrix (.nc file)
        - cfg.paths.output_dir: Directory for output CSV
        - cfg.metrics.*: Boolean flags for which metrics to compute
        
    Output:
        Saves models_properties.csv to cfg.paths.output_dir with columns ordered for
        downstream regression analysis.
        
    Note:
        PREREQUISITES:
        1. RSA matrix must exist if compute_alignment_shift_from_rsa=True
        2. Metadata CSV must exist if specified (should contain 'model_name' column)
        3. All feature files must exist in the specified directories
        
        LIMITATION: Works only for full-dimension features with full_W transformations.
    """
    
    # LIMITATION: Model list is loaded from shared configuration
    # Currently configured for full-dimension feature workflow only
    # Resolve models list
    models = list(cfg.models) if cfg.include_new_models == False else list(cfg.models) + list(cfg.new_models)
    models.sort()
    # Prepare output dir
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Iterate through all models and compute their properties
    # Each model is processed independently to allow partial completion if some models fail
    for m in models:
        logger.info(f"Processing {m}")
        try:
            # Compute all metrics for this model (ED, Gride, cosine distance, etc.)
            # Returns a dictionary with all computed properties
            rows.append(build_properties_for_model(m, cfg))
        except Exception as e:
            # If processing fails for any reason (missing files, computation errors, etc.),
            # log the error but continue processing other models
            logger.error(f"Failed {m}: {e}")
            # Add a minimal row with just the model name to preserve the model in the output
            # This ensures we can track which models failed and potentially debug later
            rows.append({'model_name': m})

    # Convert list of dictionaries to a pandas DataFrame
    # Each row represents one model, columns are the various metrics
    # Models that failed will have NaN values for all metrics except model_name
    df = pd.DataFrame(rows)

    # Compute alignment-induced representational shift from RSA if configured
    # PREREQUISITE: RSA matrix must exist and must have been built from full-dimension features
    # with _raw and _post model name suffixes
    rsa_path = getattr(cfg.paths, 'rsa_xarray_path', None)
    if rsa_path and cfg.metrics.get('compute_alignment_shift_from_rsa', True):
        try:
            logger.info(f"Loading RSA similarity matrix from {rsa_path}")
            ds = xr.open_dataset(rsa_path)
            
            # Find the similarity matrix variable (should be the main data variable)
            sim_var = None
            for var_name in ds.data_vars:
                if ds[var_name].ndim == 2:  # Should be 2D (model1, model2)
                    sim_var = var_name
                    break
            
            if sim_var is None:
                raise ValueError("Could not find 2D similarity matrix in RSA dataset")
            
            logger.info(f"Using similarity matrix variable: {sim_var}")
            sim_data = ds[sim_var]
            
            # Compute alignment shifts using the simple approach
            model_to_shift = {}
            for model_name in df['model_name']:
                try:
                    shift = get_models_pre_post_embed_dist(model_name, sim_data)
                    model_to_shift[model_name] = shift
                    logger.debug(f"Model {model_name}: alignment shift = {shift:.4f}")
                except Exception as e:
                    logger.warning(f"Could not compute alignment shift for {model_name}: {e}")
                    model_to_shift[model_name] = np.nan
            
            # Attach to dataframe
            df['alignment_induced_shift'] = df['model_name'].map(model_to_shift)
            logger.info(f"Computed alignment shifts for {len([v for v in model_to_shift.values() if not np.isnan(v)])} models")

        except Exception as e:
            logger.warning(f"Failed to compute alignment-induced shift from RSA: {e}")

    # Add the parameters number from metadata if provided
    # PREREQUISITE: Metadata CSV must exist and must have been generated using in-house
    # package used for THINGS feature extraction
    meta_path = cfg.paths.metadata_csv_path
    if meta_path and os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if 'model_name' not in meta.columns:
            logger.warning("metadata_csv_path provided but missing 'model_name' column; skipping merge")
        else:
            df = df.merge(meta, on='model_name', how='left', suffixes=('', '_meta'))
            # Prefer base columns when present; fill from metadata if missing
            for col in meta.columns:
                if col != 'model_name' and f'{col}_meta' in df.columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[f'{col}_meta'])
                    else:
                        df[col] = df[f'{col}_meta']
            # Drop duplicate metadata columns
            df.drop(columns=[c for c in df.columns if c.endswith('_meta')], inplace=True)
            logger.info(f"Merged metadata from {meta_path}")
    
    # OPTIMIZATION: Build column order dynamically based on what's actually in the dataframe
    # This ensures we only include columns that were computed or merged
    preferred_cols = [
        'model_name',
        'n_params',                     # #Parameters (from metadata)
        'embed_dim',                    # #Features (penultimate layer units)
    ]
    
    # Add metric columns in logical order, only if they exist in the dataframe
    metric_cols_order = [
        'ED_pre',                       # Original features ED
        'ED_post',                      # Transformed features ED
        'delta_ed(post-pre)',           # Change in ED
        'pre_gride',                    # Original features Gride ID
        'post_gride',                   # Transformed features Gride ID
        'delta_gride',                  # Change in Gride ID
        'pre_twonn',                    # Original features Two-NN ID
        'post_twonn',                   # Transformed features Two-NN ID
        'delta_twonn',                  # Change in Two-NN ID
        'alignment_induced_shift',      # Cosine distance between embeddings / Alignment-induced shift
    ]
    
    # Build final column list: base cols + metrics that exist + any extras
    cols = [c for c in preferred_cols if c in df.columns]
    cols += [c for c in metric_cols_order if c in df.columns]
    # Add any remaining columns not in our preferred lists (extras from metadata, etc.)
    cols += [c for c in df.columns if c not in cols]
    
    df = df[cols]

    out_csv = out_dir / 'models_properties.csv'
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved models properties to {out_csv}")
    logger.info(f"Output contains {len(df)} models with columns: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()


