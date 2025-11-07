"""
================================================================================
THINGS Odd-One-Out Dataset Preprocessing
================================================================================

Purpose:
    Converts raw THINGS behavioral data from CSV format into PyTorch tensors
    for efficient loading during model training and evaluation. Separates data
    into train and test sets based on noise ceiling assignment.

Context in Pipeline:
    This is the FIRST preprocessing step in the analysis pipeline:
    1) Reads raw THINGS behavioral CSV with human triplet judgments
    2) Converts to tensors with 0-based indexing (CSV uses 1-based)
    3) Splits data into train set and noise ceiling test sets 1-3
    4) Creates three output tensors per dataset:
       - Triplets: Which images form each triplet [N×3]
       - Answer positions: Which position was chosen as odd-one-out [N×1]
       - Answer image indices: Which image ID was chosen [N×1]
    5) Saves summary statistics for dataset verification

Inputs:
    - Data/Things_data/behavior_data/triplets_large_final_correctednc_correctedorder.csv
      Columns: image1, image2, image3, choice, subject_id, noise_ceiling
      - image1/2/3: 1-based concept indices (1-1854)
      - choice: 1-based position of odd-one-out (1, 2, or 3)
      - noise_ceiling: 0=train, 1/2/3=test set assignment
      
      Download from: https://osf.io/f5rn6/ (data/triplet_dataset/)
      THINGS images: https://osf.io/jum2f/ (THINGS image database)

Outputs (all in Data/Things_data_preprocessed/):
    For each dataset (train_set, testset1/2/3_noise_ceiling, all):
        * Things_odd_one_out_triplets_{dataset}.pt: [N×3] triplet indices
        * Things_odd_one_out_answers_positions_{dataset}.pt: [N] choice positions (0/1/2)
        * Things_odd_one_out_answers_image_index_{dataset}.pt: [N] chosen image IDs
    
    Summary file:
        * things_datasets_info.csv: Statistics per dataset
          (n_triplets, n_unique_triplets, n_unique_images)

Data Statistics (typical THINGS dataset):
    - Training set: ~4.2M judgments, ~190K unique triplets
    - Test set 2: ~250K judgments, ~12K unique triplets (used for noise ceiling)
    - Test set 3: ~250K judgments, ~12K unique triplets
    - All combined: ~4.7M judgments

Indexing Convention:
    CSV uses 1-based indexing (images 1-1854)
    Output tensors use 0-based indexing (images 0-1853)
    This matches PyTorch's standard indexing and allows direct use with
    torch.nn.Embedding and feature tensor indexing.

Usage:
    python analysis_scripts/odd_one_out_text_preprocess.py
    
    Requires configuration via: scripts_configurations/odd_one_out_preprocess.yaml

Dependencies:
    - Raw THINGS behavioral CSV must be downloaded
    - No other preprocessing required (this is the starting point)

Related Scripts:
    - things_noise_ceiling_eval.py: Uses these tensors to compute noise ceiling
    - evaluate_THINGS_OOO_accuracy_CV.py: Uses train/test tensors for model eval
    - create_data_generating_models.py: Uses train set for fitting W matrices

Verification:
    The script performs sanity checks:
    - Triplet tensors match image index arrays in first 3 columns
    - Number of triplets equals number of answers
    - Image indices are within valid range [0, 1853]

================================================================================
"""
import os
import sys
from pathlib import Path

# Set up paths to allow relative imports
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import pandas as pd
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

def create_tensors_from_df(df: pd.DataFrame, dataset_id:int):
    """
    Creates three tensors from the dataframe for a specific dataset:
    1) position_array - NX1 tensor with the odd one out position(index) in the triplet 
    2) image_index_array - NX1 tensor with the odd one out image index in the general images index 
    3) triplets_array - NX3 tensor with the triplets images index  
    We do this to separate triplets that are used to compute the noise ceiling from those that are not.
    Args:
        df: DataFrame containing the triplets data
        dataset_id: Dataset identifier to filter ('testset1', 'testset2', 'testset3', 'trainset1', or numeric code)
        
    Returns:
        tuple: (position_tensor, image_index_tensor)
    """
    # Filter dataframe for specific dataset
    if dataset_id == 'all':
        dataset_df = df.copy()
    else:
        dataset_df = df[df['noise_ceiling'] == dataset_id].copy()
    
    # Convert from 1-based to 0-based indexing
    dataset_df['image1'] = dataset_df['image1'] - 1
    dataset_df['image2'] = dataset_df['image2'] - 1 
    dataset_df['image3'] = dataset_df['image3'] - 1
    dataset_df['choice'] = dataset_df['choice'] - 1
    
    # Create position tensor
    position_array  = pd.DataFrame({
        'image1': dataset_df['image1'],
        'image2': dataset_df['image2'], 
        'image3': dataset_df['image3'],
        'odd_position': dataset_df['choice']
    }).values
    
    # Create image index tensor
    # Get the odd one out image based on position
    odd_image_idx = []
    for i,odd_position in enumerate(position_array[:,3]):
        odd_image_idx.append(position_array[i,odd_position])
    
    # Convert to numpy array for consistency
    odd_image_idx = np.array(odd_image_idx)
    image_index_array = pd.DataFrame({
        'image1': dataset_df['image1'],
        'image2': dataset_df['image2'],
        'image3': dataset_df['image3'], 
        'odd_image': odd_image_idx
    }).values
    
    #convert to torch 3 tensors of NX3,NX1,NX1
    position_tensor = torch.tensor(position_array[:,3])
    image_index_tensor = torch.tensor(image_index_array[:,3])
    triplets_tensor = torch.tensor(position_array[:,0:3])
    #sanity check
    assert triplets_tensor.shape[0] == image_index_array.shape[0] == position_tensor.shape[0] #check if they have the same number of rows
    assert torch.allclose(triplets_tensor, torch.tensor(image_index_array[:,0:3]))  # Check if triplets_tensor matches the first two columns of image_index_tensor
    
    
    return triplets_tensor,position_tensor,image_index_tensor


@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="odd_one_out_preprocess")
def main(cfg: DictConfig):
    """Main preprocessing function using Hydra configuration."""
    
    # Load behavioral data from configured path
    print(f"Loading behavioral data from: {cfg.paths.behavior_csv}")
    df = pd.read_csv(cfg.paths.behavior_csv, sep="\t")
    
    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    # Process each dataset split from configuration
    all_datasets_info = []
    for split_name, split_config in cfg.dataset_splits.items():
        print(f"\n{'='*60}")
        print(f"Processing: {split_name}")
        print(f"Description: {split_config.description}")
        print(f"{'='*60}")
        
        triplets_tensor, position_tensor, image_index_tensor = create_tensors_from_df(
            df, split_config.noise_ceiling_id
        )
        
        # Construct output paths using templates
        triplets_path = os.path.join(
            cfg.paths.output_dir, 
            cfg.output_tensors.triplets_template.format(dataset=split_name)
        )
        positions_path = os.path.join(
            cfg.paths.output_dir,
            cfg.output_tensors.positions_template.format(dataset=split_name)
        )
        indices_path = os.path.join(
            cfg.paths.output_dir,
            cfg.output_tensors.image_indices_template.format(dataset=split_name)
        )
        
        # Save tensors
        torch.save(triplets_tensor, triplets_path)
        torch.save(position_tensor, positions_path)
        torch.save(image_index_tensor, indices_path)
        
        # Validation checks if enabled
        if cfg.validation.check_tensor_shapes:
            assert triplets_tensor.shape[0] == position_tensor.shape[0] == image_index_tensor.shape[0]
        
        if cfg.validation.check_index_ranges:
            assert torch.all(triplets_tensor >= 0) and torch.all(triplets_tensor < cfg.N_concepts)
        
        # Compute statistics
        sorted_triplets_tensor, _ = torch.sort(triplets_tensor, dim=1)
        n_unique_triplets = len(torch.unique(sorted_triplets_tensor, dim=0))
        n_unique_concepts = len(torch.unique(triplets_tensor))
        
        if cfg.validation.verbose:
            print(f"Dataset: {split_name}")
            print(f"  Number of triplets: {triplets_tensor.shape[0]:,}")
            print(f"  Number of unique triplets: {n_unique_triplets:,}")
            print(f"  Number of unique concepts used: {n_unique_concepts}")
            print(f"  Saved to: {cfg.paths.output_dir}")
        
        all_datasets_info.append({
            'dataset_name': split_name,
            'number_of_triplets': triplets_tensor.shape[0],
            'number_of_answers_positions': position_tensor.shape[0], 
            'number_of_unique_items_used_to_create_triplets': n_unique_concepts,
            'number_of_unique_triplets': n_unique_triplets
        })
    
    # Save summary statistics
    df_info = pd.DataFrame(all_datasets_info)
    df_info.to_csv(cfg.paths.summary_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {cfg.paths.summary_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
