"""
Utility Functions Module for Triplet-Based Models

This module provides various utility functions for triplet-based models, including:
1. Optimizer configuration helpers
2. Feature extraction utilities
3. Batch size calculation for memory-efficient processing
4. Triplet generation and manipulation functions
5. Dataset partitioning utilities for cross-validation
6. Model instantiation utilities

These utilities are designed to support the implementation of cognitive models
that work with triplet-based data and neural network representations.
"""

#Add to path the parents folder
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch.optim as optim
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from triplets_data_set import ConceptsDataSet
from tqdm import tqdm
import os
from triplets_data_set import OddOneOutDataset
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict
import numpy as np
import pickle

def set_optimizer(optimizer_name, model_parameters, **kwargs):
    """
    Set up an optimizer based on the given choice.

    Parameters:
    - optimizer_name (str): The name of the optimizer ('LBFGS', 'Adam', etc.)
    - model_parameters (iterable): The parameters of the model.
    - **kwargs: Other keyword arguments for the optimizer.

    Returns:
    - optimizer: The initialized optimizer.
    """
    iter = 150
    multi = 3
    if optimizer_name == 'LBFGS':
        return optim.LBFGS(
            model_parameters,
            lr=kwargs.get('lr', 1.0),
            max_iter=kwargs.get('max_iter', iter),
            max_eval=kwargs.get('max_eval', iter*multi),
            tolerance_grad=kwargs.get('tolerance_grad', 1e-7),
            tolerance_change=kwargs.get('tolerance_change', 1e-4),
            history_size=kwargs.get('history_size', 50),
            line_search_fn=kwargs.get('line_search_fn', "strong_wolfe")
        )

    elif optimizer_name == 'Adam':
        return optim.Adam(
            model_parameters,
            lr=kwargs.get('lr', 0.001),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-08),
            weight_decay=kwargs.get('weight_decay', 0),
            amsgrad=kwargs.get('amsgrad', False)
        )

    elif optimizer_name == 'SGD':
        return optim.SGD(
            model_parameters,
            lr=kwargs.get('lr', 0.01),
            momentum=kwargs.get('momentum', 0),
            dampening=kwargs.get('dampening', 0),
            weight_decay=kwargs.get('weight_decay', 0),
            nesterov=kwargs.get('nesterov', False)
        )

    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(
            model_parameters,
            lr=kwargs.get('lr', 0.01),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-08),
            weight_decay=kwargs.get('weight_decay', 0),
            momentum=kwargs.get('momentum', 0),
            centered=kwargs.get('centered', False)
        )

    elif optimizer_name == 'Adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=kwargs.get('lr', 0.01),
            lr_decay=kwargs.get('lr_decay', 0),
            weight_decay=kwargs.get('weight_decay', 0),
            initial_accumulator_value=kwargs.get('initial_accumulator_value', 0)
        )

    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(
            model_parameters,
            lr=kwargs.get('lr', 1.0),
            rho=kwargs.get('rho', 0.9),
            eps=kwargs.get('eps', 1e-06),
            weight_decay=kwargs.get('weight_decay', 0)
        )

    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized!")

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle batches of PIL images.
    
    Args:
        batch: List of items from the dataset
        
    Returns:
        list: The batch as a list
    """
    #a specific function for the loader to work with list of PIL images
    return list(batch)


def calculate_batch_size(n_triplets: int, feature_dim: int, max_memory_gb: float = 12.0) -> int:
    """
    Calculate the maximum batch size that fits within the specified GPU memory limit.
    
    Args:
        n_triplets: Total number of triplets
        feature_dim: Dimension of the feature vectors
        max_memory_gb: Maximum GPU memory to use in gigabytes
    
    Returns:
        Optimal batch size
    """
    # Memory calculations (in bytes):
    bytes_per_float = 4  # float32
    
    # For matmul method:
    # 1. Selected features: batch_size × 3 × feature_dim
    # 2. Similarity matrix: batch_size × 3 × 3
    # 3. Output: batch_size × 3
    # Plus some buffer for computation
    
    max_memory_bytes = max_memory_gb * 1024**3  # Convert GB to bytes
    memory_per_triplet = (3 * feature_dim + 9 + 3) * bytes_per_float
    
    # Add 20% buffer for computational overhead
    safe_memory = max_memory_bytes * 0.4  # More conservative
    
    batch_size = int(safe_memory / memory_per_triplet)
    
    # Round down to nearest 1000 for good measure
    batch_size = (batch_size // 1000) * 1000
    
    return max(1000, min(batch_size, n_triplets))

def consolidate_choice_counts(triplet_set, choice_counts):
    """
    Given a tensor of choice counts, consolidate the counts for each pair of choices,
    so each row is a unique triplet. This includes permutations.
    
    Args:
        triplet_set (torch.Tensor): A tensor of shape (n_triplets, 3) containing the triplet image indices
        choice_counts (torch.Tensor): A tensor of shape (n_triplets, n_choices)

    Returns:
        tuple: 
            - unique_triplets (torch.Tensor): A tensor of shape (n2_triplets, 3) where n2_triplets is the number of unique triplets.
            - consolidated_choice_counts (torch.Tensor): A tensor of shape (n2_triplets, n_choices)
            where n2_triplets is the number of unique triplets.
    """

    assert isinstance(choice_counts, torch.Tensor), "choice_counts must be a tensor"
    assert choice_counts.ndim == 2, "choice_counts must be a 2D tensor"
    n_triplets, n_choices = choice_counts.shape

    assert isinstance(triplet_set, torch.Tensor), "triplet_set must be a tensor"
    assert triplet_set.ndim == 2, "triplet_set must be a 2D tensor"
    assert triplet_set.shape == choice_counts.shape, "triplet_set and choice_counts must have the same shape"

    # Sort each triplet and get the indices
    s_triplet_set, indices = torch.sort(triplet_set, dim=1)

    # Reorder choice_counts according to the sorted indices
    s_choice_counts = torch.gather(choice_counts, 1, indices)

    # Find unique triplets and their inverse indices
    unique_triplets, inverse_indices = torch.unique(s_triplet_set, dim=0, return_inverse=True)

    # Initialize tensor for consolidated choice counts
    consolidated_choice_counts = torch.zeros((unique_triplets.size(0), choice_counts.size(1)), dtype=choice_counts.dtype, device=choice_counts.device)
    consolidated_choice_counts = consolidated_choice_counts.scatter_add(0, inverse_indices.unsqueeze(1).expand(-1, choice_counts.size(1)), s_choice_counts)

    return unique_triplets, consolidated_choice_counts

def generate_unique_triplets(N_concepts: int, N_triplets: int, seed: Optional[int]) -> torch.Tensor:
    """
    Generate N_triplets unique triplets from N_concepts.
    Each triplet contains 3 different numbers from 0 to N_concepts-1.
    
    Args:
        N_concepts (int): Number of concepts to choose from (0 to N_concepts-1)
        N_triplets (int): Number of unique triplets to generate
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Tensor of shape (N_triplets, 3) containing unique triplets
        
    Raises:
        ValueError: If N_triplets exceeds the maximum possible number of unique triplets
    """
    # Calculate maximum possible unique triplets
    max_possible = (N_concepts * (N_concepts-1) * (N_concepts-2)) // 6
    
    if N_triplets > max_possible:
        raise ValueError(f"Cannot generate {N_triplets} unique triplets. Maximum possible is {max_possible}")
    
    # Initialize empty tensor to store triplets
    triplets = torch.zeros((N_triplets, 3), dtype=torch.long)
    
    # Keep track of generated triplets using a set
    seen = set()
    i = 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    while i < N_triplets:
        # Sample 3 different numbers
        sample = torch.randperm(N_concepts, generator=generator)[:3]
        # Sort to ensure consistent ordering
        sample, _ = torch.sort(sample)
        
        # Convert to tuple for set operations
        triplet = tuple(sample.tolist())
        
        # Only add if we haven't seen this triplet before
        if triplet not in seen:
            triplets[i] = sample
            seen.add(triplet)
            i += 1
            
    return triplets

def generate_unique_triplets_subset_of_concepts(
    concepts: torch.Tensor,
    N_triplets: int,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    seed: Optional[int] = None
    ) -> torch.Tensor:
    """
    Generate N_triplets unique triplets from a subset of concepts.
    Each triplet contains 3 different numbers from the provided concepts tensor.
    
    Args:
        concepts (torch.Tensor): 1D tensor of concept indices to choose from
        N_triplets (int): Number of unique triplets to generate
        triplets_to_exclude (torch.Tensor): NX3 tensor of triplets to exclude
        seed (int): seed for the random number generator
    Returns:
        torch.Tensor: Tensor of shape (N_triplets, 3) containing unique triplets
    """
    # Make sure the concepts are unique 
    if type(concepts) == np.ndarray:
        concepts = torch.from_numpy(concepts)
    concepts = torch.unique(concepts)
    N_concepts = len(concepts)
    if N_concepts < 3:
        raise ValueError("Cannot generate triplets. Minimum 3 concepts are required.")
    
    # Process triplets to exclude
    excluded_set = set()
    if triplets_to_exclude is not None:
        assert isinstance(triplets_to_exclude, torch.Tensor), "triplets_to_exclude must be a tensor"
        assert triplets_to_exclude.ndim == 2, "triplets_to_exclude must be a 2D tensor"
        assert triplets_to_exclude.shape[1] == 3, "triplets_to_exclude must have 3 columns"
        
        # Filter triplets_to_exclude to only include valid concept combinations
        valid_mask = torch.all(torch.isin(triplets_to_exclude, concepts), dim=1)
        triplets_to_exclude = triplets_to_exclude[valid_mask]
        
        # Add sorted tuples to excluded set
        for triplet in triplets_to_exclude:
            sorted_triplet = tuple(sorted(triplet.tolist()))
            excluded_set.add(sorted_triplet)
    
    # Calculate maximum possible unique triplets
    max_possible = ((N_concepts * (N_concepts-1) * (N_concepts-2)) // 6) - len(excluded_set)
    
    if N_triplets > max_possible:
        raise ValueError(f"Cannot generate {N_triplets} unique triplets. Maximum possible is {max_possible}")
    
    # Initialize empty tensor to store triplets
    triplets = torch.zeros((int(N_triplets), 3), dtype=torch.long)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    
    # Keep track of generated triplets using a set
    seen = set()
    i = 0
    
    while i < N_triplets:
        # Sample 3 different indices from concepts
        sample_idx = torch.randperm(N_concepts, generator=generator)[:3]
        sample = concepts[sample_idx]
        # Sort to ensure consistent ordering
        sample, _ = torch.sort(sample)
        
        # Convert to tuple for set operations
        triplet = tuple(sample.tolist())
        
        # Only add if we haven't seen this triplet before and it's not in excluded set
        if triplet not in seen and triplet not in excluded_set:
            triplets[i] = sample
            seen.add(triplet)
            i += 1
            
    return triplets


import torch
from typing import Optional

def generate_unique_triplets_subset_of_concepts_new(
    concepts: torch.Tensor,
    N_triplets: int,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_factor: int = 5
) -> torch.Tensor:
    """
    Generate N_triplets unique triplets from a subset of concepts, on GPU for speed.
    Each triplet (a,b,c) is returned in ascending order (a <= b <= c) and is guaranteed
    not to appear in `triplets_to_exclude` (if provided).

    This function encodes triplets into a single integer key for fast membership checks
    via torch.searchsorted, rather than row-by-row set membership in Python.

    Args:
        concepts (torch.Tensor):
            1D tensor of concept indices from which triplets are sampled. 
            For best performance, pass it on GPU already if possible.
        N_triplets (int):
            Number of unique triplets to generate.
        triplets_to_exclude (torch.Tensor, optional):
            Nx3 tensor of triplets to exclude. Each row is sorted or unsorted (we will sort).
        seed (int, optional):
            Seed for the random number generator (GPU if device='cuda').
        device (str, optional):
            GPU or CPU device (e.g., "cuda" or "cpu"). If None, defaults to `concepts.device`.
        batch_factor (int):
            Factor controlling batch size. 
            If we want N_triplets, each batch samples `N_triplets * batch_factor` rows.

    Returns:
        torch.Tensor:
            (N_triplets, 3) of unique triplets in ascending order, on `device`.
    """

    # --------------------------------------------------------------------------
    # 0) Helper: Encoding and decoding
    # --------------------------------------------------------------------------
    def encode_triplets(trips: torch.Tensor, M: int) -> torch.Tensor:
        # trips: shape (N, 3), sorted rowwise
        # key = a*M^2 + b*M + c
        return trips[:, 0] * (M**2) + trips[:, 1] * M + trips[:, 2]
    
    # --------------------------------------------------------------------------
    # 1) Basic checks, device setup
    # --------------------------------------------------------------------------
    if not torch.is_tensor(concepts):
        concepts = torch.tensor(concepts, dtype=torch.long)
    concepts = torch.unique(concepts)  # ensure uniqueness of concept IDs

    N_concepts = concepts.size(0)
    if N_concepts < 3:
        raise ValueError("Cannot generate triplets. Need at least 3 distinct concepts.")

    # Convert N_triplets to an int if it's not already
    N_triplets = int(N_triplets)

    # Decide on device
    if device is None:
        device = concepts.device
    concepts = concepts.to(device)

    # For encoding to be collision-free:
    # We'll set M = max(concepts)+1
    # Check for potential overflow if concept IDs are very large
    M = torch.max(concepts).item() + 1

    # --------------------------------------------------------------------------
    # 2) Build the excluded_keys from triplets_to_exclude
    # --------------------------------------------------------------------------
    excluded_keys = None
    if triplets_to_exclude is not None and len(triplets_to_exclude) > 0:
        # Ensure tensor
        if not torch.is_tensor(triplets_to_exclude):
            raise ValueError("triplets_to_exclude must be a Torch tensor.")
        if triplets_to_exclude.ndim != 2 or triplets_to_exclude.shape[1] != 3:
            raise ValueError("triplets_to_exclude must be Nx3.")

        # Filter out triplets that do not belong to 'concepts'
        # For membership checking, move them to CPU if needed
        # (isin doesn't exist in older PyTorch versions, so ensure version or do custom checks)
        triplets_to_exclude_cpu = triplets_to_exclude.cpu()
        mask_valid = torch.all(torch.isin(triplets_to_exclude_cpu, concepts.cpu()), dim=1)
        triplets_to_exclude_cpu = triplets_to_exclude_cpu[mask_valid]

        # Sort each row so that (b,a,c) or (c,a,b) becomes (a,b,c)
        triplets_to_exclude_cpu, _ = torch.sort(triplets_to_exclude_cpu, dim=1)

        # Encode them as single integer keys
        ex_keys_cpu = encode_triplets(triplets_to_exclude_cpu, M=M)

        # Remove duplicates and sort
        ex_keys_cpu = torch.unique(ex_keys_cpu)
        ex_keys_cpu = torch.sort(ex_keys_cpu).values

        # Move to device
        excluded_keys = ex_keys_cpu.to(device)

    # Compute the maximum number of possible triplets: C(N_concepts, 3)
    max_possible = (N_concepts * (N_concepts - 1) * (N_concepts - 2)) // 6
    # Subtract those explicitly excluded (since each is unique in excluded_keys):
    if excluded_keys is not None:
        max_possible = max_possible - excluded_keys.size(0)

    if N_triplets > max_possible:
        raise ValueError(
            f"Cannot generate {N_triplets} unique triplets. "
            f"Maximum possible after excluding is {max_possible}."
        )

    # --------------------------------------------------------------------------
    # 3) Random generation + membership check in batches
    # --------------------------------------------------------------------------
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    desired = N_triplets
    result_accum = []

    while desired > 0:
        batch_size = int(desired * batch_factor)  # oversample for efficiency

        # Randomly pick triplets: shape (batch_size, 3), each index in [0, N_concepts)
        # We want distinct columns (a != b != c).
        # One approach is to sample each row with randperm for small N_concepts,
        # but that can be slow if N_concepts is large. Here, we do simpler random draws
        # and filter out duplicates.
        candidates_idx = torch.randint(
            low=0,
            high=N_concepts,
            size=(batch_size, 3),
            generator=generator,
            device=device
        )

        # Keep only rows with distinct indices
        row_mask = (
            (candidates_idx[:, 0] != candidates_idx[:, 1]) &
            (candidates_idx[:, 1] != candidates_idx[:, 2]) &
            (candidates_idx[:, 0] != candidates_idx[:, 2])
        )
        candidates_idx = candidates_idx[row_mask]

        # Sort each row so that the triplet is (a <= b <= c)
        candidates_idx, _ = torch.sort(candidates_idx, dim=1)

        # Map local indices to actual concept IDs
        candidates = concepts[candidates_idx]  # shape (X, 3)

        # Remove duplicates among these newly generated triplets
        candidates = torch.unique(candidates, dim=0)

        # If we have no excluded_keys, we can skip search
        if excluded_keys is not None and excluded_keys.size(0) > 0 and candidates.size(0) > 0:
            # Encode the candidate triplets
            cand_keys = encode_triplets(candidates, M=M)

            # Use searchsorted to find membership
            # idxs[i] = insertion position of cand_keys[i] in excluded_keys
            #   => if cand_keys[i] is actually in excluded_keys, then 
            #      excluded_keys[idxs[i]] == cand_keys[i] (assuming 0 <= idxs[i] < len(excluded_keys))
            idxs = torch.searchsorted(excluded_keys, cand_keys, right=False)  
            # Now check actual membership
            inside_bounds = (idxs >= 0) & (idxs < excluded_keys.size(0))
            # For those inside bounds, compare excluded_keys[idx] with cand_keys
            is_excluded = torch.zeros_like(inside_bounds, dtype=torch.bool)
            valid_positions = inside_bounds.nonzero().flatten()
            if valid_positions.numel() > 0:
                is_excluded[valid_positions] = (excluded_keys[idxs[valid_positions]] == cand_keys[valid_positions])
            # Keep the rows that are NOT excluded
            keep_mask = ~is_excluded
            candidates = candidates[keep_mask]

        # Now add to the accumulation
        num_avail = candidates.size(0)
        if num_avail == 0:
            continue

        needed = min(desired, num_avail)
        needed = int(needed)  # ensure Python int for slicing
        result_accum.extend(candidates[:needed].cpu())  # store them on CPU in a list
        desired -= needed

    # --------------------------------------------------------------------------
    # 4) Build final output tensor
    # --------------------------------------------------------------------------
    out_triplets = torch.stack(result_accum, dim=0).to(device)  # shape (N_triplets, 3)
    return out_triplets


def triplets_partitioning(
    triplets:Union[torch.Tensor,np.ndarray], #triplets to partition
    train_objects:Union[torch.Tensor,np.ndarray], #objects in the train set
    val_objects:Union[torch.Tensor,np.ndarray,None], #objects in the validation set
    odd_one_out_index:Union[torch.Tensor,np.ndarray,None] = None, #optional odd one out index
    ) -> Union[
        Tuple[Dict[str, List[int]],Dict[str, List[int]]],
        Tuple[Dict[str, List[int]],Dict[str, List[int]],Dict[str, List[int]]]]: 

    """
    Adjusted version of the triplets partitioning function.
    Partition triplets into three disjoint object sets for training, validation and testing.
    taken from the original implementation of the triplets partitioning function
    presented in the paper: "Human alignment of neural network representations"
    Implementation taken from:
    https://github.com/LukasMut/human_alignment/blob/main/utils/probing/helpers.py#L20
    and adjusted to work to fit our needs.
    Reference:
        @inproceedings{muttenthaler2023,
            author    = {Muttenthaler, Lukas and Dippel, Jonas and Linhardt, Lorenz and Vandermeulen, Robert A and Kornblith, Simon},
            title     = {Human alignment of neural network representations},
            booktitle = {11th International Conference on Learning Representations (ICLR)},
            year      = {2023},
            publisher = {OpenReview.net}
        }
    
    args:
    triplets:Union[torch.Tensor,np.ndarray], #triplets to partition
    odd_one_out_index:Union[torch.Tensor,np.ndarray,None], #optional odd one out index
    train_objects:Union[torch.Tensor,np.ndarray], #objects in the train set
    val_objects:Union[torch.Tensor,np.ndarray] #objects in the validation set

    returns either:    
    - triplets_indexes_partitioning: Dict[str, List[int]], #triplets indexes partitioning
    - triplet_partitioning: Dict[str, List[int]], #triplets partitioning
    or:
    - triplets_indexes_partitioning: Dict[str, List[int]], #triplets indexes partitioning
    - triplet_partitioning: Dict[str, List[int]], #triplets partitioning
    - response_partitioning: Dict[str, List[int]], #odd one out index partitioning
    """
    if odd_one_out_index is not None:
        assert len(odd_one_out_index) == len(triplets), "odd_one_out_index must have the same length as triplets"
    triplet_partitioning = defaultdict(list)
    response_partitioning = defaultdict(list) 
    triplets_indexes_partitioning = defaultdict(list)
    for triplet_idx, triplet in enumerate(triplets):
        if val_objects is not None:
            splits = (list(
                map(lambda obj: "train" if obj in train_objects else "val" if
                    obj in val_objects else "test", triplet)
            ))
        else:
            splits = (list(
                map(lambda obj: "train" if obj in train_objects else "test", triplet)
            ))
        
        if len(set(splits)) == 1:
            triplet_partitioning[splits.pop()].append(triplet.tolist())
            triplets_indexes_partitioning[splits.pop()].append(triplet_idx)
            if odd_one_out_index is not None:
                response_partitioning[splits.pop()].append(odd_one_out_index[triplet_idx])
            
    if odd_one_out_index is not None:
        return triplets_indexes_partitioning,triplet_partitioning,response_partitioning
    else:
        return triplets_indexes_partitioning,triplet_partitioning

def create_partitioned_datasets_THINGS(
    full_triplet_set:Union[torch.Tensor,np.ndarray],
    full_odd_one_out_index:Union[torch.Tensor,np.ndarray],
    K_folds:int = 3,
    N_concepts:int = 1854,
    random_seed:int = 42,
    with_validation:bool = True,
    save_path:Optional[str] = None
    ) -> List[Tuple[OddOneOutDataset,OddOneOutDataset,OddOneOutDataset]]:
    # Create the k-fold splitter
    kf = KFold(n_splits=K_folds,shuffle=True,random_state=random_seed)
    concepts_range = np.arange(N_concepts)
    if save_path is not None:
        # Create the directory if it does not exist
        os.makedirs(save_path,exist_ok=True)
        
    # Split the concepts into K folds
    concepts_folds = kf.split(concepts_range)
    folds_datasets_list = []
    for fold_idx, (train_idx,_) in enumerate(concepts_folds,start=1):
        train_val_objects = concepts_range[train_idx]
        if with_validation:
            # Split 20% of train_val_objects for validation
            concepts_train_index,concepts_val_index = train_test_split(train_val_objects,test_size=0.2,random_state=random_seed)
        else:
            concepts_train_index = train_val_objects
            concepts_val_index = None
        # Create the triplet partitioning
        triplets_indexes_partitioning,_ = triplets_partitioning(
            triplets=full_triplet_set,
            train_objects=concepts_train_index,
            val_objects=concepts_val_index)
        # Create a path for saving the partitioning indices
        # Save the partitioning indices both as a pickle file and as a torch tensor
        if save_path is not None: 
            save_path_folds = os.path.join(save_path,f'fold_{fold_idx}_partitioning_indexes.pkl')
            with open(save_path_folds,'wb') as f:
                pickle.dump(triplets_indexes_partitioning,f)
            train_tensor = torch.tensor(triplets_indexes_partitioning["train"])
            test_tensor = torch.tensor(triplets_indexes_partitioning["test"])
            torch.save(train_tensor,os.path.join(save_path,f'fold_{fold_idx}_train_partitioning_indexes.pt'))
            torch.save(test_tensor,os.path.join(save_path,f'fold_{fold_idx}_test_partitioning_indexes.pt'))
            # Create the train and test sets
            train_set = OddOneOutDataset(
            triplets= full_triplet_set[triplets_indexes_partitioning["train"]],
            choice_position=full_odd_one_out_index[triplets_indexes_partitioning["train"]])

            test_set = OddOneOutDataset(
            triplets= full_triplet_set[triplets_indexes_partitioning["test"]],
            choice_position=full_odd_one_out_index[triplets_indexes_partitioning["test"]])
    
            if with_validation:  # When a validation set is required
                val_tensor = torch.tensor(triplets_indexes_partitioning["val"])
                torch.save(val_tensor,os.path.join(save_path,f'fold_{fold_idx}_val_partitioning_indexes.pt'))
                val_set = OddOneOutDataset(
                    triplets= full_triplet_set[triplets_indexes_partitioning["val"]],
                    choice_position=full_odd_one_out_index[triplets_indexes_partitioning["val"]])
                folds_datasets_list.append((train_set,val_set,test_set))
            else:
                folds_datasets_list.append((train_set,test_set))
    return folds_datasets_list

def load_partitioning_indexes_to_datasets(
    folds_indexes_paths_list:List[str],
    full_triplet_set:Union[torch.Tensor,np.ndarray],
    full_odd_one_out_index:Union[torch.Tensor,np.ndarray],
    with_validation:bool = True,
    ) -> List[Tuple[OddOneOutDataset,OddOneOutDataset,OddOneOutDataset]]:
    folds_datasets_list = []
    for fold_indexes_path in folds_indexes_paths_list:
        with open(fold_indexes_path,'rb') as f:
            partitioning_indexes = pickle.load(f)
        train_set = OddOneOutDataset(
            triplets= full_triplet_set[partitioning_indexes["train"]],
            choice_position=full_odd_one_out_index[partitioning_indexes["train"]])
        test_set = OddOneOutDataset(
            triplets= full_triplet_set[partitioning_indexes["test"]],
            choice_position=full_odd_one_out_index[partitioning_indexes["test"]])
        if with_validation:
            val_set = OddOneOutDataset(
                triplets= full_triplet_set[partitioning_indexes["val"]],
                choice_position=full_odd_one_out_index[partitioning_indexes["val"]])
            folds_datasets_list.append((train_set,val_set,test_set))
        else:
            folds_datasets_list.append((train_set,test_set))
    return folds_datasets_list

def calculate_NLL(pairs_dist, y):
    """
    Calculate the negative log-likelihood (NLL) as the sum of the log of the softmax of the pairs_dist
    
    Args:
        pairs_dist (torch.Tensor): Distribution over pairs, shape (batch_size, n_choices)
        y (torch.Tensor): Target values, either as indices (1D) or one-hot encoded (2D)
        
    Returns:
        torch.Tensor: Negative log-likelihood value
    """ 
    if len(y.shape) == 1: # for the case y is represent the index of the correct choice lets use the one hot encoding
        y_one_hot = torch.zeros_like(pairs_dist)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        #calculate the NLL as the sum of the log of the softmax of the pairs_dist
        return -torch.sum(torch.log_softmax(pairs_dist, dim=1)*y_one_hot)
    else:
        #calculate the NLL as the sum of the log of the softmax of the pairs_dist
        return -torch.sum(torch.log_softmax(pairs_dist, dim=1)*y)
    
def is_diagonal(C, tol=1e-6):
    """
    Checks if a given square tensor C is diagonal by ensuring
    all off-diagonal elements are (close to) zero.

    Args:
        C (torch.Tensor): The matrix to check.
        tol (float): Tolerance for floating-point comparisons.

    Returns:
        bool: True if C is diagonal, False otherwise.
    """
    with torch.no_grad():
        if C.shape[0] != C.shape[1]:  # Ensure it's square
            return False

        off_diag_mask = ~torch.eye(C.shape[0], dtype=torch.bool, device=C.device)
        return torch.all(C[off_diag_mask].abs() < tol).item()

def generate_unique_triplets_subset_of_concepts_new(
    concepts: torch.Tensor,
    N_triplets: int,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_factor: int = 5
) -> torch.Tensor:
    """
    Generate N_triplets unique triplets from a subset of concepts, on GPU for speed.
    Each triplet (a,b,c) is returned in ascending order (a <= b <= c) and is guaranteed
    not to appear in `triplets_to_exclude` (if provided).

    This function encodes triplets into a single integer key for fast membership checks
    via torch.searchsorted, rather than row-by-row set membership in Python.

    Args:
        concepts (torch.Tensor):
            1D tensor of concept indices from which triplets are sampled. 
            For best performance, pass it on GPU already if possible.
        N_triplets (int):
            Number of unique triplets to generate.
        triplets_to_exclude (torch.Tensor, optional):
            Nx3 tensor of triplets to exclude. Each row is sorted or unsorted (we will sort).
        seed (int, optional):
            Seed for the random number generator (GPU if device='cuda').
        device (str, optional):
            GPU or CPU device (e.g., "cuda" or "cpu"). If None, defaults to `concepts.device`.
        batch_factor (int):
            Factor controlling batch size. 
            If we want N_triplets, each batch samples `N_triplets * batch_factor` rows.

    Returns:
        torch.Tensor:
            (N_triplets, 3) of unique triplets in ascending order, on `device`.
    """

    # --------------------------------------------------------------------------
    # 0) Helper: Encoding and decoding
    # --------------------------------------------------------------------------
    def encode_triplets(trips: torch.Tensor, M: int) -> torch.Tensor:
        # trips: shape (N, 3), sorted rowwise
        # key = a*M^2 + b*M + c
        return trips[:, 0] * (M**2) + trips[:, 1] * M + trips[:, 2]
    
    # --------------------------------------------------------------------------
    # 1) Basic checks, device setup
    # --------------------------------------------------------------------------
    if not torch.is_tensor(concepts):
        concepts = torch.tensor(concepts, dtype=torch.long)
    concepts = torch.unique(concepts)  # ensure uniqueness of concept IDs

    N_concepts = concepts.size(0)
    if N_concepts < 3:
        raise ValueError("Cannot generate triplets. Need at least 3 distinct concepts.")

    # Convert N_triplets to an int if it's not already
    N_triplets = int(N_triplets)

    # Decide on device
    if device is None:
        device = concepts.device
    concepts = concepts.to(device)

    # For encoding to be collision-free:
    # We'll set M = max(concepts)+1
    # Check for potential overflow if concept IDs are very large
    M = torch.max(concepts).item() + 1

    # --------------------------------------------------------------------------
    # 2) Build the excluded_keys from triplets_to_exclude
    # --------------------------------------------------------------------------
    excluded_keys = None
    if triplets_to_exclude is not None and len(triplets_to_exclude) > 0:
        # Ensure tensor
        if not torch.is_tensor(triplets_to_exclude):
            raise ValueError("triplets_to_exclude must be a Torch tensor.")
        if triplets_to_exclude.ndim != 2 or triplets_to_exclude.shape[1] != 3:
            raise ValueError("triplets_to_exclude must be Nx3.")

        # Filter out triplets that do not belong to 'concepts'
        # For membership checking, move them to CPU if needed
        # (isin doesn't exist in older PyTorch versions, so ensure version or do custom checks)
        triplets_to_exclude_cpu = triplets_to_exclude.cpu()
        mask_valid = torch.all(torch.isin(triplets_to_exclude_cpu, concepts.cpu()), dim=1)
        triplets_to_exclude_cpu = triplets_to_exclude_cpu[mask_valid]

        # Sort each row so that (b,a,c) or (c,a,b) becomes (a,b,c)
        triplets_to_exclude_cpu, _ = torch.sort(triplets_to_exclude_cpu, dim=1)

        # Encode them as single integer keys
        ex_keys_cpu = encode_triplets(triplets_to_exclude_cpu, M=M)

        # Remove duplicates and sort
        ex_keys_cpu = torch.unique(ex_keys_cpu)
        ex_keys_cpu = torch.sort(ex_keys_cpu).values

        # Move to device
        excluded_keys = ex_keys_cpu.to(device)

    # Compute the maximum number of possible triplets: C(N_concepts, 3)
    max_possible = (N_concepts * (N_concepts - 1) * (N_concepts - 2)) // 6
    # Subtract those explicitly excluded (since each is unique in excluded_keys):
    if excluded_keys is not None:
        max_possible = max_possible - excluded_keys.size(0)

    if N_triplets > max_possible:
        raise ValueError(
            f"Cannot generate {N_triplets} unique triplets. "
            f"Maximum possible after excluding is {max_possible}."
        )

    # --------------------------------------------------------------------------
    # 3) Random generation + membership check in batches
    # --------------------------------------------------------------------------
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    desired = N_triplets
    result_accum = []

    while desired > 0:
        batch_size = int(desired * batch_factor)  # oversample for efficiency

        # Randomly pick triplets: shape (batch_size, 3), each index in [0, N_concepts)
        # We want distinct columns (a != b != c).
        # One approach is to sample each row with randperm for small N_concepts,
        # but that can be slow if N_concepts is large. Here, we do simpler random draws
        # and filter out duplicates.
        candidates_idx = torch.randint(
            low=0,
            high=N_concepts,
            size=(batch_size, 3),
            generator=generator,
            device=device
        )

        # Keep only rows with distinct indices
        row_mask = (
            (candidates_idx[:, 0] != candidates_idx[:, 1]) &
            (candidates_idx[:, 1] != candidates_idx[:, 2]) &
            (candidates_idx[:, 0] != candidates_idx[:, 2])
        )
        candidates_idx = candidates_idx[row_mask]

        # Sort each row so that the triplet is (a <= b <= c)
        candidates_idx, _ = torch.sort(candidates_idx, dim=1)

        # Map local indices to actual concept IDs
        candidates = concepts[candidates_idx]  # shape (X, 3)

        # Remove duplicates among these newly generated triplets
        candidates = torch.unique(candidates, dim=0)

        # If we have no excluded_keys, we can skip search
        if excluded_keys is not None and excluded_keys.size(0) > 0 and candidates.size(0) > 0:
            # Encode the candidate triplets
            cand_keys = encode_triplets(candidates, M=M)

            # Use searchsorted to find membership
            # idxs[i] = insertion position of cand_keys[i] in excluded_keys
            #   => if cand_keys[i] is actually in excluded_keys, then 
            #      excluded_keys[idxs[i]] == cand_keys[i] (assuming 0 <= idxs[i] < len(excluded_keys))
            idxs = torch.searchsorted(excluded_keys, cand_keys, right=False)  
            # Now check actual membership
            inside_bounds = (idxs >= 0) & (idxs < excluded_keys.size(0))
            # For those inside bounds, compare excluded_keys[idx] with cand_keys
            is_excluded = torch.zeros_like(inside_bounds, dtype=torch.bool)
            valid_positions = inside_bounds.nonzero().flatten()
            if valid_positions.numel() > 0:
                is_excluded[valid_positions] = (excluded_keys[idxs[valid_positions]] == cand_keys[valid_positions])
            # Keep the rows that are NOT excluded
            keep_mask = ~is_excluded
            candidates = candidates[keep_mask]

        # Now add to the accumulation
        num_avail = candidates.size(0)
        if num_avail == 0:
            continue

        needed = min(desired, num_avail)
        needed = int(needed)  # ensure Python int for slicing
        result_accum.extend(candidates[:needed].cpu())  # store them on CPU in a list
        desired -= needed

    # --------------------------------------------------------------------------
    # 4) Build final output tensor
    # --------------------------------------------------------------------------
    out_triplets = torch.stack(result_accum, dim=0).to(device)  # shape (N_triplets, 3)
    return out_triplets


def generate_unique_triplets_subset_of_concepts_very_new(
    concepts: torch.Tensor,
    N_triplets: int,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_factor: int = 5
) -> torch.Tensor:
    """
    Generate N_triplets unique triplets from a subset of concepts using Torch.
    Each triplet (a, b, c) is sorted in ascending order and does not appear
    in triplets_to_exclude. Global uniqueness is ensured across batches.
    """
    def encode_triplets(trips: torch.Tensor, M: int) -> torch.Tensor:
        # Encode a sorted triplet as a single integer.
        return trips[:, 0] * (M ** 2) + trips[:, 1] * M + trips[:, 2]
    
    if not torch.is_tensor(concepts):
        concepts = torch.tensor(concepts, dtype=torch.long)
    concepts = torch.unique(concepts)
    N_concepts = concepts.size(0)
    if N_concepts < 3:
        raise ValueError("Need at least 3 distinct concepts.")
    N_triplets = int(N_triplets)
    if device is None:
        device = concepts.device
    concepts = concepts.to(device)
    M = int(torch.max(concepts).item()) + 1

    # Process excluded triplets.
    excluded_keys = None
    if triplets_to_exclude is not None and len(triplets_to_exclude) > 0:
        if not torch.is_tensor(triplets_to_exclude):
            raise ValueError("triplets_to_exclude must be a Torch tensor.")
        if triplets_to_exclude.ndim != 2 or triplets_to_exclude.shape[1] != 3:
            raise ValueError("triplets_to_exclude must be Nx3.")
        triplets_to_exclude_cpu = triplets_to_exclude.cpu()
        mask_valid = torch.all(torch.isin(triplets_to_exclude_cpu, concepts.cpu()), dim=1)
        triplets_to_exclude_cpu = triplets_to_exclude_cpu[mask_valid]
        triplets_to_exclude_cpu, _ = torch.sort(triplets_to_exclude_cpu, dim=1)
        ex_keys_cpu = encode_triplets(triplets_to_exclude_cpu, M)
        ex_keys_cpu = torch.unique(ex_keys_cpu)
        ex_keys_cpu = torch.sort(ex_keys_cpu).values
        excluded_keys = ex_keys_cpu.to(device)
    
    max_possible = (N_concepts * (N_concepts - 1) * (N_concepts - 2)) // 6
    if excluded_keys is not None:
        max_possible -= excluded_keys.numel()
    if N_triplets > max_possible:
        raise ValueError(f"Cannot generate {N_triplets} triplets. Maximum available is {max_possible}.")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Global containers for accepted triplets.
    accepted_keys = torch.empty((0,), dtype=torch.long, device=device)
    accepted_triplets = []
    accepted_count = 0
    desired = N_triplets

    while accepted_count < N_triplets:
        batch_size = int(desired * batch_factor)
        candidates_idx = torch.randint(low=0, high=N_concepts, size=(batch_size, 3),
                                       generator=generator, device=device)
        # Keep rows with all distinct indices.
        row_mask = ((candidates_idx[:, 0] != candidates_idx[:, 1]) &
                    (candidates_idx[:, 1] != candidates_idx[:, 2]) &
                    (candidates_idx[:, 0] != candidates_idx[:, 2]))
        candidates_idx = candidates_idx[row_mask]
        # Sort each row so that each triplet is in ascending order.
        candidates_idx, _ = torch.sort(candidates_idx, dim=1)
        candidates = concepts[candidates_idx]
        candidates = torch.unique(candidates, dim=0)
        if (excluded_keys is not None) and (excluded_keys.numel() > 0) and (candidates.numel() > 0):
            cand_keys = encode_triplets(candidates, M)
            idxs = torch.searchsorted(excluded_keys, cand_keys, right=False)
            inside_bounds = (idxs >= 0) & (idxs < excluded_keys.numel())
            is_excluded = torch.zeros_like(inside_bounds, dtype=torch.bool)
            valid_positions = inside_bounds.nonzero().flatten()
            if valid_positions.numel() > 0:
                is_excluded[valid_positions] = (excluded_keys[idxs[valid_positions]] == cand_keys[valid_positions])
            keep_mask = ~is_excluded
            candidates = candidates[keep_mask]
            cand_keys = encode_triplets(candidates, M)
        else:
            cand_keys = encode_triplets(candidates, M) if candidates.numel() > 0 else torch.tensor([], dtype=torch.long, device=device)
        # Remove candidates already accepted.
        if accepted_keys.numel() > 0 and candidates.numel() > 0:
            mask = ~torch.isin(cand_keys, accepted_keys)
            candidates = candidates[mask]
            cand_keys = cand_keys[mask]
        if candidates.numel() == 0:
            continue
        num_avail = candidates.size(0)
        take = min(desired, num_avail)
        accepted_triplets.append(candidates[:take].cpu())
        accepted_keys = torch.cat([accepted_keys, cand_keys[:take]])
        accepted_count += take
        desired = N_triplets - accepted_count
    final_result = torch.cat(accepted_triplets, dim=0)[:N_triplets]
    return final_result

def run_sanity_checks_unique_triplets_subset_of_concepts(result, concepts, triplets_to_exclude, expected_count):
    """
    Run sanity checks on the generated triplets.
    Checks:
      0. Every element in the result is from the provided 'concepts' set.
      1. Result shape is (expected_count, 3).
      2. Each triplet is in ascending order.
      3. There are no duplicate triplets.
      4. None of the generated triplets appear in the exclusion set.
    Returns True if all tests pass; otherwise prints errors and returns False.
    """
    # If torch tensor, convert to numpy for each.
    if torch.is_tensor(result):
        result = result.cpu().numpy()
    if torch.is_tensor(concepts):
        concepts = concepts.cpu().numpy()
    if torch.is_tensor(triplets_to_exclude):
        triplets_to_exclude = triplets_to_exclude.cpu().numpy()
    
    # 0. Membership check: ensure every element in result is from 'concepts'.
    if not np.all(np.isin(result, concepts)):
        print("Membership check failed: some triplet elements are not in the provided concepts.")
        return False

    # 1. Shape.
    if result.shape[0] != expected_count or result.shape[1] != 3:
        print("Shape check failed.")
        return False

    # 2. Ascending order.
    if not (np.all(result[:, 0] <= result[:, 1]) and np.all(result[:, 1] <= result[:, 2])):
        print("Ascending order check failed.")
        return False

    # 3. Uniqueness.
    M = int(np.max(concepts)) + 1
    M2 = M * M
    encoded = result[:, 0] * M2 + result[:, 1] * M + result[:, 2]
    if np.unique(encoded).size != expected_count:
        print("Uniqueness check failed.")
        return False

    # 4. Exclusion.
    def encode_triplets(trips, M, M2):
        return trips[:, 0] * M2 + trips[:, 1] * M + trips[:, 2]
    triplets_to_exclude = np.asarray(triplets_to_exclude, dtype=np.int64)
    if triplets_to_exclude.ndim != 2 or triplets_to_exclude.shape[1] != 3:
        print("Triplets_to_exclude shape error.")
        return False
    triplets_to_exclude = np.sort(triplets_to_exclude, axis=1)
    ex_keys = np.sort(np.unique(encode_triplets(triplets_to_exclude, M, M2)))
    common = np.intersect1d(encoded, ex_keys)
    if common.size != 0:
        print("Exclusion check failed.")
        return False

    return True

def freeze_params(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in a model by setting requires_grad to False.
    
    Args:
        model (nn.Module): The model whose parameters should be frozen
        
    Returns:
        nn.Module: The model with frozen parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    return model



# # Example usage:
# if __name__ == "__main__":
#     # Suppose you have 1,854 concepts:
#     concepts = np.arange(1854)
    
#     # And 4.2 million triplets to exclude (for example, generated arbitrarily):
#     # Here, we simulate by randomly generating 4.2e6 triplets.
#     rng = np.random.default_rng(123)
#     excl_indices = rng.integers(0, 1854, size=(4200000, 3))
#     excl_triplets = np.sort(excl_indices, axis=1)
    
#     # Generate 4.2 million valid unique triplets.
#     triplets = generate_unique_triplets_subset_of_concepts_numpy(
#         concepts=concepts,
#         N_triplets=4200000,
#         triplets_to_exclude=excl_triplets,
#         seed=456,
#         batch_factor=5
#     )
    
#     print("Generated triplets shape:", triplets.shape)


    
# if __name__ == '__main__':
    
    
# #triplet partitioning sanity check
#     concepts = np.arange(20)
#     train_objects = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
#     val_objects = np.array([12,13,14,15])
#     test_objects = np.array([16,17,18,19])
#     triplets = np.array([
#         [0,1,2], #train (all in train_objects 0-11)
#         [4,5,6], #train (all in train_objects 0-11)
#         [8,9,10], #train (all in train_objects 0-11)
#         [12,13,14], #val (all in val_objects 12-15)
#         [13,14,15], #val (all in val_objects 12-15)
#         [16,17,18], #test (all in test_objects 16-19)
#         [17,18,19], #test (all in test_objects 16-19)
#         [1,12,3], #dump (mixed train/val)
#         [15,5,18], #dump (mixed val/test)
#         [11,14,17], #dump (mixed train/val/test)
#         [2,3,4], #train (all in train_objects 0-11)
#         [5,6,7], #train (all in train_objects 0-11)
#         [12,14,15], #val (all in val_objects 12-15)
#         [16,18,19], #test (all in test_objects 16-19)
#         [0,15,7], #dump (mixed train/val)
#         [13,4,19], #dump (mixed train/val/test)
#         [8,10,11], #train (all in train_objects 0-11)
#         [12,13,15], #val (all in val_objects 12-15)
#         [16,17,19], #test (all in test_objects 16-19)
#         [9,14,18] #dump (mixed train/val/test)
#         ])
    
#     # odd_one_out_index = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
#     triplets_index_partition,triplet_partitioning = triplets_partitioning(triplets=triplets,odd_one_out_index=None,train_objects=train_objects,val_objects=val_objects)
#     # Expected outcomes based on the triplets and answers
#     expected_triplet_partitioning = {
#         'train': [[0,1,2], [4,5,6], [8,9,10], [2,3,4], [5,6,7], [8,10,11]],
#         'val': [[12,13,14], [13,14,15], [12,14,15], [12,13,15]],
#         'test': [[16,17,18], [17,18,19], [16,18,19], [16,17,19]]
#     }

#     expected_response_partitioning = {
#         'train': [0, 1, 2, 1, 2, 1],
#         'val': [0, 1, 0, 2],
#         'test': [2, 0, 1, 0]
#     }

#     # Verify the partitioning matches expected outcomes
#     assert triplet_partitioning == expected_triplet_partitioning, "Triplet partitioning does not match expected"
#     # assert response_partitioning == expected_response_partitioning, "Response partitioning does not match expected"
    
#     print(triplet_partitioning)
#     # print(response_partitioning)

    # Sanity check for generate_unique_triplets_subset_of_concepts
    
    # # Test case 1: Basic functionality
    # concepts = torch.tensor([0, 1, 2, 3, 4])
    # n_triplets = 3
    # triplets = generate_unique_triplets_subset_of_concepts_new(concepts, n_triplets, seed=42)
    # print("Test 1 - Generated triplets:", triplets)
    # assert len(triplets) == n_triplets, "Wrong number of triplets generated"
    # assert torch.all(torch.isin(triplets, concepts.to(triplets.device))), "Generated triplets contain invalid concepts"
    
    # # Test case 2: With exclusions
    # triplets_to_exclude = torch.tensor([[0, 1, 2], [1, 2, 3],[4,5,6]])
    # triplets = generate_unique_triplets_subset_of_concepts_new(
    #     concepts, 
    #     N_triplets=2, 
    #     triplets_to_exclude=triplets_to_exclude,
    #     seed=42
    # )
    # print("\nTest 2 - Generated triplets with exclusions:", triplets)
    # # Check that excluded triplets are not in generated triplets
    # for excluded in triplets_to_exclude.to(triplets.device):
    #     assert not torch.any(torch.all(triplets == excluded, dim=1)), "Generated triplets contain excluded triplet"
    
    # # Test case 3: Edge case - maximum possible triplets
    # small_concepts = torch.tensor([0, 1, 2, 3])
    # max_triplets = (len(small_concepts) * (len(small_concepts)-1) * (len(small_concepts)-2)) // 6
    # triplets = generate_unique_triplets_subset_of_concepts_new(small_concepts, max_triplets, seed=42)
    # print(f"\nTest 3 - Generated all possible {max_triplets} triplets:", triplets)
    # assert len(triplets) == max_triplets, "Failed to generate maximum possible triplets"
    
    # print("\nAll sanity checks passed!")
    
    
def autobatch(inputs, func, batch_size=16):
    """
    Autobatch a function over inputs,
    dynamically building the container tensor

    Args:
        inputs: torch.Tensor or other iterable - inputs to the function
        func: callable - function to apply to each batch
        batch_size: int - batch size
        
    Returns:
        torch.Tensor: outputs from the function
        
    Raises:
        AssertionError: If output sizes don't match expected batch size
    """

    outputs = None

    def batch_indices(n, batch_size):
        """
        Generate slices of batch indices
        
        Args:
            n (int): Total number of items
            batch_size (int): Size of each batch
            
        Yields:
            slice: Indices for the current batch
        """
        for i in range(0, n, batch_size):
            yield slice(i, min(n, i + batch_size))

    n_items = len(inputs)
    for i_batch, batch_slice in enumerate(batch_indices(n_items, batch_size)):
        cur_outputs = func(inputs[batch_slice])
        output_size = cur_outputs.size()[1:]

        if outputs is None:
            outputs = torch.full((n_items,) + output_size, fill_value=float('nan'), dtype=cur_outputs.dtype, device=cur_outputs.device)
        
        assert outputs[batch_slice].size() == cur_outputs.size(), "Batch output size mismatch"
        outputs[batch_slice] = cur_outputs

    return outputs

def test_autobatch():
    """
    Test function for the autobatch utility to verify it works correctly.
    
    Raises:
        AssertionError: If the autobatched result doesn't match the full computation
    """
    func = lambda x: (x ** 2).sum(dim=-1)
    inputs = torch.randn(100, 10, 20)
    outputs = autobatch(inputs, func, batch_size=16)
    assert torch.allclose(outputs, (inputs ** 2).sum(dim=-1)), "Autobatch failed"
    
if __name__ == "__main__":

    test_autobatch()
