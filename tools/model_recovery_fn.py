#python libraries
from typing import List,Union,Optional
#add file folder to path
# Add parent directory to path to allow imports
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
work_dir = str(Path(__file__).parent)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if work_dir not in sys.path:
    sys.path.append(work_dir)
import time
#external libraries
import torch
torch.set_float32_matmul_precision('high')
from tqdm import tqdm
#my libraries
from Cog_model import CogModel
from triplets_data_set import TripletsDataLoader,OddOneOutDataset
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from utils import is_diagonal,generate_unique_triplets_subset_of_concepts_very_new
from model_io import load_cog_model


def create_model_recovery_datasets_simplified(
    data_generating_model:CogModel,
    features_folder:Union[str,Path],
    N_concepts: int,
    N_train_triplets:int,
    seed : int,
    K_folds:int = 5,
    triplets_to_exclude:Optional[torch.Tensor] = None,
    N_participants:int = 100,
    batch_size:int = 100,
):
    """
    Create datasets for model recovery experiment using cross-validation across disjoint concept subsets.
    
    This function creates K disjoint subsets of concepts, generates triplets for each subset,
    simulates human responses, and creates K cross-validation folds where each fold uses
    different concept subsets for training, validation, and testing.
    
    Process:
    1. Divide N_concepts into K disjoint subsets
    2. Calculate number of triplets per subset (N_train_triplets // K_folds)
    3. Generate triplets for each concept subset using only concepts from that subset
    4. Simulate human responses for all triplets
    5. Create K folds via cross-validation:
       - 1 subset → test set
       - 1 subset → validation set  
       - Remaining (K-2) subsets → training set
    6. Rotate subset assignments across folds
    
    Parameters:
        data_generating_model (CogModel): The model used to generate simulated responses
        features_folder (Union[str,Path]): Path to the folder containing model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets (int): Total number of triplets to generate (divided across K subsets)
        seed (int): Random seed for reproducibility
        K_folds (int, optional): Number of disjoint concept subsets and CV folds. Defaults to 5.
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation. 
                                                               Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 100.
        batch_size (int, optional): Batch size for processing. Defaults to 100.
    
    Returns:
        List[Tuple[OddOneOutDataset, OddOneOutDataset, OddOneOutDataset]]: A list of tuples, 
        where each tuple contains (train_set, val_set, test_set) for a fold
        
    Notes:
        - The function handles model name standardization (e.g., Google_ViT_Large_224 → Google_ViT_Large)
        - Each concept subset is completely disjoint (no overlap between subsets)
        - Each subset generates N_train_triplets // K_folds triplets
        - Cross-validation ensures each concept subset serves as test, validation, and training data
        - The data-generating model is used to simulate human responses for all generated triplets
    """
    ###### Odd-One-Out Data Generation ######
    # Load the features
    if data_generating_model.model_name == 'Google_ViT_Large_224':
        data_generating_model_name = 'Google_ViT_Large'
    else:
        data_generating_model_name = data_generating_model.model_name
    if "PCA_500" in data_generating_model_name:
        data_generating_model_name = data_generating_model_name.replace("_PCA_500","")
    model_features_path = features_folder + "/" + f"{data_generating_model_name}.pt"
    model_features = torch.load(model_features_path)
    
    # Step 1: Create K disjoint subsets of concepts
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=seed)
    concepts_range = np.arange(N_concepts)
    concepts_folds = list(kf.split(concepts_range))
    
    # Step 2: Calculate number of triplets per subset
    N_triplets_per_fold = N_train_triplets // (K_folds-1)
    
    # Step 3: Generate triplets for each concept subset
    folds_triplets_list = []
    for fold_idx, (_, concepts_subset_indices) in enumerate(concepts_folds):
        triplets = generate_unique_triplets_subset_of_concepts_very_new(
            concepts_subset_indices, N_triplets_per_fold, 
            triplets_to_exclude=triplets_to_exclude, seed=seed + fold_idx
        )
        folds_triplets_list.append(triplets)
    
    # Step 4: Generate responses for each triplet set
    folds_responses_list = []
    for fold_idx, triplets in enumerate(folds_triplets_list):
        triplets, choice_position = data_generating_model.simulate_human_response(
            triplets, model_features, N_participants=N_participants, batch_size=batch_size
        )
        folds_responses_list.append(choice_position)
        # Update the triplets list with the processed triplets
        folds_triplets_list[fold_idx] = triplets
    
    # Step 5-6: Create cross-validation folds
    folds_datasets_list = []
    for fold_idx in range(K_folds):
        # Determine subset assignments for this fold
        test_idx = fold_idx
        val_idx = (fold_idx + 1) % K_folds
        train_indices = [i for i in range(K_folds) if i != test_idx and i != val_idx]
        
        # Create test set from one subset
        test_set = OddOneOutDataset(
            triplets=folds_triplets_list[test_idx],
            choice_position=folds_responses_list[test_idx]
        )
        
        # Create validation set from one subset
        val_set = OddOneOutDataset(
            triplets=folds_triplets_list[val_idx],
            choice_position=folds_responses_list[val_idx]
        )
        
        # Create training set from remaining subsets
        train_triplets = []
        train_choices = []
        for train_idx in train_indices:
            train_triplets.append(folds_triplets_list[train_idx])
            train_choices.append(folds_responses_list[train_idx])
        
        train_set = OddOneOutDataset(
            triplets=torch.cat(train_triplets, dim=0),
            choice_position=torch.cat(train_choices, dim=0)
        )
        
        folds_datasets_list.append((train_set, val_set, test_set))
    
    return folds_datasets_list

def create_model_recovery_datasets(
    data_generating_model:CogModel,
    features_folder:Union[str,Path],
    N_concepts: int,
    N_train_triplets:int,
    seed : int,
    K_folds:int = 3,
    triplets_to_exclude:Optional[torch.Tensor] = None,
    N_participants:int = 100,
    batch_size:int = 100,
):
    """
    Create datasets for the model recovery experiment using K-fold cross-validation.
    
    This function generates train, validation, and test datasets for each fold in a K-fold
    cross-validation setup. It loads model features, partitions concepts across folds, 
    generates triplets, simulates human responses using the data-generating model, and 
    creates dataset objects for each fold.
    
    Parameters:
        data_generating_model (CogModel): The model used to generate simulated responses
        features_folder (Union[str,Path]): Path to the folder containing model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets (int): Number of triplets to generate for the training set
        seed (int): Random seed for reproducibility
        K_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation. 
                                                               Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 100.
        batch_size (int, optional): Batch size for processing. Defaults to 100.
    
    Returns:
        List[Tuple[OddOneOutDataset, OddOneOutDataset, OddOneOutDataset]]: A list of tuples, 
        where each tuple contains (train_set, val_set, test_set) for a fold
        
    Notes:
        - The function handles model name standardization (e.g., Google_ViT_Large_224 → Google_ViT_Large)
        - For each fold, concepts are divided into train+validation and test sets
        - From the train set, a small portion is further separated for validation
        - Test and validation sets are sized proportionally to the training set
        - The data-generating model is used to simulate human responses for all generated triplets
    """
    ###### Odd-One-Out Data Generation ######
    # Divide the concepts into training and test folds
    # Create the K-fold splitter
    # Load the model features
    if data_generating_model.model_name == 'Google_ViT_Large_224':
        data_generating_model_name = 'Google_ViT_Large'
    else:
        data_generating_model_name = data_generating_model.model_name
    if "PCA_500" in data_generating_model_name:
        data_generating_model_name = data_generating_model_name.replace("_PCA_500","")
    model_features_path = features_folder + "/" + f"{data_generating_model_name}.pt"
    model_features = torch.load(model_features_path)
    # Create the K-fold splitter
    kf = KFold(n_splits=K_folds,shuffle=True,random_state=seed)
    # Create the range of concept indices
    concepts_range = np.arange(N_concepts)
    # Split the concepts into K folds
    concepts_folds = kf.split(concepts_range) 
    folds_datasets_list = []
    for fold_idx, (train_val_idx,test_idx) in enumerate(concepts_folds,start=1):
        train_val_objects = concepts_range[train_val_idx]
        
        # Split 20% of the training concepts for validation
        val_size = round((K_folds/((K_folds-1)*5))*len(train_val_objects))
        concepts_train_index,concepts_val_index = train_test_split(train_val_objects,test_size=val_size,random_state=seed)
        concepts_test_index = concepts_range[test_idx]
        ## Calculate the number of triplets for the train, validation, and test sets
        N_train_triplets = N_train_triplets  # Assuming that the train set is 80% of the total number of triplets
        N_test_triplets = round(N_train_triplets/8)  # Assuming that the test set is 10% of the total number of triplets
        N_validation_triplets = round(N_train_triplets/8)  # Assuming that the validation set is 10% of the total number of triplets
        # Generate the train, validation, and test triplets 
        train_triplets= generate_unique_triplets_subset_of_concepts_very_new(concepts_train_index,N_train_triplets,triplets_to_exclude=triplets_to_exclude,seed=seed)
        test_triplets = generate_unique_triplets_subset_of_concepts_very_new(concepts_test_index,N_test_triplets,triplets_to_exclude=triplets_to_exclude,seed=seed)
        validation_triplets = generate_unique_triplets_subset_of_concepts_very_new(concepts_val_index,N_validation_triplets,triplets_to_exclude=triplets_to_exclude,seed=seed)
        # Simulate the human choices for the train, validation, and test triplets
        train_triplets,train_choice_position = data_generating_model.simulate_human_response(train_triplets,model_features,N_participants=N_participants,batch_size=batch_size)
        test_triplets,test_choice_position = data_generating_model.simulate_human_response(test_triplets,model_features,N_participants=N_participants,batch_size=batch_size)
        validation_triplets,validation_choice_position = data_generating_model.simulate_human_response(validation_triplets,model_features,N_participants=N_participants,batch_size=batch_size)
        # Create the train, validation, and test datasets
        train_set = OddOneOutDataset(
            triplets= train_triplets,
            choice_position=train_choice_position)
        val_set = OddOneOutDataset(
            triplets= validation_triplets,
            choice_position=validation_choice_position)
        test_set = OddOneOutDataset(
            triplets= test_triplets,
            choice_position=test_choice_position)
        folds_datasets_list.append((train_set,val_set,test_set))        
        
    return folds_datasets_list



def model_recovery_single_simulation_single_model(
    data_generating_model: CogModel,
    reference_model_names_list: List[str],
    data_generating_features_folder: Union[str, Path],
    reference_models_features_folder: Union[str, Path],
    N_concepts: int,
    N_train_triplets: int,
    seed: int,
    reg_func: str,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    N_participants: int = 1,
    batch_size: Union[int, str] = 'full',
    K_folds: int = 3,
    regularization_constants_list: Optional[List[float]] = None,
    constraints: Optional[str] = None,
    dim_reduction_method: Optional[str] = None,
    
):
    """
    Run a single model recovery simulation for a given data-generating model.

    This function evaluates how well each reference model can recover the structure of data
    generated by a specific data-generating model. It creates datasets for K-fold cross-validation,
    trains reference models with different regularization constants, and identifies the best
    performing configuration for each reference model.
    
    Parameters:
        data_generating_model (CogModel): The model used to generate simulated responses
        reference_model_names_list (List[str]): List of model names to evaluate as reference models
        data_generating_features_folder (Union[str, Path]): Path to the folder containing data-generating model features
        reference_models_features_folder (Union[str, Path]): Path to the folder containing reference model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets (int): Number of triplets to generate for the training set
        seed (int): Random seed for reproducibility
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation. 
                                                              Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 1.
        batch_size (Union[int, str], optional): Batch size for processing, or 'full' for full batch.
                                               Defaults to 'full'.
        K_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        regularization_constants_list (Optional[List[float]], optional): List of regularization constants
                                                                        to evaluate. Defaults to None.
        constraints (Optional[str], optional): Constraints on model parameter matrix (e.g., 'diagonal', 
                                              'zero_shot'). Defaults to None.
        dim_reduction_method (Optional[str], optional): Dimensionality reduction method to apply.
                                                       Defaults to None.
    
    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - full_results_dict: Nested dictionary mapping reference models to results for all
                                regularization constants
            - best_results_dict: Dictionary mapping reference models to results with the best
                                regularization constant (based on validation accuracy)
                                
    Notes:
        - For each reference model and regularization constant, performance is evaluated on all K folds
        - Results include test, validation, and training accuracy and negative log-likelihood (NLL)
        - The best regularization constant for each model is selected based on validation accuracy
        - Special handling is applied for 'zero_shot' constraints (pre-trained models) and 'diagonal'
          constraints (diagonal W matrix)
        - Models are cleaned up after evaluation to free GPU memory
    """

    # Create the datasets  
    datasets_list = create_model_recovery_datasets(
        data_generating_model=data_generating_model,
        features_folder=data_generating_features_folder,
        N_concepts=N_concepts,
        N_train_triplets=N_train_triplets,
        seed=seed,
        K_folds=K_folds,
        triplets_to_exclude=triplets_to_exclude,
        N_participants=N_participants,
        batch_size=batch_size,
    )
    # Create the results dictionary
    full_results_dict = {model_name: {} for model_name in reference_model_names_list}   
    best_results_dict = {}

    for reference_model_name in tqdm(reference_model_names_list, desc="Model recovery over reference models"):
        # Load the reference model features
        
        if reference_model_name == 'Google_ViT_Large_224':
            reference_model_name_load = 'Google_ViT_Large'
            reference_model_features_path = reference_models_features_folder + "/" + f"{reference_model_name_load}.pt"
        else:
            reference_model_features_path = reference_models_features_folder + "/" + f"{reference_model_name}.pt"
        
        reference_model_features = torch.load(reference_model_features_path)
        # Wrap the regularization constants loop with tqdm for additional feedback
        for regularization_constant in tqdm(regularization_constants_list, desc=f"{reference_model_name} reg constants", leave=False):
            # Create lists for the folds
            test_folds_acc_list = []
            val_folds_acc_list = []
            train_folds_acc_list = []
            test_folds_nll_list = []
            val_folds_nll_list = []
            train_folds_nll_list = []

            # Iterate over the folds with a nested tqdm progress bar
            for train_set, val_set, test_set in tqdm(datasets_list, desc="Folds", leave=False):
                if constraints == 'zero_shot':
                    ref_model = load_cog_model(
                        save_folder_path=f"Data/models_data_generating/full/models_zero_shot",
                        file_name=f"{reference_model_name}.pt"
                    )
                else: # Train the reference model
                    ref_model = CogModel(
                        model_name=reference_model_name,
                        images_features_dim=reference_model_features.shape[1],
                        constraints=constraints,
                        dim_reduction_method=dim_reduction_method
                    )

                    losses = ref_model.fit_W_matrix(
                        train_set=train_set,
                        images_features=reference_model_features,
                        reg_con=regularization_constant,
                        verbose=False,
                        reg_func=reg_func
                    )
                        
                if constraints == 'diagonal':
                        #make sure the W matrix is diagonal after the fit
                        assert is_diagonal(ref_model.W_mat.weight), "The W matrix is not diagonal"
                
                if constraints == 'zero_shot': #make sure no training was done
                    #make sure the W matrix is an identity matrix
                    assert torch.allclose(ref_model.W_mat.weight,torch.eye(ref_model.images_features_dim,device=ref_model.device)), "The W matrix is not an identity matrix"
                #set the model to eval mode and disable gradient calculation 
                with torch.no_grad():
                    # Evaluate the reference model
                    train_acc, train_nll = ref_model.test_model(train_set, reference_model_features, batch_size=batch_size)
                    val_acc, val_nll = ref_model.test_model(val_set, reference_model_features, batch_size=batch_size)
                    test_acc, test_nll = ref_model.test_model(test_set, reference_model_features, batch_size=batch_size)
                
                val_folds_acc_list.append(val_acc.item())
                val_folds_nll_list.append(val_nll.item())
                test_folds_acc_list.append(test_acc.item())
                test_folds_nll_list.append(test_nll.item())
                train_folds_acc_list.append(train_acc.item())
                train_folds_nll_list.append(train_nll.item())
                #Delete the reference model to ensure no leak between folds.
                del ref_model
                torch.cuda.empty_cache()

            # Calculate the mean of the folds    
            test_acc_mean = np.mean(test_folds_acc_list)
            val_acc_mean = np.mean(val_folds_acc_list)
            train_acc_mean = np.mean(train_folds_acc_list)
            test_nll_mean = np.mean(test_folds_nll_list)
            val_nll_mean = np.mean(val_folds_nll_list)
            train_nll_mean = np.mean(train_folds_nll_list)
            # Store the results in nested structure
            full_results_dict[reference_model_name][regularization_constant] = {
                "test_accuracy": test_acc_mean,
                "val_accuracy": val_acc_mean,
                "train_accuracy": train_acc_mean,
                "test_nll": test_nll_mean,
                "val_nll": val_nll_mean,
                "train_nll": train_nll_mean
            }
        # Get the best regularization constant based on validation accuracy
        best_reg_const = max(
            full_results_dict[reference_model_name].keys(), 
            key=lambda reg: full_results_dict[reference_model_name][reg]["val_accuracy"]
        )
        
        # Store best results
        best_results_dict[reference_model_name] = {
            "test_accuracy": full_results_dict[reference_model_name][best_reg_const]["test_accuracy"],
            "train_accuracy": full_results_dict[reference_model_name][best_reg_const]["train_accuracy"],
            "val_accuracy": full_results_dict[reference_model_name][best_reg_const]["val_accuracy"],
            "test_nll": full_results_dict[reference_model_name][best_reg_const]["test_nll"],
            "train_nll": full_results_dict[reference_model_name][best_reg_const]["train_nll"],
            "val_nll": full_results_dict[reference_model_name][best_reg_const]["val_nll"],
            "chosen_reg_con": best_reg_const,
            "dim_reduction_method": dim_reduction_method,
            "constraint_type": constraints if type(constraints) == str or constraints is None else constraints[0],
        }

    return full_results_dict, best_results_dict


def model_recovery_single_simulation_single_model_simplified(
    data_generating_model: CogModel,
    reference_model_names_list: List[str],
    data_generating_features_folder: Union[str, Path],
    reference_models_features_folder: Union[str, Path],
    N_concepts: int,
    N_train_triplets: int,
    seed: int,
    reg_func: str,
    triplets_to_exclude: Optional[torch.Tensor] = None,
    N_participants: int = 1,
    batch_size: Union[int, str] = 'full',
    K_folds: int = 5,
    regularization_constants_list: Optional[List[float]] = None,
    constraints: Optional[str] = None,
    dim_reduction_method: Optional[str] = None,
    
):
    """
    Run a single model recovery simulation with proper hyperparameter tuning.

    This function evaluates how well each reference model can recover the structure of data
    generated by a specific data-generating model. For each fold and reference model, it:
    1. Trains models with different regularization constants on training set
    2. Evaluates each on validation set to find best hyperparameter
    3. Retrains best model on train+validation data
    4. Evaluates final model on test set and records all metrics
    
    Parameters:
        data_generating_model (CogModel): The model used to generate simulated responses
        reference_model_names_list (List[str]): List of model names to evaluate as reference models
        data_generating_features_folder (Union[str, Path]): Path to the folder containing data-generating model features
        reference_models_features_folder (Union[str, Path]): Path to the folder containing reference model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets (int): Number of triplets to generate for the training set
        seed (int): Random seed for reproducibility
        reg_func (str): Regularization function to use
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation. 
                                                              Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 1.
        batch_size (Union[int, str], optional): Batch size for processing, or 'full' for full batch.
                                               Defaults to 'full'.
        K_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        regularization_constants_list (Optional[List[float]], optional): List of regularization constants
                                                                        to evaluate. Defaults to None.
        constraints (Optional[str], optional): Constraints on model parameter matrix (e.g., 'diagonal', 
                                              'zero_shot'). Defaults to None.
        dim_reduction_method (Optional[str], optional): Dimensionality reduction method to apply.
                                                       Defaults to None.
    
    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - full_results_dict: Nested dictionary mapping reference models to results for all
                                regularization constants and folds
            - best_results_dict: Dictionary mapping reference models to results with the best
                                regularization constant (based on validation accuracy)
                                
    Notes:
        - Uses create_model_recovery_datasets_simplified for cross-validation across concept subsets
        - For each fold: hyperparameter tuning on train/val, final evaluation on test
        - Results include test, validation, and training accuracy and negative log-likelihood (NLL)
        - The best regularization constant for each model is selected based on validation accuracy
        - Final model is retrained on train+val data and evaluated on test set
        - Special handling is applied for 'zero_shot' constraints (pre-trained models)
    """

    # Create the datasets using the simplified version
    datasets_list = create_model_recovery_datasets_simplified(
        data_generating_model=data_generating_model,
        features_folder=data_generating_features_folder,
        N_concepts=N_concepts,
        N_train_triplets=N_train_triplets,
        seed=seed,
        K_folds=K_folds,
        triplets_to_exclude=triplets_to_exclude,
        N_participants=N_participants,
        batch_size=batch_size,
    )
    
    # Create the results dictionary
    full_results_dict = {model_name: {} for model_name in reference_model_names_list}   
    best_results_dict = {}

    for reference_model_name in tqdm(reference_model_names_list, desc="Model recovery over reference models"):
        # Load the reference model features 
        if reference_model_name == 'Google_ViT_Large_224':
            reference_model_name_load = 'Google_ViT_Large'
            reference_model_features_path = reference_models_features_folder + "/" + f"{reference_model_name_load}.pt"
        else:
            reference_model_features_path = reference_models_features_folder + "/" + f"{reference_model_name}.pt"
        
        reference_model_features = torch.load(reference_model_features_path)
        
        # Initialize results storage for this model across all folds
        all_fold_results = []  # Store best results from each fold
        
        # Process each fold
        for fold_idx, (train_set, val_set, test_set) in enumerate(datasets_list):
            print(f"Processing fold {fold_idx + 1}/{K_folds} for {reference_model_name}")
            
            # Step 1: Hyperparameter tuning - train on train_set, evaluate on val_set
            best_val_acc = -1
            best_reg_const = None
            hyperparams_results = {}
            
            for regularization_constant in tqdm(regularization_constants_list, 
                                              desc=f"Hyperparameter tuning fold {fold_idx + 1}", leave=False):
                
                if constraints == 'zero_shot':
                    ref_model = load_cog_model(
                        save_folder_path=f"Data/models_data_generating/full/models_zero_shot",
                        file_name=f"{reference_model_name}.pt"
                    )
                else:
                    # Train model on training set only
                    ref_model = CogModel(
                        model_name=reference_model_name,
                        images_features_dim=reference_model_features.shape[1],
                        constraints=constraints,
                        dim_reduction_method=dim_reduction_method
                    )
                    
                    losses = ref_model.fit_W_matrix(
                        train_set=train_set,
                        images_features=reference_model_features,
                        reg_con=regularization_constant,
                        verbose=False,
                        reg_func=reg_func
                    )
                
                # Validate constraints
                if constraints == 'diagonal':
                    assert is_diagonal(ref_model.W_mat.weight), "The W matrix is not diagonal"
                if constraints == 'zero_shot':
                    assert torch.allclose(ref_model.W_mat.weight,torch.eye(ref_model.images_features_dim,device=ref_model.device)), "The W matrix is not an identity matrix"
                
                # Evaluate on validation set
                with torch.no_grad():
                    val_acc, val_nll = ref_model.test_model(val_set, reference_model_features, batch_size=batch_size)
                
                hyperparams_results[regularization_constant] = {
                    'val_acc': val_acc.item(),
                    'val_nll': val_nll.item()
                }
                
                # Track best hyperparameter
                if val_acc.item() > best_val_acc:
                    best_val_acc = val_acc.item()
                    best_reg_const = regularization_constant
                
                # Clean up
                del ref_model
                torch.cuda.empty_cache()
            # Combine train and validation sets
            from triplets_data_set import OddOneOutDataset
            train_val_triplets = torch.cat([train_set.triplets, val_set.triplets], dim=0)
            train_val_choices = torch.cat([train_set.choice_position, val_set.choice_position], dim=0)
            train_val_combined = OddOneOutDataset(train_val_triplets, train_val_choices)
            
            # Step 2: Retrain best model on train+val data, evaluate on test
            if constraints == 'zero_shot':
                final_model = load_cog_model(
                    save_folder_path=f"Data/models_data_generating/full/models_zero_shot",
                    file_name=f"{reference_model_name}.pt"
                )
            
            else:
                # Train final model on combined data
                final_model = CogModel(
                    model_name=reference_model_name,
                    images_features_dim=reference_model_features.shape[1],
                    constraints=constraints,
                    dim_reduction_method=dim_reduction_method
                )
                
                losses = final_model.fit_W_matrix(
                    train_set=train_val_combined,
                    images_features=reference_model_features,
                    reg_con=best_reg_const,
                    verbose=False,
                    reg_func=reg_func
                )
            
            # Step 3: Final evaluation on all sets (including train+val combined)
            with torch.no_grad():
                train_acc, train_nll = final_model.test_model(train_set, reference_model_features, batch_size=batch_size)
                val_acc, val_nll = final_model.test_model(val_set, reference_model_features, batch_size=batch_size)
                test_acc, test_nll = final_model.test_model(test_set, reference_model_features, batch_size=batch_size)
                # Evaluate on the combined train+val set (what the model was actually trained on)
                trainval_acc, trainval_nll = final_model.test_model(train_val_combined, reference_model_features, batch_size=batch_size)
            
            # Debug: Print dataset sizes
            print(f"Fold {fold_idx + 1} - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}, Train+Val: {len(train_val_combined)}")
            
            # Store best results for this fold
            fold_result = {
                "test_accuracy": test_acc.item(),
                "val_accuracy": val_acc.item(),
                "train_accuracy": train_acc.item(),
                "trainval_accuracy": trainval_acc.item(),  # Performance on what the model was trained on
                "test_nll": test_nll.item(),
                "val_nll": val_nll.item(),
                "train_nll": train_nll.item(),
                "trainval_nll": trainval_nll.item(),
                "best_reg_const": best_reg_const,
                "fold_idx": fold_idx
            }
            
            all_fold_results.append(fold_result)
            
            # Store individual fold results in full_results_dict
            full_results_dict[reference_model_name][f"fold_{fold_idx}"] = fold_result
            
            # Clean up
            del final_model
            torch.cuda.empty_cache()
        
        # Step 4: Calculate mean results across all folds (without grouping by regularization constant)
        if all_fold_results:
            # Store best results as mean across all folds
            best_results_dict[reference_model_name] = {
                "test_accuracy": np.mean([r["test_accuracy"] for r in all_fold_results]),
                "train_accuracy": np.mean([r["train_accuracy"] for r in all_fold_results]),
                "val_accuracy": np.mean([r["val_accuracy"] for r in all_fold_results]),
                "trainval_accuracy": np.mean([r["trainval_accuracy"] for r in all_fold_results]),  # Performance on training data
                "test_nll": np.mean([r["test_nll"] for r in all_fold_results]),
                "train_nll": np.mean([r["train_nll"] for r in all_fold_results]),
                "val_nll": np.mean([r["val_nll"] for r in all_fold_results]),
                "trainval_nll": np.mean([r["trainval_nll"] for r in all_fold_results]),
                "dim_reduction_method": dim_reduction_method,
                "constraint_type": constraints if type(constraints) == str or constraints is None else constraints[0],
                "reg_constants_used": [r["best_reg_const"] for r in all_fold_results],  # Track which reg constants were chosen
                "n_folds": len(all_fold_results)
            }

    return full_results_dict, best_results_dict


def model_recovery_over_data_generating_models(
    data_generating_models:List[CogModel],
    reference_model_names_list:List[str],
    features_folder:Union[str,Path],
    N_concepts: int,
    N_train_triplets:int,
    seed : int,
    triplets_to_exclude:Optional[torch.Tensor] = None,
    N_participants:int = 100,
    batch_size:Union[int,str] = 100,
    K_folds:int = 3,
    regularization_constants_list:Optional[List[float]] = None
):
    """
    Run model recovery experiments across multiple data-generating models.
    
    This function iterates over a list of data-generating models and runs the model recovery
    experiment for each one, evaluating how well each reference model can recover the structure
    of data generated by each data-generating model.
    
    Parameters:
        data_generating_models (List[CogModel]): List of models to use as data generators
        reference_model_names_list (List[str]): List of model names to evaluate as reference models
        features_folder (Union[str, Path]): Path to the folder containing model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets (int): Number of triplets to generate for the training set
        seed (int): Random seed for reproducibility
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation.
                                                              Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 100.
        batch_size (Union[int, str], optional): Batch size for processing. Defaults to 100.
        K_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        regularization_constants_list (Optional[List[float]], optional): List of regularization
                                                                       constants to evaluate. 
                                                                       Defaults to None.
    
    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - over_dg_full_results_dict: Nested dictionary mapping data-generating models to full results
                                        for all reference models and regularization constants
            - over_dg_best_results_dict: Nested dictionary mapping data-generating models to best results
                                        for all reference models
                                        
    Notes:
        - This is a higher-level function that calls model_recovery_single_simulation_single_model
          for each data-generating model
        - Results are organized in a nested structure to preserve the relationship between
          data-generating models and reference models
    """
    # Create the results dictionaries
    over_dg_full_results_dict = {data_generating_model.model_name: {} for data_generating_model in data_generating_models}
    over_dg_best_results_dict = {data_generating_model.model_name: {} for data_generating_model in data_generating_models}
    # Iterate over the data-generating models
    for data_generating_model in data_generating_models:
        # Create the results entry for the current data-generating model
        full_results_dict,best_results_dict = model_recovery_single_simulation_single_model(
            data_generating_model=data_generating_model,
            reference_model_names_list=reference_model_names_list,
            features_folder=features_folder,
            N_concepts=N_concepts,
            N_train_triplets=N_train_triplets,
            seed=seed,
            triplets_to_exclude=triplets_to_exclude,
            N_participants=N_participants,
            batch_size=batch_size,
            K_folds=K_folds,
            regularization_constants_list=regularization_constants_list
        )

        over_dg_full_results_dict[data_generating_model.model_name] = full_results_dict
        over_dg_best_results_dict[data_generating_model.model_name] = best_results_dict
    return over_dg_full_results_dict,over_dg_best_results_dict  


def model_recovery_over_different_N_train_triplets(
    data_generating_models:List[CogModel],
    reference_model_names_list:List[str],
    features_folder:Union[str,Path],
    N_concepts: int,
    N_train_triplets_list:List[int],
    triplets_to_exclude:Optional[torch.Tensor] = None,
    N_participants:int = 100,
    batch_size:Union[int,str] = 100,
    K_folds:int = 3,
    regularization_constants_list:Optional[List[float]] = None,
    N_simulations:int = 10,
    simulations_seed_list:Optional[List[int]] = None
):
    """
    Run model recovery experiments across multiple training set sizes and simulation runs.
    
    This function conducts a comprehensive analysis by varying both the number of training
    triplets and running multiple simulations with different random seeds. It evaluates how
    well reference models can recover data-generating models under these different conditions.
    
    Parameters:
        data_generating_models (List[CogModel]): List of models to use as data generators
        reference_model_names_list (List[str]): List of model names to evaluate as reference models
        features_folder (Union[str, Path]): Path to the folder containing model features
        N_concepts (int): Total number of concepts to use
        N_train_triplets_list (List[int]): List of different training set sizes to evaluate
        triplets_to_exclude (Optional[torch.Tensor], optional): Triplets to exclude from generation.
                                                              Defaults to None.
        N_participants (int, optional): Number of simulated participants. Defaults to 100.
        batch_size (Union[int, str], optional): Batch size for processing. Defaults to 100.
        K_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        regularization_constants_list (Optional[List[float]], optional): List of regularization
                                                                      constants to evaluate.
                                                                      Defaults to None.
        N_simulations (int, optional): Number of simulation runs to perform for each
                                      training set size. Defaults to 10.
        simulations_seed_list (Optional[List[int]], optional): List of random seeds for simulations.
                                                             If None, random seeds are generated.
                                                             Defaults to None.
    
    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - final_full_results_dict: Nested dictionary mapping triplet counts and simulation indices
                                      to full results for all data-generating and reference models
            - final_best_results_dict: Nested dictionary mapping triplet counts and simulation indices
                                      to best results for all data-generating and reference models
                                      
    Notes:
        - This is the highest-level function in the model recovery framework, allowing for
          analysis across multiple dimensions (triplet counts, simulations)
        - Random seeds can be provided for reproducibility or generated on the fly
        - Results are organized in a deeply nested structure to preserve the relationships between
          triplet counts, simulations, data-generating models, and reference models
        - This function is typically used for large-scale experiments to understand model
          recovery behavior across varying data sizes and random initializations
    """
    # Create the simulation seed list if none is provided
    if simulations_seed_list is None:
        simulations_seed_list = np.random.randint(0,1000,size=N_simulations)
        print(f"simulations_seed_list created based on N_simulations: {simulations_seed_list}")
    # Create the results dictionaries
    final_full_results_dict = {N_train_triplets: {} for N_train_triplets in N_train_triplets_list}
    final_best_results_dict = {N_train_triplets: {} for N_train_triplets in N_train_triplets_list}
    # Iterate over the requested numbers of training triplets and simulation indices
    for N_train_triplets in N_train_triplets_list:
        for simulation in range(N_simulations):
            # Retrieve the simulation seed
            seed = simulations_seed_list[simulation]
            # Run the model recovery procedure over the data-generating models
            over_dg_full_results_dict,over_dg_best_results_dict = model_recovery_over_data_generating_models(
                data_generating_models=data_generating_models,
                reference_model_names_list=reference_model_names_list,
                features_folder=features_folder,
                N_concepts=N_concepts,
                N_train_triplets=N_train_triplets,
                seed=seed,
                triplets_to_exclude=triplets_to_exclude,
                N_participants=N_participants,
                batch_size=batch_size,
                K_folds=K_folds,
                regularization_constants_list=regularization_constants_list
            )
            # Store the results
            final_full_results_dict[N_train_triplets][simulation] = over_dg_full_results_dict
            final_best_results_dict[N_train_triplets][simulation] = over_dg_best_results_dict
    return final_full_results_dict,final_best_results_dict
