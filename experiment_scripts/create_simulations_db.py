"""
Create Simulations Database for Model Recovery Experiments

This script initializes a SQLite database with all the simulation jobs needed for model recovery
experiments. It uses Hydra for configuration management and creates a comprehensive job database
that tracks:
- Different training triplet counts (dataset sizes)
- Multiple simulation runs per condition
- Various data-generating models (neural network models)
- Different constraint types (e.g., full_W, diagonal, zero_shot)
- Dimensionality reduction methods (e.g., PCA_500)
- Regularization methods (e.g., L1, L2, eye_distance)

The database will be used by worker scripts to fetch and execute simulation jobs in parallel.

Configuration:
    The script is configured via Hydra using create_simulations_db.yaml which defines:
    - Experiment types (full_W, full_W_L1, zero_shot, diagonal, etc.)
    - Paths to data files and results folders
    - Experiment parameters (N_train_triplets_list, N_simulations, etc.)
    - Model lists imported from shared_args.yaml

Usage:
    python create_simulations_db.py experiment_type=<experiment_name>
    
    Examples:
        python create_simulations_db.py experiment_type=full_W
        python create_simulations_db.py experiment_type=full_W_L1
        python create_simulations_db.py experiment_type=zero_shot

Output:
    Creates a SQLite database at: Results/Simulations_experiments/<results_file_name>/<results_file_name>.db
    The database contains two main tables:
    - jobs: All simulation jobs to be executed
    - reference_model_results: Performance metrics for each reference model (populated by worker scripts)
"""

import os
import sys
from pathlib import Path

# Set up paths relative to project root
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))

import re
import hydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from tools.db_tools import init_database, create_jobs_from_combinations


def parse_tuple_string(s):
    """
    Parse a string representation of a tuple into an actual tuple.
    
    This function handles constraint specifications that are written as tuples in the
    configuration file, such as ("rectangular", 30) for rectangular constraints with
    dimensionality 30.
    
    Args:
        s: String to parse, expected format: "(string_value, integer_value)"
        
    Returns:
        tuple: (str, int) if the string matches the expected pattern
        str: Original string if it doesn't match the pattern
        
    Example:
        >>> parse_tuple_string("(rectangular, 30)")
        ('rectangular', 30)
        >>> parse_tuple_string("full_W")
        'full_W'
    """
    match = re.fullmatch(r'\(\s*([a-zA-Z_]+)\s*,\s*(\d+)\s*\)', s)
    if match:
        text_part = match.group(1)
        int_part = int(match.group(2))
        return (text_part, int_part)
    else:
        return s


@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="create_simulations_db.yaml")
def main(cfg: DictConfig):
    """
    Main function to create simulation database based on Hydra configuration.
    
    This function:
    1. Loads experiment configuration from YAML files
    2. Validates and processes configuration parameters
    3. Creates results directory structure
    4. Initializes SQLite database with proper schema
    5. Creates all simulation jobs based on parameter combinations
    
    Args:
        cfg: Hydra DictConfig object containing all experiment parameters
        
    The function reads the experiment_type from cfg to select which experiment
    configuration to use (e.g., full_W, full_W_L1, zero_shot, diagonal, etc.)
    """
    experiment_type = cfg.experiment_type
    print(f"\nCreating simulation database for experiment type: {experiment_type}")
    
    # Load the specific experiment configuration
    CONFIG_DICT = cfg[experiment_type]
    
    # Convert ListConfig constraints to tuple for consistency
    c = CONFIG_DICT.experiment.constraints
    if isinstance(c, ListConfig) or isinstance(c, list):
        CONFIG_DICT.experiment.constraints = tuple(c)
    
    print(f"\nExperiment Configuration:")
    print(f"- Constraints: {CONFIG_DICT.experiment.constraints}")
    print(f"- Dimension Reduction Method: {CONFIG_DICT.experiment.dim_reduction_method}")
    print(f"- Number of simulations per condition: {CONFIG_DICT.experiment.N_simulations}")
    
    # Determine which models to use: base models or base + additional models
    data_generating_models_names_list = cfg.models if not CONFIG_DICT['include_additional_models'] else cfg.models + cfg.new_models
    
    # Set default regularization methods if not specified
    # Default is "eye_distance" which represents scalar matrix shrinkage regularization
    if CONFIG_DICT.experiment.candidate_model_regularization is None:
        CONFIG_DICT.experiment.candidate_model_regularization = "eye_distance"
    if CONFIG_DICT.experiment.data_generating_model_regularization is None:
        CONFIG_DICT.experiment.data_generating_model_regularization = "eye_distance"
        
    # Extract regularization settings for data-generating and candidate models
    data_gen_reg = CONFIG_DICT.experiment.data_generating_model_regularization
    candidate_reg = CONFIG_DICT.experiment.candidate_model_regularization
    
    # Convert Hydra ListConfig to regular Python lists for easier manipulation
    if isinstance(data_gen_reg, ListConfig):
        data_gen_reg = list(data_gen_reg)
    if isinstance(candidate_reg, ListConfig):
        candidate_reg = list(candidate_reg)
    
    # Calculate number of regularization combinations
    # Handle both single values and lists of regularization methods
    if isinstance(data_gen_reg, (list, ListConfig)):
        len_data_gen_reg = len(data_gen_reg)
    else:
        len_data_gen_reg = 1
        
    if isinstance(candidate_reg, (list, ListConfig)):
        len_candidate_reg = len(candidate_reg)
    else:
        len_candidate_reg = 1
    
    # Total regularization combinations (cross product)
    len_regularization = len_data_gen_reg * len_candidate_reg
    
    # Set up results folder structure and database file path
    results_folder = f"{CONFIG_DICT.paths.results_folder}/{CONFIG_DICT.paths.results_file_name}"
    db_file_name = f"{results_folder}/{CONFIG_DICT.paths.results_file_name}.db"
    
    # Calculate total number of simulations to be created
    # This is the cross-product of all experimental parameters
    len_data_generating_models = len(data_generating_models_names_list)
    len_constraints = len(CONFIG_DICT.experiment.constraints) if CONFIG_DICT.experiment.constraints is not None else 1
    len_dim_reduction_method = len(CONFIG_DICT.experiment.dim_reduction_method) if CONFIG_DICT.experiment.dim_reduction_method is not None else 1
    len_N_train_triplets_list = len(CONFIG_DICT.experiment.N_train_triplets_list)
    len_N_simulations = CONFIG_DICT.experiment.N_simulations
    
    total_simulations = (len_data_generating_models * len_constraints * len_dim_reduction_method * 
                        len_N_train_triplets_list * len_N_simulations * len_regularization)
    
    # Print simulation statistics
    print(f"\nSimulation Statistics:")
    print(f"- Total number of simulations: {total_simulations}")
    print(f"- Number of simulations per dataset size: {total_simulations/len_N_train_triplets_list}")
    print(f"- Number of simulations per data generating model: {total_simulations/(len_data_generating_models)}")
    print(f"- Number of simulations per initialization: {total_simulations/(len_N_simulations)}")
    
    print(f"\nDatabase Configuration:")
    print(f"- Results folder: {results_folder}")
    print(f"- Database file: {db_file_name}")

    # Create and initialize database if it doesn't exist
    if not os.path.exists(db_file_name):
        print("\nCreating new database...")
        # Create results folder structure if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
        
        # Initialize database with proper schema (creates jobs and reference_model_results tables)
        init_database(db_file_name)
        
        print(f"\nCreating simulation jobs:")
        print(f"- Number of different triplet sizes: {len(CONFIG_DICT.experiment.N_train_triplets_list)}")
        print(f"- Triplet sizes: {CONFIG_DICT.experiment.N_train_triplets_list}")
        print(f"- Data generating model regularization: {CONFIG_DICT.experiment.data_generating_model_regularization}")
        print(f"- Candidate model regularization: {CONFIG_DICT.experiment.candidate_model_regularization}")
        print(f"- Number of simulations: {CONFIG_DICT.experiment.N_simulations}")
        print(f"- Constraints: {CONFIG_DICT.experiment.constraints}")
        print(f"- Dimension Reduction Method: {CONFIG_DICT.experiment.dim_reduction_method}")
    
        # Create jobs for all combinations of experimental parameters
        # This generates the full cross-product of: n_triplets × simulations × models × constraints × dim_reduction × regularizations
        create_jobs_from_combinations(
            db_path=db_file_name,
            n_triplets_list=CONFIG_DICT.experiment.N_train_triplets_list,
            n_simulations=CONFIG_DICT.experiment.N_simulations,
            data_generating_models=data_generating_models_names_list,
            constraints=CONFIG_DICT.experiment.constraints,
            dim_reduction_method=CONFIG_DICT.experiment.dim_reduction_method,
            data_generating_model_regularization=data_gen_reg,
            candidate_model_regularization=candidate_reg
        )
        print("\nDatabase creation completed successfully!")
        print(f"The database is located at: {db_file_name}")
    else:
        raise ValueError(f"The database file already exists at {db_file_name}. "
                        "Please delete it if you want to create a new one, or use a different results_file_name in the configuration.")

if __name__ == "__main__":
    main()

        
        