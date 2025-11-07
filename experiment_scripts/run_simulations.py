"""
Run Model Recovery Simulations - Main Execution Script
=======================================================

This script executes model recovery simulation jobs by:
1. Fetching pending jobs from the simulation database
2. Loading pre-trained data-generating models
3. Running model recovery experiments for each job
4. Saving results back to the database

CRITICAL WORKFLOW INTEGRATION:
-------------------------------
This script is the THIRD stage in a three-stage pipeline:

STAGE 1: create_data_generating_models.py
  - Fits neural network features to THINGS behavioral data
  - Creates CogModel instances with learned W matrices
  - Saves to Data/models_data_generating_<regularization_type (empty if its eye_distance)>/<dim_type>/<constraint_type>/
  - Output: Pre-trained data-generating models (.pkl files)

STAGE 2: create_simulations_db.py
  - Creates SQLite database with all simulation jobs
  - Jobs specify: n_triplets, simulation_idx, model, constraint
  - Output: Database with 'jobs' and 'reference_model_results' tables

STAGE 3: run_simulations.py (THIS SCRIPT)
  - Fetches pending jobs atomically from database
  - Loads data-generating model from Stage 1 output
  - Generates synthetic behavioral data
  - Tests all reference models on the data
  - Saves results to database
  - Repeats until no pending jobs or time limit reached

PARALLELIZATION & CLUSTER DEPLOYMENT:
--------------------------------------
This script is designed for parallel execution on shared storage systems.

**Single Machine**:
  Multiple worker instances can run simultaneously on one machine.
  Jobs are claimed atomically via SQLite's BEGIN EXCLUSIVE locking.

**Cluster Environment** (e.g., BGU ISE-CS-DT cluster):
  - Deploy multiple workers across different cluster nodes
  - All workers access the same shared database via network filesystem (NFS/similar)
  - Database locking ensures thread-safe job claiming across nodes
  - No coordination required - workers automatically discover pending jobs
  - Tested on BGU ISE-CS-DT cluster with 10+ simultaneous workers

**Example Cluster Deployment**:
  ```bash
  # Submit multiple SLURM/PBS jobs, each running this script
  for i in {1..10}; do
      sbatch --time=24:00:00 --mem=32G \\
             run_worker.sh experiment_type=full_W
  done
  ```

**Benefits of Parallel Execution**:
  - Linear speedup: 10 workers = ~10x faster completion
  - Fault tolerance: Worker failures don't affect other workers
  - Dynamic load balancing: Fast jobs finish early, workers continue
  - No job resubmission needed: Workers run until time limit or jobs exhausted

Usage:
------
# Single worker
python run_simulations.py experiment_type=full_W

# Multiple workers (parallel on same machine)
for i in {1..4}; do
    python run_simulations.py experiment_type=full_W &
done

# Cluster deployment (SLURM example)
sbatch --array=1-10 run_simulations_worker.sh

"""

import os
import sys
from pathlib import Path

# Set up paths relative to project root
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))

import time
import json
import re
import hydra
from omegaconf import DictConfig
import torch 

from tools.model_recovery_fn import model_recovery_single_simulation_single_model
from tools.model_io import load_cog_model
from tools.db_tools import get_pending_job, save_reference_model_results



def parse_tuple_string(s):
    """
    Parse string representation of a tuple into an actual tuple.
    
    Handles constraint specifications stored as strings in the database.
    For example, rectangular constraints like ("rectangular", 30) are 
    stored as strings and need to be converted back to tuples.
    
    Parameters:
        s (str): String to parse, format: "(text_part, integer_part)"
        
    Returns:
        Union[Tuple[str, int], str]: 
            - Tuple (text, int) if string matches pattern
            - Original string otherwise (e.g., "diagonal", "None")
        
    Examples:
        >>> parse_tuple_string('(rectangular, 30)')
        ('rectangular', 30)
        >>> parse_tuple_string('diagonal')
        'diagonal'
    """
    match = re.fullmatch(r'\(\s*([a-zA-Z_]+)\s*,\s*(\d+)\s*\)', s)
    if match:
        text_part = match.group(1)
        int_part = int(match.group(2))
        return (text_part, int_part)
    else:
        return s

    
@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="run_simulations.yaml")
def main(cfg: DictConfig):
    """
    Main function to run model recovery simulations.
    
    WORKFLOW:
    ---------
    1. Load Configuration: Extract experiment type and paths from Hydra config
    2. Get Pending Job: Atomically fetch job from database (thread-safe)
    3. Load Data-Generating Model: Load pre-trained CogModel from .pkl file
    4. Setup Simulation Parameters: Combine job info with config settings
    5. Run Simulation: Execute model_recovery_single_simulation_single_model()
    6. Save Results: Store to database and JSON files
    
    CONFIGURATION STRUCTURE:
    ------------------------
    cfg contains:
      - experiment_type: Which experiment to run (full_W, diagonal, etc.)
      - [experiment_type]: Nested config with paths and parameters
        - paths:
            - data_generating_models: Path to fitted CogModels
            - reference_models_features_folder: Path to neural features
            - db_file_name: Database filename
        - experiment:
            - K_folds: CV folds for regularization selection
            - reg_func: Regularization type (L1, L2, eye_distance)
            - regularization_constants_list: Grid to search
            - N_concepts, N_participants: Simulation parameters
    
    JOB STRUCTURE (from database):
    ------------------------------
    Each job contains:
      - id: Unique job identifier
      - n_train_triplets: Training set size
      - simulation_idx: Random seed for reproducibility
      - data_generating_model: Model name (e.g., "ResNet50")
      - constraint_type: W matrix constraint ("None", "diagonal", etc.)
      - dim_reduction_method: Feature reduction ("None", "PCA_500")
      - data_generating_model_regularization: Reg used for data-gen model
      - candidate_model_regularization: Reg to use for candidate models
    
    REGULARIZATION HANDLING:
    ------------------------
    Two cases:
    1. Single regularization (most experiments):
       - data_gen_reg == candidate_reg
       - Use single data_generating_models path
       
    2. Cross-regularization comparison (special experiments):
       - data_gen_reg != candidate_reg
       - Config must have both _L2 and _eye_distance paths
       - Selects path based on data_gen_reg from job
    
    SIMULATION EXECUTION:
    ---------------------
    For each job:
      1. Generate N_train_triplets from data-gen model's concepts
      2. Simulate responses using data-gen model's W + temperature
      3. For each reference model (20-30 models):
         a. Load features
         b. Test regularization constants via K-fold CV
         c. Select best constant (highest validation accuracy)
         d. Retrain on train+val, test on held-out fold
         e. Record: test_acc, train_acc, val_acc, NLL, chosen_reg_con
      4. Save all reference model results
    
    OUTPUTS:
    --------
    1. Database: reference_model_results table (efficient querying)
    2. JSON files (debugging/inspection):
       - full_results: All regularization constants tested
       - best_results: Best-performing regularization per model
    
    PARALLEL EXECUTION & SHARED STORAGE:
    ------------------------------------
    This function is designed for cluster deployment with shared storage:
    
    **Shared Storage Requirements**:
      - Database file must be accessible from all worker nodes
      - Feature files must be on shared filesystem (NFS/Lustre/etc.)
      - Model files must be on shared filesystem
      - Results folder must be writable from all nodes
    
    **Atomic Job Claiming**:
      - get_pending_job() uses SQLite's BEGIN EXCLUSIVE transaction
      - Ensures only one worker claims each job across all cluster nodes
      - Database locking prevents race conditions on shared filesystems
      - Tested on BGU ISE-CS-DT cluster with NFS-mounted shared storage
    
    **Cluster Best Practices**:
      - Use network filesystem (NFS, Lustre, GPFS) for all data paths
      - Ensure sufficient file handle limits for concurrent database access
      - Monitor disk I/O if many workers access same features simultaneously
      - Set appropriate time limits (23.5 hours) for cluster schedulers
    
    Parameters:
        cfg (DictConfig): Hydra configuration object with experiment settings
        
    Raises:
        ValueError: If paths don't match expected structure
        FileNotFoundError: If data-generating model file not found
        
    Notes:
        - Function is called repeatedly in while loop until time limit
        - Each call processes exactly one job
        - Job claiming is atomic (thread-safe for parallel execution)
        - Safe for deployment on cluster systems with shared storage
        - Tested with 10+ simultaneous workers on BGU ISE-CS-DT cluster
    """
    # =========================================================================
    # STEP 1: Load Configuration and Setup Paths
    # =========================================================================
    results_folder = cfg.db_folder
    experiment_type = cfg.experiment_type
    CONFIG_dict = cfg[experiment_type]
    
    # =========================================================================
    # STEP 2: Build Reference Models List from Features Folder
    # =========================================================================
    # Automatically discover all available models by scanning features folder
    models_features_path = CONFIG_dict['paths']['reference_models_features_folder']
    # Build candidate models list
    candidate_models_names_list = cfg.models if not CONFIG_dict['experiment']['used_additional_models'] else cfg.models + cfg.additional_models

    # Validate all models exist in one operation
    models_features_path = CONFIG_dict['paths']['reference_models_features_folder']
    available_models = {f.replace('.pt', '') for f in os.listdir(models_features_path) if f.endswith('.pt')}
    missing_models = set(candidate_models_names_list) - available_models
    if missing_models:
        raise FileNotFoundError(
            f"Missing {len(missing_models)} model features in {models_features_path}:\n" +
            "\n".join(f"  • {m}" for m in sorted(missing_models))
        )
    print(f"✓ Validated {len(candidate_models_names_list)} reference models")


    # =========================================================================
    # STEP 3: Load Database and Get Pending Job
    # =========================================================================
    db_path = os.path.join(results_folder, experiment_type, CONFIG_dict['paths']['db_file_name'])
    triplets_to_exclude = torch.load(CONFIG_dict['paths']['triplets_to_exclude'])
    
    # Atomically fetch a pending job (thread-safe for parallel execution)
    job = get_pending_job(db_path=db_path)
    assert job is not None, "No pending jobs available in the database"
    # =========================================================================
    # STEP 4: Determine Regularization Configuration
    # =========================================================================
    # Handle two cases:
    # Case 1: Single regularization (data_gen_reg == candidate_reg)
    # Case 2: Cross-regularization comparison (data_gen_reg != candidate_reg)
    
    paths_config = CONFIG_dict['paths']
    
    # Extract regularization info from job (may be None for single-reg experiments)
    data_gen_reg = job.get('data_generating_model_regularization', 'None')
    candidate_reg = job.get('candidate_model_regularization', 'None')
    
    # Case 1: Single regularization experiment
    if data_gen_reg == 'None' or data_gen_reg is None or data_gen_reg == candidate_reg:
        if 'data_generating_models' not in paths_config:
            raise ValueError("Single regularization mode requires 'data_generating_models' path in configuration")
        
        data_generating_models_path = CONFIG_dict['paths']['data_generating_models']
        reg_func = CONFIG_dict['experiment'].get('reg_func', 'eye_distance')
        
        if reg_func == "None":
            reg_func = "eye_distance"
        
        candidate_reg = reg_func
    
    # Case 2: Cross-regularization comparison experiment
    else:
        if 'data_generating_models_L2' not in paths_config or 'data_generating_models_eye_distance' not in paths_config:
            raise ValueError("Cross-regularization mode requires both 'data_generating_models_L2' "
                           "and 'data_generating_models_eye_distance' paths in configuration")
        
        # Select path based on data-generating model's regularization type
        if data_gen_reg == 'L2':
            data_generating_models_path = CONFIG_dict['paths']['data_generating_models_L2']
        elif data_gen_reg == 'eye_distance':
            data_generating_models_path = CONFIG_dict['paths']['data_generating_models_eye_distance']
        else:
            raise ValueError(f"Unsupported data_generating_model_regularization: {data_gen_reg}. "
                           f"Expected 'L2' or 'eye_distance'")
        
        # Use candidate model regularization for reg_func
        if candidate_reg == 'None' or candidate_reg is None:
            reg_func = 'eye_distance'
            candidate_reg = 'eye_distance'
        else:
            reg_func = candidate_reg
    
    # =========================================================================
    # STEP 5: Extract and Prepare Simulation Arguments
    # =========================================================================
    # Parse constraint type (may be a tuple string like "(rectangular, 30)")
    job['constraint_type'] = parse_tuple_string(job['constraint_type'])
    
    # Extract parameters from config
    K_folds = CONFIG_dict['experiment']['K_folds']
    N_participants = CONFIG_dict['experiment']['N_participants']
    N_concepts = CONFIG_dict['experiment']['N_concepts']
    regularization_constants_list = CONFIG_dict['experiment']['regularization_constants_list']
    
    # Extract parameters from job
    N_train_triplets = job['n_train_triplets']
    seed = job['simulation_idx']
    data_generating_model = job['data_generating_model']
    constraint_type = job['constraint_type']
    dim_reduction_method = job['dim_reduction_method']
    
    # Get paths for features
    data_generating_features_folder = CONFIG_dict['paths']['data_generating_features_folder']
    reference_models_features_folder = CONFIG_dict['paths']['reference_models_features_folder']
    
    # Build arguments dictionary for simulation function
    experiments_args = {
        "data_generating_model": load_cog_model(data_generating_models_path, data_generating_model),
        "reference_model_names_list": candidate_models_names_list,
        "data_generating_features_folder": data_generating_features_folder,
        "reference_models_features_folder": reference_models_features_folder,
        "N_concepts": N_concepts,
        "N_train_triplets": N_train_triplets,
        "seed": seed, 
        "triplets_to_exclude": triplets_to_exclude,
        "N_participants": N_participants,
        "K_folds": K_folds,
        "regularization_constants_list": regularization_constants_list,
        "constraints": constraint_type if constraint_type != "None" else None,
        "dim_reduction_method": dim_reduction_method if dim_reduction_method != "None" else None,
        "reg_func": candidate_reg
    }
    
    # =========================================================================
    # STEP 6: Print Job Information
    # =========================================================================
    print("\n" + "="*80)
    print(f"STARTING JOB #{job['id']}")
    print("="*80)
    
    print("\n📋 JOB CONFIGURATION:")
    print(f"  Data-Generating Model:     {data_generating_model}")
    print(f"  Model Path:                {data_generating_models_path}")
    print(f"  Training Triplets:         {N_train_triplets:,}")
    print(f"  Simulation Index (Seed):   {seed}")
    print(f"  Constraint Type:           {constraint_type}")
    print(f"  Dim Reduction Method:      {dim_reduction_method}")
    print(f"  Data-Gen Regularization:   {data_gen_reg}")
    print(f"  Candidate Regularization:  {candidate_reg}")
    
    print("\n⚙️  SIMULATION PARAMETERS:")
    print(f"  N_concepts:                {N_concepts}")
    print(f"  N_participants:            {N_participants}")
    print(f"  K_folds (CV):              {K_folds}")
    print(f"  Reg constants to test:     {len(regularization_constants_list)} values")
    print(f"  Reference models:          {len(candidate_models_names_list)} models")
    
    print("\n🔬 RUNNING MODEL RECOVERY SIMULATION...")
    print("-"*80)
    
    # =========================================================================
    # STEP 7: Execute Model Recovery Simulation
    # =========================================================================
    full_results_dict, best_results_dict = model_recovery_single_simulation_single_model(**experiments_args)
    
    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("-"*80)
    print(f"✅ JOB #{job['id']} COMPLETED SUCCESSFULLY")
    print("\n💾 SAVING RESULTS...")
    
    # Prepare results for database
    best_results_list = []
    for model_name in candidate_models_names_list:
        best_results_list.append({
            'name': model_name,
            'test_accuracy': best_results_dict[model_name]['test_accuracy'],
            'train_accuracy': best_results_dict[model_name]['train_accuracy'],
            'validation_accuracy': best_results_dict[model_name]['val_accuracy'],
            'test_nll': best_results_dict[model_name]['test_nll'],
            'train_nll': best_results_dict[model_name]['train_nll'],
            'validation_nll': best_results_dict[model_name]['val_nll'],
            "chosen_reg_con": best_results_dict[model_name]['chosen_reg_con'],
        })
    
    # Save to database
    save_reference_model_results(
        db_path=db_path,
        job_id=job['id'],
        reference_models_list=best_results_list
    )
    print(f"  ✓ Database updated: {len(best_results_list)} reference models saved")
    
    # Save to JSON files (for debugging/inspection)
    json_folder = os.path.join(results_folder, experiment_type, "json")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    
    full_results_file = f"{json_folder}/{experiment_type}_full_results_{job['n_train_triplets']}_{job['simulation_idx']}_{job['data_generating_model']}.json"
    best_results_file = f"{json_folder}/{experiment_type}_best_results_{job['n_train_triplets']}_{job['simulation_idx']}_{job['data_generating_model']}.json"
    
    with open(full_results_file, "w") as f:
        json.dump(full_results_dict, f, indent=2)
    
    with open(best_results_file, "w") as f:
        json.dump(best_results_dict, f, indent=2)
    
    print(f"  ✓ JSON files saved to: {json_folder}/")
    print(f"    - full_results: All regularization constants tested")
    print(f"    - best_results: Best-performing regularization per model")
    
    print("\n" + "="*80)
    print(f"JOB #{job['id']} COMPLETE - Ready for next job")
    print("="*80 + "\n")

if __name__ == "__main__":
    """
    Entry point for the simulation script.
    
    Runs the main function in a loop with error handling:
    1. Process pending jobs until time limit is reached (23.5 hours)
    2. Catch and log any exceptions during job processing
    3. Continue to next job even if current one fails
    
    The time limit ensures graceful completion within cluster job limits.
    """
    t0 = time.time()
    max_time = 3600 * 23.5  # 23.5 hours to allow safe shutdown
    job_count = 0
    
    print("\n" + "="*80)
    print("MODEL RECOVERY SIMULATION WORKER STARTED")
    print("="*80)
    print(f"Start time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time limit:     {max_time/3600:.1f} hours")
    print(f"Will stop at:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0 + max_time))}")
    print("="*80 + "\n")
    
    while (time.time() - t0) < max_time:
        try:
            job_count += 1
            main()
            
        except Exception as e:
            print("\n" + "!"*80)
            print("❌ ERROR OCCURRED DURING JOB EXECUTION")
            print("!"*80)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
            print("!"*80)
            print("⚠️  Continuing to next job...\n")
            continue
    
    # Time limit reached
    elapsed_time = time.time() - t0
    print("\n" + "="*80)
    print("WORKER SHUTDOWN - TIME LIMIT REACHED")
    print("="*80)
    print(f"Total runtime:       {elapsed_time/3600:.2f} hours")
    print(f"Jobs processed:      {job_count}")
    print(f"End time:            {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
        
        