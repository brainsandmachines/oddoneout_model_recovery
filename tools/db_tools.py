# experiment_tools.py

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from itertools import product
import time
import random
import numpy as np

def _current_time_str() -> str:
    """
    Returns a concise time string (day-month hour:minute), e.g. '04-01 13:47'.
    
    This function creates a formatted timestamp string used for tracking when jobs start and finish
    in the database. The format is deliberately kept concise for readability in database results.
    
    Returns:
        str: Formatted timestamp string in the format 'DD-MM HH:MM'
    """
    return datetime.now().strftime('%d-%m %H:%M')

def init_database(db_path: str) -> None:
    """
    Creates the jobs and reference_model_results tables if they do not exist.
    
    This function initializes the SQLite database with the necessary schema for tracking
    model recovery experiments. It creates two main tables:
    - 'jobs': Stores information about each simulation job including status and timestamps
    - 'reference_model_results': Stores performance metrics for each reference model tested
      within a job
      
    The function enforces uniqueness constraints to prevent duplicate job entries and sets up
    appropriate foreign key relationships between tables.
    
    Parameters:
        db_path (str): Path to the SQLite database file to initialize
        
    Notes:
        - If the tables already exist, they will not be modified
        - Jobs table has a UNIQUE constraint on (n_train_triplets, simulation_idx, 
          data_generating_model, constraint_type, dim_reduction_method)
        - Results table has foreign key references back to the jobs table
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        # Table for jobs
        c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            n_train_triplets INTEGER,
            simulation_idx INTEGER,
            data_generating_model TEXT,
            constraint_type TEXT,
            data_generating_model_regularization TEXT,
            candidate_model_regularization TEXT,
            dim_reduction_method TEXT,
            status TEXT,
            start_time TEXT,
            finish_time TEXT,
            UNIQUE (n_train_triplets, simulation_idx, data_generating_model, constraint_type,dim_reduction_method,data_generating_model_regularization,candidate_model_regularization)
        );
        """
        )

        # Table for storing reference model metrics
        c.execute("""
        CREATE TABLE IF NOT EXISTS reference_model_results (
            job_id INTEGER,
            simulation_idx INTEGER,
            data_generating_model TEXT,
            n_train_triplets INTEGER,
            reference_model TEXT,
            constraint_type TEXT,
            dim_reduction_method TEXT,
            data_generating_model_regularization TEXT,
            candidate_model_regularization TEXT,
            test_accuracy REAL,
            train_accuracy REAL,
            validation_accuracy REAL,
            test_nll REAL,
            train_nll REAL,
            validation_nll REAL,
            chosen_reg_con REAL,
            FOREIGN KEY(job_id) REFERENCES jobs(id),
            FOREIGN KEY(data_generating_model) REFERENCES jobs(data_generating_model),
            FOREIGN KEY(data_generating_model_regularization) REFERENCES jobs(data_generating_model_regularization),
            FOREIGN KEY(candidate_model_regularization) REFERENCES jobs(candidate_model_regularization),
            FOREIGN KEY(constraint_type) REFERENCES jobs(constraint_type),
            FOREIGN KEY(dim_reduction_method) REFERENCES jobs(dim_reduction_method)
        );
        """
        )
        
        #TODO: in the feature we might want to add noise ceiling estimation results. 
        # # Updated noise ceiling table
        # c.execute("""
        # CREATE TABLE IF NOT EXISTS noise_ceiling (
        #     id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     data_generating_model TEXT,
        #     simulation_idx INTEGER,
        #     n_train_triplets INTEGER,
        #     fold INTEGER,
        #     constraint_type TEXT,
        #     mean_upper_bound_acc REAL,
        #     mean_upper_bound_nll REAL,
        #     mean_lower_bound_acc REAL,
        #     mean_lower_bound_nll REAL,
        #     UNIQUE(data_generating_model, simulation_idx, n_train_triplets, fold, constraint_type)
        # );
        # """)

        # conn.commit()
        # conn.close()

def create_jobs_from_combinations(
    db_path: str,
    n_triplets_list: List[int],
    n_simulations: int,
    data_generating_models: List[str],
    constraints: Optional[List[str]] = None,
    dim_reduction_method: Optional[str] = None,
    data_generating_model_regularization: Optional[str] = None,
    candidate_model_regularization: Optional[str] = None
) -> None:
    """
    Create job entries in the database for all combinations of experimental parameters.
    
    This function generates all possible combinations of simulation parameters (n_triplets,
    simulation_idx, data_generating_model, constraint_type, dim_reduction_method) and inserts
    them into the jobs table. Each combination represents a single simulation job that needs
    to be run in the model recovery experiment.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        n_triplets_list (List[int]): List of training triplet counts to test
        n_simulations (int): Number of simulation runs for each parameter combination
        data_generating_models (List[str]): List of models to use as data generators
        constraints (Optional[List[str]]): List of constraint types to apply (default: ["None"])
        dim_reduction_method (Optional[str]): Dimensionality reduction method to apply (default: "None")
        data_generating_model_regularization (Optional[str]): Regularization method for data generating model (default: "None")
        candidate_model_regularization (Optional[str]): Regularization method for candidate model (default: "None")
    Notes:
        - Each inserted job has status='pending', ready to be picked up by a worker
        - The UNIQUE constraint in the database prevents duplicate jobs from being created
        - If a string is provided for constraints/dim_reduction_method, it's converted to a list
          for proper processing in the product function
    """
    if constraints is None:
        constraints = ["None"]
    if dim_reduction_method is None:
        dim_reduction_method = ["None"]
    if data_generating_model_regularization is None:
        data_generating_model_regularization = ["None"]
    if candidate_model_regularization is None:
        candidate_model_regularization = ["None"]
    # If constraints is None, use a list with a single None value
    if type(constraints) is not list:
        constraints = [constraints] #make sure constraints is in a list for the product function
    if type(dim_reduction_method) is not list:
        dim_reduction_method = [dim_reduction_method] #make sure dim_reduction_method is in a list for the product function
    if type(data_generating_model_regularization) is not list:
        data_generating_model_regularization = [data_generating_model_regularization] #make sure data_generating_model_regularization is in a list for the product function
    if type(candidate_model_regularization) is not list:
        candidate_model_regularization = [candidate_model_regularization] #make sure candidate_model_regularization is in a list for the product function
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        for sim_idx, n_tt, gen_model, constraint, dim_red_method, data_gen_reg, candidate_reg in product(
            range(n_simulations), n_triplets_list, data_generating_models, constraints, 
            dim_reduction_method, data_generating_model_regularization, candidate_model_regularization
        ):
            c.execute("""
                INSERT INTO jobs (
                    n_train_triplets, simulation_idx, data_generating_model, constraint_type, dim_reduction_method,
                    data_generating_model_regularization, candidate_model_regularization,
                    status, start_time, finish_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (n_tt, sim_idx, gen_model, constraint, dim_red_method, data_gen_reg, candidate_reg, 'pending', None, None))
            conn.commit()
        
        # conn.close()

def update_stuck_jobs(db_path):
    """
    Reset 'running' jobs that have been active for too long (likely stuck or crashed).
    
    This function identifies jobs that have been in 'running' status for more than 5 hours
    and resets them to 'pending' status, allowing them to be picked up and run again.
    This helps recover from situations where a job failed but wasn't properly marked as such
    in the database.
    
    Parameters:
        db_path (str): Path to the SQLite database file
    
    Notes:
        - Uses an exclusive transaction to ensure atomicity of the update
        - The time comparison is based on parsing the start_time field from its 'DD-MM HH:MM' format
        - Only affects jobs that have been running for more than 5 hours
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        # Use an exclusive transaction to update safely.
        conn.execute("BEGIN EXCLUSIVE")
        c = conn.cursor()
        c.execute("""
            UPDATE jobs
            SET status = 'pending', start_time = NULL
            WHERE status = 'running'
            AND start_time IS NOT NULL
            AND datetime(
                strftime('%Y', 'now') || '-' ||       -- Current year
                SUBSTR(start_time, 4, 2) || '-' ||    -- Month
                SUBSTR(start_time, 1, 2) || ' ' ||    -- Day
                SUBSTR(start_time, 7, 5)             -- Time (HH:MM)
                ) < datetime('now', '-5 hours');
        """)
        conn.commit()  # Commit to ensure changes are visible.
        
def get_new_job(db_path, no_heavy_lifting = False) -> Optional[Dict[str, Any]]:
    """
    Retrieve a pending job from the database and mark it as running.
    
    This function fetches the next available pending job from the database in FIFO order
    (lowest ID first), marks it as 'running', and returns its details. If no_heavy_lifting
    is True, only jobs with n_train_triplets < 100,000 will be considered, which allows
    for prioritizing lighter workloads on less powerful machines.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        no_heavy_lifting (bool): If True, only retrieve jobs with smaller triplet counts
                                (< 100,000) to avoid memory-intensive jobs. Default: False
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing job details if a pending job was found,
                                 including 'id', 'n_train_triplets', 'simulation_idx', etc.
                                 Returns None if no pending jobs are available.
    
    Notes:
        - Uses an exclusive transaction to ensure only one process can claim a job
        - Sets the job's status to 'running' and records the current time as start_time
        - Prioritizes jobs by ID (FIFO order) to ensure systematic processing
        - Backward compatible with older database schemas that don't have regularization columns
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        # Start an exclusive transaction.
        conn.execute("BEGIN EXCLUSIVE")
        c = conn.cursor()
        
        # Check if regularization columns exist in the jobs table
        c.execute("PRAGMA table_info(jobs)")
        columns_info = c.fetchall()
        column_names = [col[1] for col in columns_info]
        has_regularization = 'data_generating_model_regularization' in column_names and 'candidate_model_regularization' in column_names
        
        if has_regularization:
            # New schema with regularization columns
            if no_heavy_lifting:
                c.execute("""
                    SELECT id, n_train_triplets, simulation_idx, data_generating_model, constraint_type, dim_reduction_method,
                           data_generating_model_regularization, candidate_model_regularization
                    FROM jobs
                    WHERE status = 'pending' AND n_train_triplets < 100000
                    ORDER BY id
                    LIMIT 1
                """)  
            else:    
                c.execute("""
                    SELECT id, n_train_triplets, simulation_idx, data_generating_model, constraint_type, dim_reduction_method,
                           data_generating_model_regularization, candidate_model_regularization
                    FROM jobs
                    WHERE status = 'pending'
                    ORDER BY id
                    LIMIT 1
                """)
            row = c.fetchone()
            if row is None:
                conn.commit()
                return None
            
            job_id, n_tt, sim_idx, gen_model, constraint, dim_reduction_method, data_gen_reg, candidate_reg = row
            
        else:
            # Old schema without regularization columns
            if no_heavy_lifting:
                c.execute("""
                    SELECT id, n_train_triplets, simulation_idx, data_generating_model, constraint_type, dim_reduction_method
                    FROM jobs
                    WHERE status = 'pending' AND n_train_triplets < 100000
                    ORDER BY id
                    LIMIT 1
                """)  
            else:    
                c.execute("""
                    SELECT id, n_train_triplets, simulation_idx, data_generating_model, constraint_type, dim_reduction_method
                    FROM jobs
                    WHERE status = 'pending'
                    ORDER BY id
                    LIMIT 1
                """)
            row = c.fetchone()
            if row is None:
                conn.commit()
                return None
            
            job_id, n_tt, sim_idx, gen_model, constraint, dim_reduction_method = row
            # Set default values for regularization parameters
            data_gen_reg, candidate_reg = "None", "None"
        
        # Mark the fetched job as running.
        c.execute("""
            UPDATE jobs
            SET status = 'running', start_time = ?
            WHERE id = ?
        """, (_current_time_str(), job_id))
        conn.commit()  # Commit the change.
        
        return {
            "id": job_id,
            "n_train_triplets": n_tt,
            "simulation_idx": sim_idx,
            "data_generating_model": gen_model,
            "constraint_type": constraint,
            "dim_reduction_method": dim_reduction_method,
            "data_generating_model_regularization": data_gen_reg,
            "candidate_model_regularization": candidate_reg
        }


def get_pending_job(db_path: str, max_retries: int = 10, retry_delay: float = 1.0,no_heavy_lifting = False) -> Optional[Dict[str, Any]]:
    """
    Fetch a pending job from the database with retry logic for handling database locks.
    
    This function is a wrapper around get_new_job that adds retry logic to handle database
    locking issues. It first checks for and resets any stuck jobs, then attempts to retrieve
    a pending job, retrying if the database is locked by another process.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        max_retries (int): Maximum number of retry attempts if the database is locked. Default: 10
        retry_delay (float): Base delay in seconds between retry attempts. Default: 1.0
        no_heavy_lifting (bool): If True, only retrieve jobs with n_train_triplets < 100,000.
                                Default: False
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing job details if a pending job was found,
                                or None if no pending jobs are available or max retries exceeded
    
    Notes:
        - Adds random jitter to retry delays to prevent synchronized retries
        - Exponentially backs off on retries to reduce database contention
        - Calls update_stuck_jobs before fetching to ensure stalled jobs are reset
    """
    for attempt in range(max_retries):
        try:
            # Open a new connection for each attempt.
            update_stuck_jobs(db_path)
            return get_new_job(db_path,no_heavy_lifting)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                jitter = random.uniform(0, 0.5)
                actual_delay = retry_delay + jitter
                print(f"Database locked. Retrying in {actual_delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(actual_delay)
            else:
                raise e
    print("Failed to fetch a pending job after multiple retries.")
    return None


def get_best_models_per_generating_model(
    db_path: str,
    n_train_triplets: int,
    constraint_type: str
) -> Dict[str, Dict[str, List[str]]]:
    """
    Get the best-performing reference models for each data-generating model.
    
    For each data-generating model, this function identifies which reference models achieved:
    1. The highest test accuracy
    2. The lowest test negative log-likelihood (NLL)
    across all completed simulations with the specified training triplet count and constraint type.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        n_train_triplets (int): Number of training triplets to filter by
        constraint_type (str): Constraint type to filter by
    
    Returns:
        Dict[str, Dict[str, List[str]]]: A nested dictionary where:
            - Outer keys are data-generating model names
            - Inner dictionary has keys 'best_accuracy' and 'best_nll'
            - Values are lists of reference model names that performed best in each simulation
              
    Example:
        {
            'ResNet50': {
                'best_accuracy': ['ViT_Large', 'ResNet50', 'ViT_Large'],  # Best models for 3 simulations
                'best_nll': ['ResNet50', 'ResNet50', 'ResNet50']
            },
            'ViT_Large': {
                'best_accuracy': [...],
                'best_nll': [...]
            }
        }
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()

        # First get all unique data generating models
        c.execute("""
            SELECT DISTINCT data_generating_model 
            FROM jobs 
            WHERE n_train_triplets = ?
            AND constraint_type = ?
        """, (n_train_triplets, constraint_type))
        
        data_generating_models = [row[0] for row in c.fetchall()]
        results = {}

        for gen_model in data_generating_models:
            results[gen_model] = {
                'best_accuracy': [],
                'best_nll': []
            }
            
            # Get all simulation IDs for this generating model and n_train_triplets
            c.execute("""
                SELECT id, simulation_idx 
                FROM jobs 
                WHERE data_generating_model = ? 
                AND n_train_triplets = ?
                AND constraint_type = ?
                AND status = 'done'
                ORDER BY simulation_idx
            """, (gen_model, n_train_triplets, constraint_type))
            
            job_ids = c.fetchall()
            
            for job_id, _ in job_ids:
                # Get reference model with highest test accuracy
                c.execute("""
                    SELECT reference_model, test_accuracy 
                    FROM reference_model_results 
                    WHERE job_id = ? 
                    ORDER BY test_accuracy DESC 
                    LIMIT 1
                """, (job_id,))
                best_accuracy_model = c.fetchone()
                
                # Get reference model with lowest test NLL
                c.execute("""
                    SELECT reference_model, test_nll 
                    FROM reference_model_results 
                    WHERE job_id = ? 
                    ORDER BY test_nll ASC 
                    LIMIT 1
                """, (job_id,))
                best_nll_model = c.fetchone()
                
                if best_accuracy_model:
                    results[gen_model]['best_accuracy'].append(best_accuracy_model[0])
                if best_nll_model:
                    results[gen_model]['best_nll'].append(best_nll_model[0])

        # conn.close()
        return results

def get_test_accuracies_per_model(
    db_path: str,
    n_triplets_list: List[int],
    constraint_type: str
) -> Dict[str, Dict[int, List[float]]]:
    """
    Get test accuracies for each model when predicting its own generated data.
    
    This function retrieves test accuracies for all cases where a model was used as both the
    data-generating model and the reference model (self-recognition). Results are organized
    by model and triplet count, collecting accuracies across all simulations.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        n_triplets_list (List[int]): List of training triplet counts to include
        constraint_type (str): Constraint type to filter by
    
    Returns:
        Dict[str, Dict[int, List[float]]]: A nested dictionary where:
            - Outer keys are model names
            - Inner keys are triplet counts
            - Values are lists of test accuracy values from different simulations
              
    Example:
        {
            'ResNet50': {
                1000: [0.78, 0.81, 0.79],  # Accuracies from 3 simulations with 1000 triplets
                10000: [0.85, 0.86, 0.84]
            },
            'ViT_Large': {
                1000: [0.76, 0.75, 0.77],
                10000: [0.84, 0.83, 0.85]
            }
        }
    
    Notes:
        - Only includes results from completed ('done') jobs
        - Only includes results where reference_model matches data_generating_model
        - Models with no matching results for a triplet count will not have an entry for that count
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        
        # Get all unique data generating models
        c.execute("SELECT DISTINCT data_generating_model FROM jobs")
        models = [row[0] for row in c.fetchall()]
        
        results = {model: {} for model in models}
        
        for model in models:
            for n_triplets in n_triplets_list:
                # Get test accuracies where the reference model matches the data generating model
                c.execute("""
                    SELECT r.test_accuracy
                    FROM jobs j
                    JOIN reference_model_results r ON j.id = r.job_id
                    WHERE j.data_generating_model = ?
                    AND j.n_train_triplets = ?
                    AND r.reference_model = j.data_generating_model
                    AND j.constraint_type = ?
                    AND j.status = 'done'
                """, (model, n_triplets, constraint_type))
                
                accuracies = [row[0] for row in c.fetchall()]
                if accuracies:  # Only add if we have results
                    results[model][n_triplets] = accuracies
        
        # conn.close()
        return results

def get_test_nlls_per_model(
    db_path: str,
    n_triplets_list: List[int],
    constraint_type: str
) -> Dict[str, Dict[int, List[float]]]:
    """
    Get test negative log-likelihood (NLL) values for each model when predicting its own generated data.
    
    This function retrieves test NLL values for all cases where a model was used as both the
    data-generating model and the reference model (self-recognition). Results are organized
    by model and triplet count, collecting NLL values across all simulations.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        n_triplets_list (List[int]): List of training triplet counts to include
        constraint_type (str): Constraint type to filter by
    
    Returns:
        Dict[str, Dict[int, List[float]]]: A nested dictionary where:
            - Outer keys are model names
            - Inner keys are triplet counts
            - Values are lists of test NLL values from different simulations
              
    Example:
        {
            'ResNet50': {
                1000: [0.55, 0.51, 0.53],  # NLL values from 3 simulations with 1000 triplets
                10000: [0.35, 0.36, 0.37]
            },
            'ViT_Large': {
                1000: [0.57, 0.58, 0.56],
                10000: [0.37, 0.38, 0.36]
            }
        }
    
    Notes:
        - Lower NLL values indicate better performance
        - Only includes results from completed ('done') jobs
        - Only includes results where reference_model matches data_generating_model
        - Models with no matching results for a triplet count will not have an entry for that count
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        
        # Get all unique data generating models
        c.execute("SELECT DISTINCT data_generating_model FROM jobs")
        models = [row[0] for row in c.fetchall()]
        
        results = {model: {} for model in models}
        
        for model in models:
            for n_triplets in n_triplets_list:
                # Get test NLLs where the reference model matches the data generating model
                c.execute("""
                    SELECT r.test_nll
                    FROM jobs j
                    JOIN reference_model_results r ON j.id = r.job_id
                    WHERE j.data_generating_model = ?
                    AND j.n_train_triplets = ?
                    AND r.reference_model = j.data_generating_model
                    AND j.constraint_type = ?
                    AND j.status = 'done'
                """, (model, n_triplets, constraint_type))
                
                nlls = [row[0] for row in c.fetchall()]
                if nlls:  # Only add if we have results
                    results[model][n_triplets] = nlls
  
        # conn.close()
        return results

def save_noise_ceiling_results(
    db_path: str,
    data_generating_model: str,
    simulation_idx: int,
    n_train_triplets: int,
    constraint_type: str,
    folds_noise_ceiling_dict: dict,
    max_retries: int = 5,
    retry_delay: float = 1.0
) -> None:
    """
    Save noise ceiling estimation results for each fold to the database.
    
    This function stores the estimated noise ceiling values (upper and lower bounds for
    accuracy and NLL metrics) for each fold in a simulation. It includes retry logic to
    handle database locks.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        data_generating_model (str): Name of the data generating model
        simulation_idx (int): Simulation index 
        n_train_triplets (int): Number of training triplets used
        constraint_type (str): Constraint type used
        folds_noise_ceiling_dict (dict): Dictionary containing noise ceiling results for each fold
        max_retries (int): Maximum number of retry attempts if database is locked. Default: 5
        retry_delay (float): Delay in seconds between retry attempts. Default: 1.0
        
    Notes:
        - Requires the 'noise_ceiling' table to exist in the database
        - For each fold, inserts upper and lower bounds for both accuracy and NLL metrics
        - Implements retry logic to handle database locks
    """
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path,timeout=20)
            c = conn.cursor()

            for fold, results in folds_noise_ceiling_dict.items():
                c.execute("""
                    INSERT INTO noise_ceiling (
                        data_generating_model,
                        simulation_idx,
                        n_train_triplets,
                        fold,
                        mean_upper_bound_acc,
                        mean_upper_bound_nll,
                        mean_lower_bound_acc,
                        mean_lower_bound_nll
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_generating_model,
                    simulation_idx,
                    n_train_triplets,
                    fold,
                    constraint_type,
                    results["mean_upper_bound_acc"],
                    results["mean_upper_bound_nll"],
                    results["mean_lower_bound_acc"],
                    results["mean_lower_bound_nll"]
                ))

            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
        finally:
            if 'conn' in locals():
                conn.close()
    print("Failed to save noise ceiling results after multiple retries.")
    
def get_confusion_table_name(base: str, n_train_triplets: int) -> str:
    """
    Generate a standard table name for confusion matrix tables.
    
    This helper function creates a consistent naming pattern for tables storing
    confusion matrix data for different numbers of training triplets.
    
    Parameters:
        base (str): Base name for the confusion table (e.g., 'confusion_accuracy')
        n_train_triplets (int): Number of training triplets
        
    Returns:
        str: The full table name, formatted as '{base}_{n_train_triplets}'
    """
    return f"{base}_{n_train_triplets}"

def create_confusion_pivot_table(db_path: str, base: str, n_train_triplets: int) -> None:
    """
    Create a pivot table to store confusion matrix data for model recovery experiments.
    
    This function creates a table with a row for each data-generating model and columns
    for each reference model. The table is designed to store counts of how many times each
    reference model was identified as the best match for a data-generating model.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        base (str): Base name for the confusion table (e.g., 'confusion_accuracy' or 'confusion_nll')
        n_train_triplets (int): Number of training triplets for this confusion matrix
        
    Notes:
        - The table is dynamically constructed based on all distinct reference models in the database
        - Each reference model gets its own column to store counts
        - The table will be created only if it doesn't already exist
        - Column names are quoted to handle special characters in model names
    """
    table_name = f"{base}_{n_train_triplets}"
    ref_models = get_distinct_reference_models(db_path)  # dynamic list
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        
    # Build the SQL for the table creation.
        columns = ["data_generating_model TEXT PRIMARY KEY"]
        for ref in ref_models:
            # Use quotes in case model names have special characters.
            columns.append(f'"{ref}" INTEGER DEFAULT 0')
        columns_sql = ", ".join(columns)
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        c.execute(create_sql)
        conn.commit()
        # conn.close()
        
def add_column_if_not_exists(db_path: str, table_name: str, column: str) -> None:
    """
    Add a column to a table if it doesn't already exist.
    
    This helper function checks if a column exists in a specified table and adds it
    if it doesn't. It's used to ensure that confusion matrix tables have columns for
    all reference models, even those added after table creation.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        table_name (str): Name of the table to modify
        column (str): Name of the column to add if it doesn't exist
        
    Notes:
        - New columns are added with INTEGER type and DEFAULT 0 value
        - Column names are quoted to handle special characters
        - This function is especially useful when new reference models are added to experiments
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        # Check the existing columns in the table.
        c.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in c.fetchall()]
        if column not in columns:
            alter_sql = f'ALTER TABLE {table_name} ADD COLUMN "{column}" INTEGER DEFAULT 0'
            c.execute(alter_sql)
            conn.commit()
        #conn.close()


def update_confusion_pivot_table(db_path: str, base: str, n_train_triplets: int,
                                 data_generating_model: str, winning_ref: str) -> None:
    """
    Update a confusion matrix pivot table by incrementing the count for a winning reference model.
    
    This function increments the count in a confusion matrix table for a specific data-generating
    model and reference model pair. If a row doesn't exist for the data-generating model, it's
    created with zeros for all reference models before incrementing the winning reference model.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        base (str): Base name for the confusion table (e.g., 'confusion_accuracy')
        n_train_triplets (int): Number of training triplets for this confusion matrix
        data_generating_model (str): The data-generating model name (row in the matrix)
        winning_ref (str): The winning reference model name (column to increment)
        
    Notes:
        - Automatically ensures the column exists by calling add_column_if_not_exists
        - Creates a new row with zeros if the data-generating model doesn't have a row yet
        - Increments the count for the winning reference model by 1
        - Uses quoted column names to handle special characters in model names
    """
    table_name = f"{base}_{n_train_triplets}"
    # Ensure the column for the winning reference model exists.
    add_column_if_not_exists(db_path, table_name, winning_ref)
    
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        # Check if a row for this data generating model exists.
        c.execute(f"SELECT * FROM {table_name} WHERE data_generating_model = ?", (data_generating_model,))
        row = c.fetchone()
        if not row:
            # Insert a new row. Get the list of columns (except the primary key) from the table.
            c.execute(f"PRAGMA table_info({table_name})")
            info = c.fetchall()
            # Column names starting from second column.
            ref_columns = [col[1] for col in info if col[1] != "data_generating_model"]
            placeholders = ", ".join(["?"] * (1 + len(ref_columns)))
            # Initialize counts to zero.
            values = [data_generating_model] + [0] * len(ref_columns)
            columns_str = ', '.join('"' + col + '"' for col in ref_columns)
            insert_sql = f"INSERT INTO {table_name} (data_generating_model, {columns_str}) VALUES ({placeholders})"

            c.execute(insert_sql, values)
            conn.commit()
        
        # Now update the winning reference model cell.
        update_sql = f"""UPDATE {table_name} 
                        SET "{winning_ref}" = "{winning_ref}" + 1 
                        WHERE data_generating_model = ?"""
        c.execute(update_sql, (data_generating_model,))
        conn.commit()
        # conn.close()


def get_distinct_reference_models(db_path: str) -> List[str]:
    """
    Get a list of all distinct reference model names from the database.
    
    This function queries the database to find all unique reference model names that have
    been used in experiments, which is useful for creating confusion matrix tables and
    other analyses that need to consider all models.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        
    Returns:
        List[str]: List of all distinct reference model names
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT reference_model FROM reference_model_results")
        models = [row[0] for row in c.fetchall()]
        # conn.close()
        return models

def get_distinct_data_generating_models(db_path: str) -> List[str]:
    """
    Get a list of all distinct data-generating model names from the database.
    
    This function queries the database to find all unique data-generating model names
    that have been used in experiments, which is useful for creating confusion matrix 
    tables and other analyses.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        
    Returns:
        List[str]: List of all distinct data-generating model names
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT data_generating_model FROM jobs")
        models = [row[0] for row in c.fetchall()]
        # conn.close()
        return models

def update_confusion_matrices(db_path: str, job_id: int) -> None:
    """
    Update confusion matrix tables with results from a completed job.
    
    This function identifies the winning reference models for a job (based on highest
    test accuracy and lowest test NLL) and updates the corresponding confusion matrix
    tables. It ensures the tables exist and have appropriate columns before updating.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        job_id (int): ID of the completed job to process
        
    Notes:
        - Creates/updates two separate confusion matrices:
          1. confusion_accuracy_* for highest test accuracy
          2. confusion_nll_* for lowest test NLL
        - Each matrix records how often each reference model wins for each data-generating model
        - Automatically creates tables and columns as needed
    """
    with sqlite3.connect(db_path, timeout=20) as conn:
        c = conn.cursor()
        # Retrieve job info.
        c.execute("SELECT n_train_triplets, data_generating_model FROM jobs WHERE id = ?", (job_id,))
        job_row = c.fetchone()
        if not job_row:
            return
        n_train_triplets, data_generating_model = job_row
        
        # Retrieve reference model results.
        c.execute("""
            SELECT reference_model, test_accuracy, test_nll
            FROM reference_model_results
            WHERE job_id = ?
        """, (job_id,))
        results = c.fetchall()
        if not results:
            return
        
    # Determine winners.
    winner_accuracy = max(results, key=lambda x: x[1])[0]  # Highest test_accuracy
    winner_nll = min(results, key=lambda x: x[2])[0]         # Lowest test_nll
    
    # Define the base names for the pivot tables.
    base_accuracy = "confusion_accuracy"
    base_nll = "confusion_nll"
    
    # Create (if needed) the pivot tables for both accuracy and NLL.
    create_confusion_pivot_table(db_path, base_accuracy, n_train_triplets)
    create_confusion_pivot_table(db_path, base_nll, n_train_triplets)
    
    # Update the pivot tables.
    update_confusion_pivot_table(db_path, base_accuracy, n_train_triplets, data_generating_model, winner_accuracy)
    update_confusion_pivot_table(db_path, base_nll, n_train_triplets, data_generating_model, winner_nll)

def save_reference_model_results(
    db_path: str,
    job_id: int,
    reference_models_list: List[Dict[str, Any]],
    max_retries: int = 5,
    retry_delay: float = float(np.random.uniform(0,3))
) -> None:
    """
    Save performance results for all reference models of a job and mark the job as 'done'.
    
    This function saves performance metrics (accuracy, NLL, etc.) for each reference model
    tested in a job and updates the job's status to 'done'. It includes retry logic to handle
    database locks and automatically updates confusion matrices with the results.
    
    Parameters:
        db_path (str): Path to the SQLite database file
        job_id (int): ID of the job to save results for
        reference_models_list (List[Dict[str, Any]]): List of dictionaries, each containing metrics
                                                    for one reference model
        max_retries (int): Maximum number of retry attempts if database is locked. Default: 5
        retry_delay (float): Random delay between 0-3 seconds between retry attempts. Default: random
        
    Notes:
        - Each reference model dictionary should contain keys like 'name', 'test_accuracy', etc.
        - The function marks the job as 'done' and records the current time as finish_time
        - After saving reference model results, it calls update_confusion_matrices to maintain
          the confusion matrix tables
        - Implements randomized retry logic to handle database locks
        - Backward compatible with old database schemas that don't have regularization columns
    """
    for attempt in range(max_retries):
        try:
            with sqlite3.connect(db_path,timeout=20) as conn:    
                c = conn.cursor()
                
                # Check if regularization columns exist in reference_model_results table
                c.execute("PRAGMA table_info(reference_model_results)")
                columns_info = c.fetchall()
                column_names = [col[1] for col in columns_info]
                has_regularization = 'data_generating_model_regularization' in column_names and 'candidate_model_regularization' in column_names
                has_n_train_triplets = 'n_train_triplets' in column_names
                has_simulation_idx = 'simulation_idx' in column_names
                
                # Get job information including all relevant fields
                if has_regularization:
                    if has_n_train_triplets and has_simulation_idx:
                        # Newest schema with regularization, n_train_triplets, and simulation_idx
                        c.execute("""
                            SELECT data_generating_model, data_generating_model_regularization, candidate_model_regularization, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, data_gen_reg, candidate_reg, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                        else:
                            raise ValueError(f"Job {job_id} not found")
                    elif has_n_train_triplets:
                        # Schema with regularization and n_train_triplets but no simulation_idx
                        c.execute("""
                            SELECT data_generating_model, data_generating_model_regularization, candidate_model_regularization, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, data_gen_reg, candidate_reg, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                        else:
                            raise ValueError(f"Job {job_id} not found")
                    else:
                        # Schema with regularization but no n_train_triplets or simulation_idx
                        c.execute("""
                            SELECT data_generating_model, data_generating_model_regularization, candidate_model_regularization, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, data_gen_reg, candidate_reg, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                        else:
                            raise ValueError(f"Job {job_id} not found")
                else:
                    if has_n_train_triplets and has_simulation_idx:
                        # Schema with n_train_triplets and simulation_idx but no regularization
                        c.execute("""
                            SELECT data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                            data_gen_reg, candidate_reg = None, None
                        else:
                            raise ValueError(f"Job {job_id} not found")
                    elif has_n_train_triplets:
                        # Schema with n_train_triplets but no regularization or simulation_idx
                        c.execute("""
                            SELECT data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                            data_gen_reg, candidate_reg = None, None
                        else:
                            raise ValueError(f"Job {job_id} not found")
                    elif has_simulation_idx:
                        # Schema with simulation_idx but no regularization or n_train_triplets
                        c.execute("""
                            SELECT data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                            data_gen_reg, candidate_reg = None, None
                        else:
                            raise ValueError(f"Job {job_id} not found")
                    else:
                        # Oldest schema without regularization, n_train_triplets, or simulation_idx
                        c.execute("""
                            SELECT data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx
                            FROM jobs WHERE id = ?
                        """, (job_id,))
                        job_row = c.fetchone()
                        if job_row:
                            data_generating_model, constraint_type, dim_reduction_method, n_train_triplets, simulation_idx = job_row
                            data_gen_reg, candidate_reg = None, None
                        else:
                            raise ValueError(f"Job {job_id} not found")
                
                for rm in reference_models_list:
                    if has_regularization and has_n_train_triplets and has_simulation_idx:
                        # Newest schema with all columns
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                simulation_idx,
                                data_generating_model,
                                n_train_triplets,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                data_generating_model_regularization,
                                candidate_model_regularization,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            simulation_idx,
                            data_generating_model,
                            n_train_triplets,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            data_gen_reg,
                            candidate_reg,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    elif has_regularization and has_n_train_triplets:
                        # Schema with regularization and n_train_triplets but no simulation_idx
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                data_generating_model,
                                n_train_triplets,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                data_generating_model_regularization,
                                candidate_model_regularization,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            data_generating_model,
                            n_train_triplets,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            data_gen_reg,
                            candidate_reg,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    elif has_regularization:
                        # Schema with regularization but no n_train_triplets or simulation_idx
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                data_generating_model,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                data_generating_model_regularization,
                                candidate_model_regularization,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            data_generating_model,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            data_gen_reg,
                            candidate_reg,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    elif has_n_train_triplets and has_simulation_idx:
                        # Schema with n_train_triplets and simulation_idx but no regularization
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                simulation_idx,
                                data_generating_model,
                                n_train_triplets,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            simulation_idx,
                            data_generating_model,
                            n_train_triplets,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    elif has_n_train_triplets:
                        # Schema with n_train_triplets but no regularization or simulation_idx
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                data_generating_model,
                                n_train_triplets,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            data_generating_model,
                            n_train_triplets,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    elif has_simulation_idx:
                        # Schema with simulation_idx but no regularization or n_train_triplets
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                simulation_idx,
                                data_generating_model,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            simulation_idx,
                            data_generating_model,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                    else:
                        # Oldest schema without regularization, n_train_triplets, or simulation_idx columns
                        c.execute("""
                            INSERT INTO reference_model_results (
                                job_id,
                                reference_model,
                                constraint_type,
                                dim_reduction_method,
                                test_accuracy,
                                train_accuracy,
                                validation_accuracy,
                                test_nll,
                                train_nll,
                                validation_nll,
                                chosen_reg_con
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            job_id,
                            rm["name"],
                            constraint_type,
                            dim_reduction_method,
                            rm.get("test_accuracy"),
                            rm.get("train_accuracy"),
                            rm.get("validation_accuracy"),
                            rm.get("test_nll"),
                            rm.get("train_nll"),
                            rm.get("validation_nll"),
                            rm.get("chosen_reg_con")
                        ))
                # Mark job as done
                c.execute("""
                    UPDATE jobs
                    SET status = 'done', finish_time = ?
                    WHERE id = ?
                """, (_current_time_str(), job_id))
                conn.commit()
            return  # Successfully saved results
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e  # Raise other errors
    print("Failed to save results after multiple retries.")

# if __name__ == "__main__":
#     init_database("test.db")
#     n_triplets_list = [100]
#     n_simulations = 3
#     data_generating_models = ["modelA", "modelB"]
#     create_jobs_from_combinations("test.db", n_triplets_list, n_simulations, data_generating_models)
#     for i in range(3):
#         job = get_pending_job("test.db")
#         print(job)
#         res = [
#             {
#                 "name": "ref_model_1",
#                 "test_accuracy": 0.90,
#                 "train_accuracy": 0.92,
#                 "validation_accuracy": 0.89,
#                 "test_nll": 0.10,
#                 "train_nll": 0.09,
#                 "validation_nll": 0.11,
#                 "chosen_temp": 1.0,
#                 "chosen_reg_con": 0.01
#             },
#             {
#                 "name": "ref_model_2",
#                 "test_accuracy": 0.88+i,
#                 "train_accuracy": 0.90+i,
#                 "validation_accuracy": 0.86+i,
#                 "test_nll": 0.20+i,
#                 "train_nll": 0.18+i,
#                 "validation_nll": 0.22+i,
#                 "chosen_temp": 0.8+i,
#                 "chosen_reg_con": 0.05
#             }
#             # etc.
#         ]
#         save_reference_model_results("test.db", job["id"], res)