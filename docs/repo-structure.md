# Repository Structure

This document provides a detailed map of the repository's file system organization.

---

## Top-Level Structure

```
.
├── README.md                          # Main repository documentation (landing page)
├── INSTRUCTIONS.md                    # Detailed setup and workflow guide
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment specification (minimal)
├── .gitignore                        # Git ignore patterns
│
├── docs/                             # 📚 Documentation (NEW)
│   ├── pipeline_map.md              # Pipeline flow and dependencies
│   └── repo-structure.md            # This file
│
├── Data/                             # 💾 Input data and pre-trained models
│   ├── Things_data/                 # Original THINGS dataset (download separately)
│   ├── Things_data_preprocessed/    # Preprocessed triplets and human choices
│   ├── models_features/             # Pre-extracted neural network features
│   ├── models_data_generating/      # Fitted CogModels (.pt files - saved PyTorch modules)
│   ├── models_data_generating_L1/   # Alternative regularization experiments
│   ├── models_data_generating_L2/   # Alternative regularization experiments
│   └── models_n_params_table/       # Model architecture metadata
│
├── Results/                          # 📊 Experimental results (GENERATED)
│   ├── Simulations_experiments/     # SQLite databases with simulation results
│   ├── models_accuracy_analysis_THINGS_accuracy/  # THINGS evaluation results
│   ├── rsa/                         # RSA similarity matrices
│   ├── regression_analysis/         # Statistical analysis outputs
│   ├── effective_dimensionality_score/  # Pre-computed dimensionality metrics
│   └── [various analysis folders]   # Other analysis outputs
│
├── Plots/                            # 📈 Publication figures (GENERATED)
│   ├── full_W/                      # Figures for full_W experiment
│   ├── Figure_*.pdf                 # Main and supplementary figures
│   └── [experiment-specific folders]
│
├── experiment_scripts/               # 🔬 Main pipeline scripts
│   ├── create_data_generating_models.py   # Stage 1: Fit models to THINGS data
│   ├── create_simulations_db.py           # Stage 2: Create simulation database
│   └── run_simulations.py                 # Stage 3: Execute model recovery experiments
│
├── figures_scripts/                  # 📊 Figure generation scripts
│   ├── create_Figures_1_and_3.py    # Main model recovery figures
│   ├── create_Figure_2.py           # THINGS accuracy figure
│   ├── create_Figure_4_MDS.py       # MDS projection of model similarities
│   ├── create_Figure_5_tradeoff.py  # Flexibility-accuracy tradeoff
│   ├── create_Figure_S1_violin_plot.py  # Supplementary: accuracy distributions
│   ├── create_Figure_S4_RSA.py      # Supplementary: RSA analysis
│   └── create_Figure_S5_S4.py       # Supplementary: combined analysis
│
├── analysis_scripts/                  # 🔍 Analysis utilities & preprocessing
│   ├── odd_one_out_text_preprocess.py     # Stage 0A: CSV → PyTorch tensor conversion
│   ├── things_noise_ceiling_eval.py       # Stage 0B: Noise ceiling calculation (test set 2: ~67.8%)
│   ├── evaluate_THINGS_OOO_accuracy_CV.py  # Evaluate models on real THINGS data (essential for Figure 2)
│   ├── save_transformed_features.py        # Apply W transformations to features
│   ├── build_rsa_sim_matrix.py            # Compute RSA similarity matrix
│   ├── create_models_properties_table.py   # Extract model architecture info
│   ├── Table_S3_regression.py             # Regression analysis for Table S3
│   └── figure_5_analysis.py               # Helper for Figure 5 generation
│
├── scripts_configurations/           # ⚙️ Hydra configuration files (YAML)
│   ├── shared_args.yaml             # Base configuration (model lists, paths)
│   ├── odd_one_out_preprocess.yaml  # Config for Stage 0A: CSV preprocessing
│   ├── things_noise_ceiling_eval.yaml  # Config for Stage 0B: Noise ceiling
│   ├── create_data_generating_models.yaml
│   ├── create_simulations_db.yaml
│   ├── run_simulations.yaml
│   ├── figures_1_and_3.yaml
│   ├── Figure_5.yaml
│   ├── RSA_MDS_figures.yaml
│   ├── evaluate_THINGS_OOO_accuracy_CV.yaml
│   └── [other experiment configs]
│
├── tools/                            # 🛠️ Core library modules
│   ├── __init__.py                  # (empty)
│   ├── cog_model.py                 # CogModel class (main model implementation)
│   ├── model_recovery_fn.py         # Model recovery experiment functions
│   ├── db_tools.py                  # Database utilities (create, query, update)
│   ├── db_analysis.py               # Analysis and visualization classes
│   ├── model_io.py                  # Save/load model utilities
│   ├── triplets_data_set.py         # Dataset classes for triplet data
│   ├── similarity_functions.py      # Similarity metrics (dot product, cosine, etc.)
│   ├── losses.py                    # Loss functions for triplet learning
│   ├── reg_funcs.py                 # Regularization functions (L1, L2, eye_distance)
│   ├── constraints.py               # Constraint parametrizations (orthogonal, diagonal)
│   ├── utils.py                     # General utilities (optimizers, triplet generation)
│   ├── plots_scripts.py             # Plotting utilities
│   ├── prepare_dfs_for_figure_2.py  # Data preparation for Figure 2
│   ├── metroplot.py                 # Custom plotting utilities
│   └──  s_triplet_test.py            # Statistical testing utilities
│   
│

├── visualizer.py                     # ⚠️ LEGACY - not used (flagged for removal)
├── temp.py                           # 🔧 Utility: filter database by regularization
└── The paper.pdf                     # 📄 Accepted NeurIPS 2025 paper
```

---

### 💾 `Data/` - Input Data and Models 



**Purpose**: Stores all input data and pre-trained models.

#### `Data/Things_data/` 

Original THINGS odd-one-out dataset:
- Images organized by concept
- Triplet metadata
- Human behavioral responses

**Download**: 
- THINGS images: [THINGS project website](https://osf.io/jum2f/)
- Odd-one-out behavioral data: [THINGS Odd-One-Out Dataset](https://osf.io/f5rn6/) (data/triplet_dataset/)

#### `Data/Things_data_preprocessed/`
**Status**: GENERATED by Stage 0 preprocessing scripts

Preprocessed THINGS data ready for experiments:

**Generated by `analysis_scripts/odd_one_out_text_preprocess.py` (Stage 0A)**:
- `Things_odd_one_out_triplets_train_set.pt` - Training triplets (0-based indexing)
- `Things_odd_one_out_answers_positions_train_set.pt` - Training responses (0/1/2)
- `Things_odd_one_out_triplets_testset1_noise_ceiling.pt` - Test set 1
- `Things_odd_one_out_triplets_testset2_noise_ceiling.pt` - Test set 2 (PRIMARY)
- `Things_odd_one_out_triplets_testset3_noise_ceiling.pt` - Test set 3
- `Things_odd_one_out_answers_positions_testset*_noise_ceiling.pt` - Test responses
- `things_datasets_info.csv` - Dataset statistics summary

**Generated by `analysis_scripts/things_noise_ceiling_eval.py` (Stage 0B)**:
- `unique_triplets_and_counts/test_2_unique_triplets.pt` - Unique triplets (test set 2)
- `unique_triplets_and_counts/test_2_answers_counts.pt` - Response counts per triplet
- `unique_triplets_and_counts/analysis_noise_ceiling.csv` - Noise ceiling: ~67.8% for test set 2 ⭐

**Note**: Test set 2 is the PRIMARY validation set used throughout the paper.

**Configuration**: 
- `scripts_configurations/odd_one_out_preprocess.yaml`
- `scripts_configurations/things_noise_ceiling_eval.yaml`

#### `Data/models_features/`
**Status**: USER-PROVIDED (extract features using external tools)

Pre-extracted neural network features for 1854 THINGS concepts:

```
models_features/
├── full/                    # Full-dimensional features
│   ├── ResNet50.pt         # [1854, 2048]
│   ├── ViT_Large.pt        # [1854, 1024]
│   ├── CLIP_RN50.pt        # [1854, 1024]
│   └── [20-30 model files]
└── PCA_500/                # PCA-reduced to 500 dimensions
    ├── ResNet50_PCA_500.pt # [1854, 500]
    └── [model files]
```

**Size**: ~2-5GB per model, ~50-100GB total

**Extraction**: Use model-specific feature extraction scripts (PyTorch, timm, transformers)

#### `Data/models_data_generating/`
**Status**: GENERATED by `create_data_generating_models.py` OR DOWNLOAD from Figshare

Fitted CogModel instances (ground truth models for simulations):

```
models_data_generating/
├── full_W/                  # Unconstrained W matrix
│   ├── ResNet50.pt
│   ├── ViT_Large.pt
│   └── [model files]
├── diagonal/                # Diagonal constraint
├── zero_shot/              # Identity matrix (no learning)
└── [constraint types]
```

**File format**: PyTorch saved modules (.pt files)  
**Each .pt contains**: CogModel (nn.Module) with fitted W matrix, temperature, similarity function

**Download**: [Figshare Repository](https://figshare.com/articles/dataset/Data_and_analysis_results_/30542690) (pre-computed results and data)



#### `Data/models_data_generating_L1/` & `Data/models_data_generating_L2/`
**Status**: GENERATED (alternative regularization experiments)

Same structure as `models_data_generating/` but with L1 or L2 regularization instead of eye_distance.

#### `Data/models_n_params_table/`
**Status**: GENERATED by `create_models_properties_table.py`

Model architecture metadata (number of parameters, layers, etc.):
- `models_properties.csv` - Table with model stats
- Used for regression analysis (Table S3)


---

### 📊 `Results/` - Experimental Results (GITIGNORED)

**Status**: GITIGNORED (large databases and intermediate outputs)

**Purpose**: Stores all experimental results and analysis outputs.

#### `Results/Simulations_experiments/`
**Status**: GENERATED by `create_simulations_db.py` and `run_simulations.py` OR DOWNLOAD from Figshare

SQLite databases with model recovery simulation results:

```
Simulations_experiments/
├── full_W/
│   └── full_W.db           # Main experiment database
├── full_W_PCA_500/
│   └── full_W_PCA_500.db   # With PCA-reduced features
├── zero_shot/
│   └── zero_shot.db        # Zero-shot (identity matrix)
├── diagonal/
│   └── diagonal.db         # Diagonal constraint
└── [other experiments]
```

**Database schema**:
- `jobs` table: Job specifications (n_triplets, simulation_idx, data_gen_model, status, etc.)
- `reference_model_results` table: Performance metrics for each reference model tested

**Download**: [Figshare Repository](https://figshare.com/articles/dataset/Data_and_analysis_results_/30542690) (pre-computed results and data)



**Query example**:
```bash
sqlite3 Results/Simulations_experiments/full_W/full_W.db \
    "SELECT COUNT(*) FROM jobs WHERE status='done';"
```

#### `Results/models_accuracy_analysis_THINGS_accuracy/`
**Status**: GENERATED by `evaluate_THINGS_OOO_accuracy_CV.py`

Cross-validated prediction accuracy on real THINGS data (not synthetic):

```
models_accuracy_analysis_THINGS_accuracy/
├── full_W/
│   ├── ResNet50_results.json
│   ├── ViT_Large_results.json
│   └── [model results]
└── [constraint types]
```

**Each JSON contains**: test_acc, train_acc, val_acc, NLL, predictions


**Usage**: Used for Figure 5 X-axis (THINGS prediction accuracy)

#### `Results/rsa/`
**Status**: GENERATED by `build_rsa_sim_matrix.py`

Representational Similarity Analysis matrices:
- `rsa_matrix.npy` - RSA similarity between all model pairs
- `rsa_matrix_pre_transformation.npy` - Before W transformation
- `rsa_matrix_post_transformation.npy` - After W transformation



**Usage**: Figure S4 (RSA heatmap), Figure 4 (MDS projection)

#### `Results/regression_analysis/`
**Status**: GENERATED by `Table_S3_regression.py`

Statistical regression analysis outputs:
- `regression_analysis.csv` - Bootstrap regression coefficients

---


---

### 🔬 `experiment_scripts/` - Main Pipeline Scripts 

**Purpose**: Core pipeline stages (data prep → simulation → results).

**Files**:

1. **`create_data_generating_models.py`** (Stage 1)
   - Fits neural network features to THINGS behavioral data
   - Creates CogModel instances with learned W matrices
   - Saves to `Data/models_data_generating/`


2. **`create_simulations_db.py`** (Stage 2)
   - Creates SQLite database with all simulation job specifications
   - Does NOT run simulations, just prepares the database
   - Output: `Results/Simulations_experiments/<experiment>/<experiment>.db`


3. **`run_simulations.py`** (Stage 3)
   - Executes model recovery experiments
   - Fetches pending jobs from database, runs them, saves results
   - Supports parallel execution (multiple workers)
**Configuration**: All scripts use Hydra configs from `scripts_configurations/`

**Warning:even with parallel execution runinng the simulations takes alot of time please**

---

### 📊 `figures_scripts/` - Figure Generation 
**Purpose**: Generate publication-ready figures from simulation results.

**Files**:

- `create_Figures_1_and_3.py` - Main model recovery analysis (and also reduced dimensions simulation analysis)
- `create_Figure_2.py` - THINGS accuracy for each model
- `create_Figure_4_MDS.py` - MDS projection represntations before and after transformation of each model. 
- `create_Figure_5_tradeoff.py` - Flexibility-accuracy tradeoff
- `create_Figure_S1_violin_plot.py` - Accuracy distribution violin plots comparing the accuracy between L2 and scalar shrinkage.
- `create_Figure_S4_RSA.py` - RSA heatmap
- `create_Figure_S5_S4.py` - Extended models with grouping type model recovery analysis

**Output**: Save figures to `Plots/`


---

### 🔍 `analysis_scripts/` - Analysis Utilities 

**Purpose**: Additional analyses not part of the main pipeline.

**Files**:

- `evaluate_THINGS_OOO_accuracy_CV.py` - Evaluate models on real THINGS data (cross-validated)
- `save_transformed_features.py` - Apply W transformations and save transformed features
- `build_rsa_sim_matrix.py` - Compute RSA similarity matrix between models
- `create_models_properties_table.py` - Extract model architecture metadata
- `Table_S3_regression.py` - Bootstrap regression analysis for Table S3
- `figure_5_analysis.py` - Helper analysis for Figure 5 generation

**Usage**: Run independently or as prerequisites for certain figures

---

### ⚙️ `scripts_configurations/` - Hydra Configs 

**Purpose**: Centralized parameter management for all scripts via Hydra.

**Structure**:

```
scripts_configurations/
├── shared_args.yaml         # Base config imported by others
│                            # Contains: model lists, paths, default parameters
├── create_data_generating_models.yaml
├── create_simulations_db.yaml
├── run_simulations.yaml
├── figures_1_and_3.yaml
├── Figure_5.yaml
├── RSA_MDS_figures.yaml
└── [other experiment configs]
```

**Key pattern**:
```yaml
# Example: figures_1_and_3.yaml
defaults:
  - shared_args  # Import base configuration

full_W:  # Experiment name
  db_path: "Results/Simulations_experiments/full_W/full_W.db"
  constraint_type: null
  n_train_triplets_list: [100, 200, 500, ..., 100000]
```

**Usage**:
```bash
python run_simulations.py experiment_type=full_W  # Selects 'full_W' section from config
python create_Figures_1_and_3.py experiment_name=full_W_PCA_500  # Override experiment name
```

---

### 🛠️ `tools/` - Core Library (COMMITTED)

**Status**: COMMITTED (Python module)

**Purpose**: Reusable library functions used by all scripts.

**Key modules**:

1. **`cog_model.py`** - CogModel class (central model implementation)
   - Implements linear transformation + similarity function
   - Supports various constraints (full_W, diagonal, rectangular, orthogonal)
   - Handles fitting, simulation, and evaluation

2. **`model_recovery_fn.py`** - Model recovery experiment orchestration
   - Creates disjoint concept subsets for cross-validation
   - Simulates human responses
   - Runs model recovery experiments

3. **`db_tools.py`** - Database utilities
   - `init_database()` - Create schema
   - `create_jobs_from_combinations()` - Generate job combinations
   - `get_pending_job()` - Atomic job fetching
   - `save_reference_model_results()` - Save experiment results

4. **`db_analysis.py`** - Analysis and visualization
   - `DBResultsAnalysis` - Query and analyze simulation results
   - `ModelRecoveryVisualizer` - Generate publication figures
   - Confusion matrices, recovery curves, precision/recall/F1/MRR

5. **`model_io.py`** - Save/load model utilities
   - `save_cog_model()` - Serialize CogModel to disk
   - `load_cog_model()` - Load CogModel from file

6. **`triplets_data_set.py`** - Dataset classes
   - `OddOneOutDataset` - Triplet data with human choices
   - `ConceptsDataSet` - Image concept metadata
   - DataLoader utilities

7. **`similarity_functions.py`** - Similarity metrics
   - `dot_product()`, `cosine_similarity()`, `euclidean_distance()`

8. **`losses.py`** - Loss functions
   - `Triplets_cross_entropy_loss` - Cross-entropy for triplet tasks

9. **`reg_funcs.py`** - Regularization
   - `RegularizationFunction` class
   - Supports: L1, L2, eye_distance (shrinkage toward identity)

10. **`constraints.py`** - Constraint parametrizations
    - `OrthogonalScaledLinear` - Orthogonal transformation with scaling
    - `DiagonalParametrization` - Diagonal matrix constraint
    - `constraints_wrapper()` - Unified interface

11. **`utils.py`** - General utilities
    - Optimizer setup, batch size calculation
    - Triplet generation functions
    - Dataset partitioning for cross-validation

---

### 🗂️ `outputs/` - Hydra Logs (GITIGNORED)

**Status**: GITIGNORED (auto-generated by Hydra)

**Purpose**: Stores run logs organized by date.

```
outputs/
├── 2025-10-27/
│   └── run_001/
│       ├── .hydra/
│       │   ├── config.yaml
│       │   └── overrides.yaml
│       └── main.log
└── [other dates]
```

**Size**: ~1-10MB per run

**Cleanup**: Can safely delete old runs

---

### 📄 Root-Level Files

#### Documentation

- **`README.md`** - Main landing page (installation, quick start)
- **`INSTRUCTIONS.md`** - Detailed setup and workflow guide
- **`USAGE_GUIDE_evaluate_THINGS.md`** - Specific guide for THINGS evaluation

#### Configuration

- **`requirements.txt`** - Python dependencies (pip)
- **`environment.yml`** - Conda environment specification
- **`.gitignore`** - Git ignore patterns
---

## Path Conventions

### Absolute vs. Relative Paths

**All scripts use relative paths** from repository root:
- ✅ `Data/models_features/full/ResNet50.pt`
- ❌ `/home/user/project/Data/models_features/full/ResNet50.pt`

**Pattern in scripts**:
```python
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Change to repo root
```

This ensures scripts work regardless of where the repo is cloned.

### Configuration Paths

**Paths defined in** `scripts_configurations/shared_args.yaml`:
```yaml
paths:
  things_data_path: "Data/Things_data_preprocessed/"
  models_features_path: "Data/models_features/full/"
  results_folder: "Results/"
  data_generating_models_path: "Data/models_data_generating/"
```

**Access in scripts** via Hydra:
```python
db_path = cfg[experiment_type]['paths']['db_path']
```

---

## Git Tracking Strategy

### What's Committed

- ✅ Source code (`.py` files)
- ✅ Configuration files (`.yaml`)
- ✅ Documentation (`.md` files, `docs/`)
- ✅ Small metadata files
- ✅ Requirements (requirements.txt, environment.yml)

### What's Ignored

- ❌ `Data/` - Too large (100GB+)
- ❌ `Results/` - Large databases and outputs
- ❌ `Plots/` - Can be regenerated
- ❌ `outputs/` - Hydra logs
- ❌ `__pycache__/` - Python bytecode
- ❌ `.vscode/` - Editor settings

---
