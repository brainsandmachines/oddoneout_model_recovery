[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

# Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One

Official code repository for the NeurIPS 2025 paper ["Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One"](https://arxiv.org/abs/2510.23321).

---

## 📝 Citation

```latex
@inproceedings{avitan2025modelbehavior,
  author    = {Avitan, Itamar and Golan, Tal},
  title     = {Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url={https://openreview.net/forum?id=kE4XEY7Bbc}
}
```


## TL;DR

Linearly transforming neural network representations to predict human behavior is standard practice in cognitive neuroscience. But does high predictive accuracy mean we've found the right model? 

**We show**: Even with 4.5M behavioral judgments, model recovery accuracy plateaus below 80% when using flexible linear transformations. Overly flexible alignment metrics may fail to identify genuinely human-aligned representations.

## Key Contributions

1. We provide a **large-scale model recovery framework** for evaluating representational alignment methods in discrete behavioral tasks
2. We empirically demonstrate that **standard linear probing fails to reliably identify data-generating models**, even with millions of training samples
3. We find that **alignment-induced representational geometry shifts and elevated effective dimensionality drive misidentification**.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/brainsandmachines/oddoneout_model_recovery
cd oddoneout_model_recovery

# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate rethinking_alignment

# Option 2: pip
pip install -r requirements.txt
```

---
## Repository Structure

```
.
├── README.md                       # This file
├── GUIDE_reanalyze_existing_data.md   # Guide: how to regenerate figures from simulation results
├── GUIDE_full_rerun_from_model_features.md  # Guide: how to run the full simulation pipeline from scratch
├── requirements.txt                # Python dependencies
├── avitan2025modelbehavior.pdf     # Camera-ready NeurIPS 2025 paper
│
├── docs/                          # Documentation
│   ├── pipeline_map.md            # Complete pipeline flow with diagrams
│   └── repo-structure.md          # Detailed file system map
│
├── Data/                          # Input data (to be downloaded separately)
│   ├── Things_data_preprocessed/  # THINGS triplets + human choices
│   ├── models_features/           # Pre-extracted neural network features
│   ├── models_data_generating/    # Fitted CogModels (.pt files)
│   └── models_n_params_table/     # Model architecture metadata
│
├── Results/                       # Experimental results 
│   ├── Simulations_experiments/   # SQLite databases with simulation results
│   ├── rsa/                      # RSA similarity matrices
│   ├── regression_analysis/       # Statistical analysis outputs
│   └── [other analysis folders]/
││
├── experiment_scripts/            # Main pipeline scripts
│   ├── create_data_generating_models.py   # Stage 1: Fit models to THINGS data
│   ├── create_simulations_db.py           # Stage 2: Create job database
│   └── run_simulations.py                 # Stage 3: Execute simulations
│
├── figures_scripts/               # Figure generation scripts
│   ├── create_Figures_1_and_3.py
│   ├── create_Figure_5_tradeoff.py
│   └── [other figure scripts]/
|
├── Plots/                         # Publication figures (generated)
│
├── analysis_scripts/                      # Analysis utilities & preprocessing
│   ├── odd_one_out_data_preprocessing.py  # Stage 0A: CSV → tensor conversion
│   ├── things_noise_ceiling_eval.py       # Stage 0B: Noise ceiling calculation
│   ├── evaluate_THINGS_OOO_accuracy_CV.py # Model evaluation (essential for Figure 2)
│   ├── Table_S3_regression.py
│   └── ...                         
│
├── tools/                        # Core library modules
│   ├── cog_model.py              # CogModel implementation
│   ├── model_recovery_fn.py      # Model recovery framework
│   ├── db_tools.py               # Database utilities
│   ├── db_analysis.py            # Analysis and visualization
│   └── ...
│
└── scripts_configurations/        # Hydra configuration files (YAML)
    ├── shared_args.yaml          # Base config (model lists, paths)
    ├── run_simulations.yaml
    ├── figures_1_and_3.yaml
    └── [other experiment configs]/
```

**See `docs/repo-structure.md` for detailed folder descriptions and data flow.**

---

## Documentation

We provide three levels of documentation for different use cases:

1. **`README.md`** (this file) - Quick start and overview
2. **User Guides** - Step-by-step workflows:
   - `GUIDE_reanalyze_existing_data.md` - Regenerate figures from pre-computed results
   - `GUIDE_full_rerun_from_model_features.md` - Complete pipeline from scratch
3. **Technical Documentation**:
   - `INSTRUCTIONS.md` - Comprehensive technical reference (pipeline details, API docs)
   - `docs/pipeline_map.md` - Visual pipeline flow with Mermaid diagrams
   - `docs/repo-structure.md` - File system organization and data dependencies

---

## Workflow Options

### Option 1: Reanalyze Pre-computed Results (Fastest)

```bash
# Download simulation databases and pre-computed results and data
# Link: https://figshare.com/articles/dataset/Data_and_analysis_results_/30542690
# Place it in the project main folder

# Generate all figures
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W"
# ... (see Quick Start above)
```

**Guide**: `GUIDE_reanalyze_existing_data.md`

### Option 2: Full Replication from Model Features

```bash
# Stage 1: Fit data-generating models
python experiment_scripts/create_data_generating_models.py experiment_type=full_W

# Stage 2: Create simulation database 
python experiment_scripts/create_simulations_db.py experiment_type=full_W

# Stage 3: Run simulations in parallel 
python experiment_scripts/run_simulations.py experiment_type=full_W

# Stage 4: Generate figures 
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W"
```

**Guide**: `GUIDE_full_rerun_from_model_features.md`

---

## Key Concepts

### Model Recovery Framework

**Question**: If we fit different neural network models to human behavioral data, can we identify which model generated new behavioral data?

**Our approach**:
1. Fit transformation matrices (W) to align each model's features with human odd-one-out judgments
2. Use fitted models to generate synthetic behavioral data
3. Test whether we can correctly identify the generating model

### Transformation Flexibility Levels

The pipeline supports different levels of transformation flexibility:

- **full_W**: Unconstrained linear transformation (maximum flexibility, p×p matrix)
- **diagonal**: Element-wise feature scaling (diagonal matrix only)
- **rectangular_k**: Low-rank projection (reduces to k dimensions)
- **zero_shot**: No transformation (identity matrix, tests raw features)
- **orthogonal**: Orthogonal transformation with scaling

### Regularization Types

Different regularization functions control how transformations are learned:

- **eye_distance** (default): Penalize distance from identity matrix
- **L2**: Standard weight decay (penalize large weights)
- **L1**: Sparse weights (penalize absolute values)

**Important - Output Directory Naming**: 

The evaluation script can compare multiple regularization types simultaneously (default: both L2 and eye_distance). Results for all types are saved in the **same directory**, with separate files per regularization:

```bash
# Default config evaluates BOTH L2 and eye_distance:
reg_func_types: ["L2", "eye_distance"]
results_out: "Results/regularization_methods_compare_all_models_eye_distance"

# This creates:
# Results/regularization_methods_compare_all_models_eye_distance/full/full_W/
#   ├── best_test_acc_L2_full_full_W.csv              # L2 results
#   ├── best_test_acc_eye_distance_full_full_W.csv    # eye_distance results
#   └── best_test_acc_for_each_model_full_full_W.csv  # Combined comparison
```

**Directory naming convention**:
- **Multiple regularizations** (comparison): Use primary/default in path suffix (e.g., `_eye_distance`)
- **Single regularization**: Use that regularization in path suffix (e.g., `_L2` for L2 only)

The same `results_out` path must be used in `figures_2.yaml` as `results_base_dir`. The figure script will then select which regularization to display using the `regularization_type` parameter.

### Dimensionality Reduction

Features can optionally be PCA-reduced before fitting:

- **Full features**: Original extracted activations (e.g., 2048-dim for ResNet50)
- **PCA_500**: Reduce to 500 dimensions 


### Transformation Constraints

We test different flexibility levels:

| Constraint | Parameters | Description | Use Case |
|-----------|-----------|-------------|----------|
| **Full W** | p² | Unconstrained linear transformation | Maximum flexibility (main experiment) |
| **Diagonal** | p | Element-wise scaling only | Minimal flexibility |
| **Rectangular** | p×k | Low-rank projection | Controlled flexibility |
| **Zero-shot** | 0 | Identity matrix (no learning) | Baseline |
---

##  Configuration System

All scripts use [Hydra](https://hydra.cc/) for configuration management.

**Base configuration**: `scripts_configurations/shared_args.yaml`
```yaml
data_generating_models: [ResNet50, ViT_Large, CLIP_RN50, ...]  # 20 models
paths:
  things_data_path: "Data/Things_data_preprocessed/"
  models_features_path: "Data/models_features/full/"
defaults:
  regularization_constants_list: [0.00001, ..., 100]
  reg_func: "eye_distance"
  K_folds: 3
```

**Override from command line**:
```bash
python run_simulations.py experiment_type=full_W device=cpu
python figures_scripts/create_Figures_1_and_3.py experiment_name=full_W_PCA_500
```

---

## Datasets

### THINGS Odd-One-Out Dataset

**Source**: 
- THINGS images: [THINGS Dataset OSF Repository](https://osf.io/jum2f/)
- Odd-one-out behavioral data: [THINGS Odd-One-Out Dataset](https://osf.io/f5rn6/) (data/triplet_dataset/)

**Citation**:
> Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020). Revealing the multidimensional mental representations of natural objects underlying human similarity judgements. *Nature Human Behaviour*, 4(11), 1173-1185.

**Description**: 4.70M human odd-one-out judgments across 1,854 object concepts

### Pre-computed Results and Data

**Download link**: [Figshare Repository - Data and Analysis Results](https://figshare.com/articles/dataset/Data_and_analysis_results_/30542690)
---

## Additional Resources

- **Pipeline Diagram**: `docs/pipeline_map.md`
- **File System Map**: `docs/repo-structure.md`
---


## Contact

**Itamar Avitan**  
Ben-Gurion University of the Negev    
Email: avitanit@post.bgu.ac.il

**Tal Golan**  
Ben-Gurion University of the Negev  
Email: golan.neuro@bgu.ac.il

Lab Website: [brainsandmachines.org](https://brainsandmachines.org)

---
