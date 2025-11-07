# Guide: Full Rerun from Model Features

This guide is for users who want to **completely reproduce the paper's results** from scratch, starting only from neural network features extracted on the THINGS dataset.

**Assumed starting point**: You have extracted model features for 1854 THINGS concepts and want to run the full analysis pipeline.

---

## Prerequisites

### Required Data

You must provide:

#### 1. THINGS Odd-One-Out Dataset

**Raw behavioral data**:
- Download from: https://osf.io/f5rn6/ (data/triplet_dataset/)
- File: `triplets_large_final_correctednc_correctedorder.csv`
- Place in: `Data/Things_data/behavior_data/`

**THINGS images** (if doing feature extraction):
- Download from: https://osf.io/jum2f/
- 1,854 concept images used in the odd-one-out task

**Preprocessing** (Stage 0 - Required before main pipeline):

```bash
# Stage 0A: Convert CSV to tensors (1-based → 0-based indexing)
python analysis_scripts/odd_one_out_text_preprocess.py

# Stage 0B: Compute noise ceiling (~67.8% for test set 2)
python analysis_scripts/things_noise_ceiling_eval.py
```

This creates preprocessed tensors in `Data/Things_data_preprocessed/`:

```
Data/Things_data_preprocessed/
├── Things_odd_one_out_triplets_train_set.pt           # Training triplets
├── Things_odd_one_out_answers_positions_train_set.pt  # Training responses
├── Things_odd_one_out_triplets_testset2_noise_ceiling.pt  # Test set 2 (PRIMARY)
├── Things_odd_one_out_answers_positions_testset2_noise_ceiling.pt
├── things_datasets_info.csv                            # Dataset statistics
└── unique_triplets_and_counts/
    ├── test_2_unique_triplets.pt                       # For noise ceiling
    ├── test_2_answers_counts.pt
    └── analysis_noise_ceiling.csv                      # ~67.8% for test set 2
```

**Configuration files**: 
- `scripts_configurations/odd_one_out_preprocess.yaml`
- `scripts_configurations/things_noise_ceiling_eval.yaml`

**Note**: Test set 2 (~67.8% noise ceiling) is the PRIMARY validation set used throughout the paper.

#### 2. Model Features

Extract features for all 1854 THINGS concepts using your neural network models.

Place in `Data/models_features/full/`:

```
Data/models_features/full/
├── ResNet50.pt           # [1854, feature_dim] torch tensor
├── ViT_Large.pt          # [1854, feature_dim] torch tensor
├── CLIP_RN50.pt          # [1854, feature_dim] torch tensor
└── [your_model.pt]       # [1854, feature_dim] torch tensor
```

**Format**: Each file must be a PyTorch tensor of shape `[1854, feature_dim]`
- 1854 rows (one per THINGS concept, in same order as triplets)
- feature_dim columns (model-dependent: 512-4096 typically)

**Feature extraction examples**:

```python
# Example 1: Extract from torchvision model
import torch
import torchvision.models as models
from PIL import Image
import glob

model = models.resnet50(pretrained=True)
model.eval()
model = model.cuda()

# Remove final classification layer to get features
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

features = []
for img_path in sorted(glob.glob('Data/Things_data/images/*.jpg')):  # Adjust path
    img = Image.open(img_path).convert('RGB')
    # Apply standard preprocessing (resize, normalize, etc.)
    img_tensor = preprocess(img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        feat = feature_extractor(img_tensor)
    features.append(feat.squeeze().cpu())

features_tensor = torch.stack(features)  # [1854, feature_dim]
torch.save(features_tensor, 'Data/models_features/full/ResNet50.pt')
```

```python
# Example 2: Extract from CLIP
import clip
import torch

model, preprocess = clip.load("RN50", device="cuda")

features = []
for img_path in sorted(glob.glob('Data/Things_data/images/*.jpg')):
    img = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
    
    with torch.no_grad():
        feat = model.encode_image(img)
    features.append(feat.squeeze().cpu())

features_tensor = torch.stack(features)
torch.save(features_tensor, 'Data/models_features/full/CLIP_RN50.pt')
```

**Size**: ~2-5GB per model (depends on feature dimensionality)

---

## Installation

Same as in `GUIDE_reanalyze_existing_data.md`:

```bash
# Clone repository
git clone https://github.com/yourusername/repo-name.git
cd repo-name

# Create conda environment
conda env create -f environment.yml
conda activate rethinking_alignment

# Verify installation
python -c "import torch; import hydra; print('Setup successful!')"
```

---

## Configuration

### Step 1: Register Your Models

Edit `scripts_configurations/shared_args.yaml` to include your models:

```yaml
# shared_args.yaml

# Base models (original 20 from paper)
data_generating_models:
  - ResNet50
  - ViT_Large
  - CLIP_RN50
  - ... (other models)

# Additional models (for extended analysis)
new_models:
  - YourNewModel1
  - YourNewModel2
```

**Important**: Model names must match your feature filenames (without .pt extension).

### Step 2: Configure Experiment Parameters

Check/modify experiment parameters in `scripts_configurations/create_data_generating_models.yaml`:

```yaml
# create_data_generating_models.yaml

defaults:
  - shared_args

full_W:  # Experiment name
  constraint_type: null  # null = unconstrained W matrix
  reg_func: "eye_distance"  # Regularization: eye_distance, L1, or L2
  regularization_constants_list: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
  K_folds: 3  # Number of CV folds
  device: "cuda"  # or "cpu"
  
  paths:
    things_data_path: "Data/Things_data_preprocessed/"
    models_features_path: "Data/models_features/full/"
    save_folder: "Data/models_data_generating/full_W/"
```

**Key parameters**:
- `constraint_type`: null (full_W), "diagonal", ("rectangular", k), "zero_shot", "orthogonal"
- `reg_func`: "eye_distance" (recommended), "L1", or "L2"
- `K_folds`: 3 or 5 (more folds = slower but more robust)
- `device`: "cuda" or "cpu" (GPU strongly recommended)

---

## Pipeline Execution

The full pipeline consists of 3 main stages:

### Stage 1: Create Data-Generating Models

**Purpose**: Fit neural network features to THINGS behavioral data.

**Run**:
```bash
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W \
    device=cuda
```

**What it does**:
1. Loads model features from `Data/models_features/full/`
2. Loads THINGS triplets and human choices
3. For each model:
   - Creates a CogModel with the specified constraints
   - Fits the W transformation matrix using the LBFGS optimizer
   - Fits the temperature parameter to match the human noise ceiling
   - Saves the fitted model to disk (see Stage 1 Output below)

After running, you should have:
```
Data/models_data_generating/<constraint_type>/<model>.pt
```

Each .pt file contains a saved PyTorch CogModel module with:
- Fitted W transformation matrix (respecting specified constraints)
- Temperature parameter
- Similarity function configuration
- Regularization settings

**Flexibility levels** (constraint_type options):
- `full_W`: Unconstrained linear transformation (maximum flexibility)
- `diagonal`: Element-wise scaling only
- `rectangular_50`: Low-rank projection to 50 dimensions
- `zero_shot`: Identity matrix (no learning)
- `orthogonal`: Orthogonal transformation with scaling

**Regularization types** (reg_func options):
- `eye_distance`: Penalize distance from identity matrix (default)
- `L2`: Standard weight decay
- `L1`: Sparse weight penalty

**Expected runtime**: 10-30 minutes per model
- Full_W (unconstrained): ~20-30 min/model
- Diagonal: ~5-10 min/model
- Zero-shot: ~1 min/model (just saves identity matrix)

**Outputs**:
```
Data/models_data_generating/full_W/
├── ResNet50.pt
├── ViT_Large.pt
├── CLIP_RN50.pt
└── [all your models]
```

**Customization**:
```bash
# Fit only specific models
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W \
    models_to_fit='["ResNet50", "ViT_Large"]'

# Use different constraint type
python experiment_scripts/create_data_generating_models.py \
    experiment_type=diagonal \
    constraint_type="diagonal"

# Use different regularization
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W_L1 \
    reg_func="L1"
```

**Monitoring progress**:
```bash
# Check number of saved models
ls Data/models_data_generating/full_W/*.pt | wc -l

# Expected: Should match number of models you're fitting
```

---

### Stage 2: Create Simulation Database

**Purpose**: Generate all simulation job specifications (but don't run them yet).

**Run**:
```bash
python experiment_scripts/create_simulations_db.py \
    experiment_type=full_W
```

**What it does**:
1. Reads experiment configuration from `scripts_configurations/create_simulations_db.yaml`
2. Generates all job combinations:
   - n_triplets × simulations × data_gen_models × constraints
   - Example: 8 triplet counts × 10 simulations × 20 models = 1,600 jobs
3. Creates SQLite database with `jobs` and `reference_model_results` tables
4. Inserts all jobs with status='pending'

**Expected runtime**: < 1 minute

**Outputs**:
```
Results/Simulations_experiments/full_W/
└── full_W.db
```

**Configuration** (`scripts_configurations/create_simulations_db.yaml`):
```yaml
full_W:
  n_train_triplets_list: [100, 200, 500, 1000, 2000, 5000, 10000, 100000]
  simulations_idx_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 simulations per condition
  
  paths:
    data_generating_models_path: "Data/models_data_generating/full_W/"
    db_path: "Results/Simulations_experiments/full_W/full_W.db"
```

**Verify database**:
```bash
sqlite3 Results/Simulations_experiments/full_W/full_W.db \
    "SELECT COUNT(*) FROM jobs WHERE status='pending';"

# Expected output: Total number of jobs (e.g., 1600)
```

---

### Stage 3: Run Simulations

**Purpose**: Execute model recovery experiments.

**THIS IS THE SLOWEST STAGE** - can take days depending on parallelization.

#### Single Worker Execution

```bash
python experiment_scripts/run_simulations.py \
    experiment_type=full_W
```

**What it does**:
1. Fetches pending jobs from database (atomically)
2. Loads data-generating model from Stage 1 output
3. Generates synthetic behavioral data
4. Tests all reference models against the data
5. Saves results to database
6. Repeats until no pending jobs or time limit reached

**Expected runtime**: 
- ~5-30 minutes per job (depends on n_triplets and number of reference models)
- Single worker, 1,600 jobs: ~5-7 days
- **Recommendation**: Use parallel workers (see below)

#### Parallel Execution (Recommended)

**Option A: Multiple Workers on Single Machine**

```bash
# Open 4 terminals, run in each:
python experiment_scripts/run_simulations.py experiment_type=full_W &
python experiment_scripts/run_simulations.py experiment_type=full_W &
python experiment_scripts/run_simulations.py experiment_type=full_W &
python experiment_scripts/run_simulations.py experiment_type=full_W &

# Workers automatically coordinate via database locking
# Each fetches next available job
```

**Option B: Cluster with SLURM** (if available)

```bash
# Create SLURM script: run_simulations.sh
#!/bin/bash
#SBATCH --job-name=model_recovery
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=1-10  # 10 parallel workers

conda activate rethinking_alignment
python experiment_scripts/run_simulations.py experiment_type=full_W
```

```bash
# Submit array job
sbatch run_simulations.sh

# Monitor progress
squeue -u $USER
```

**Option C: Multiple Machines with Shared Storage**

If you have multiple machines accessing shared network storage:

```bash
# On machine 1:
python experiment_scripts/run_simulations.py experiment_type=full_W

# On machine 2 (simultaneously):
python experiment_scripts/run_simulations.py experiment_type=full_W

# etc.
```

Database locking ensures no duplicate work.

#### Monitor Progress

```bash
# Check job status distribution
sqlite3 Results/Simulations_experiments/full_W/full_W.db \
    "SELECT status, COUNT(*) FROM jobs GROUP BY status;"

# Expected output:
# pending|1200
# running|5
# done|395

# Watch progress in real-time
watch -n 60 'sqlite3 Results/Simulations_experiments/full_W/full_W.db \
    "SELECT status, COUNT(*) FROM jobs GROUP BY status;"'
```

#### Handle Stuck Jobs

Jobs may get stuck if workers crash:

```python
# Reset jobs stuck in 'running' state for >5 hours
from tools.db_tools import update_stuck_jobs
update_stuck_jobs('Results/Simulations_experiments/full_W/full_W.db', hours_threshold=5)
```

Or via command line:
```bash
python -c "
from tools.db_tools import update_stuck_jobs
update_stuck_jobs('Results/Simulations_experiments/full_W/full_W.db', 5)
print('Reset stuck jobs')
"
```

#### Time Limit Configuration

Control how long a single worker runs:

```bash
# Run for maximum 23.5 hours (cluster-friendly)
python experiment_scripts/run_simulations.py \
    experiment_type=full_W \
    time_limit_hours=23.5
```

Worker stops gracefully before time limit, then can be restarted.

---

### Stage 4: Generate Figures

Once all simulations are complete (`status='done'` for all jobs), generate figures:

```bash
# Main figures
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W"
python figures_scripts/create_Figure_5_tradeoff.py
python figures_scripts/create_Figure_2.py

# Supplementary figures
python figures_scripts/create_Figure_S1_violin_plot.py
python figures_scripts/create_Figure_S4_RSA.py
python figures_scripts/create_Figure_4_MDS.py

# Statistical analysis
python analysis_scripts/Table_S3_regression.py
```

See `GUIDE_reanalyze_existing_data.md` for details on figure generation.

---

## Advanced Workflows

### Add New Model to Existing Experiment

```bash
# 1. Extract features for new model
# Save to: Data/models_features/full/NewModel.pt

# 2. Add to config
# Edit scripts_configurations/shared_args.yaml:
# new_models: [..., "NewModel"]

# 3. Fit data-generating version (Stage 1)
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W \
    models_to_fit='["NewModel"]'

# 4. Add jobs to database (Stage 2)
# NOTE: This will add new jobs for NewModel only if database doesn't have them
python experiment_scripts/create_simulations_db.py experiment_type=full_W

# 5. Run simulations for new jobs (Stage 3)
python experiment_scripts/run_simulations.py experiment_type=full_W

# 6. Regenerate figures (Stage 4)
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W"
```

### Test Different Constraint Types

```bash
# Diagonal constraint
python experiment_scripts/create_data_generating_models.py experiment_type=diagonal
python experiment_scripts/create_simulations_db.py experiment_type=diagonal
python experiment_scripts/run_simulations.py experiment_type=diagonal

# Low-rank (k=50)
python experiment_scripts/create_data_generating_models.py experiment_type=rectangular_50
python experiment_scripts/create_simulations_db.py experiment_type=rectangular_50
python experiment_scripts/run_simulations.py experiment_type=rectangular_50

# Zero-shot (identity matrix)
python experiment_scripts/create_data_generating_models.py experiment_type=zero_shot
python experiment_scripts/create_simulations_db.py experiment_type=zero_shot
python experiment_scripts/run_simulations.py experiment_type=zero_shot
```

### Test Different Regularization Methods

```bash
# L1 regularization
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W_L1 \
    reg_func=L1

# L2 regularization
python experiment_scripts/create_data_generating_models.py \
    experiment_type=full_W_L2 \
    reg_func=L2
```

## Next Steps

- **Explore custom analyses**: Modify scripts or create new ones
- **Test different models**: Add your own neural network features
- **Vary constraints**: Test diagonal, orthogonal, low-rank transformations
- **Statistical analyses**: Use `DBResultsAnalysis` class for custom queries

---
