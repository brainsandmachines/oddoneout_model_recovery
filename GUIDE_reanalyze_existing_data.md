# Guide: Reanalyze Existing Data

This guide is for users who have access to **pre-computed simulation databases** and want to regenerate the paper's figures and analyses without rerunning expensive simulations.

**Assumed starting point**: You have downloaded the simulation results databases but not the full THINGS dataset or model features.

---

## Prerequisites
For precalculated data and results: \
**Download link**: [Figshare Repository](https://figshare.com/articles/dataset/Data_and_analysis_results_/30542690)  
### Required Files

You must have these files downloaded and placed in the correct locations:

#### 1. Simulation Databases

Download and place in `Results/Simulations_experiments/`:


```
Results/Simulations_experiments/
│   ├── full_W/full_W.db
│   ├── diagonal/diagonal.db
│   └── zero_shot/zero_shot.db
```




#### 2. Fitted Data-Generating Models (for some analyses)

Some analyses require the fitted CogModel instances. Download and place in `Data/models_data_generating/`:


```
Data/models_data_generating/full_W/
│   ├── ResNet50.pt
│   ├── ViT_Large.pt
│   ├── CLIP_RN50.pt
│   └── [~20-30 .pt files]
└── [other constraint types if needed]
```


**Note**: Only needed for RSA/MDS analyses (Figures 4, S4). Main figures (1, 3, 5) only need the databases.

#### 3. Model Properties Tables (for Table S3)

```
Data/models_n_params_table/
└── models_properties.csv
```



---

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/repo-name.git
cd repo-name

# Create and activate conda environment
conda env create -f environment.yml
conda activate rethinking_alignment

# Verify installation
python -c "import torch; import hydra; print('Setup successful!')"
```

### Option 2: pip

```bash
# Clone repository
git clone https://github.com/yourusername/repo-name.git
cd repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import hydra; print('Setup successful!')"
```

---

## Verify Database Integrity

Before running scripts, verify the databases are complete:

```bash
# Check that databases exist and have completed jobs
sqlite3 Results/Simulations_experiments/full_W/full_W.db "SELECT COUNT(*) FROM jobs WHERE status='done';"

# Expected output: Should show number of completed jobs (e.g., 1600)
# If output is 0 or much lower than expected, database may be incomplete
```

### Database Status Check

```bash
# Check job status distribution
sqlite3 Results/Simulations_experiments/full_W/full_W.db "SELECT status, COUNT(*) FROM jobs GROUP BY status;"

# Expected output:
# done|1600
# (All jobs should be 'done'; if you see 'pending' or 'running', database is incomplete)
```

### Verify Reference Model Results

```bash
# Check that results table is populated
sqlite3 Results/Simulations_experiments/full_W/full_W.db "SELECT COUNT(*) FROM reference_model_results;"

# Expected output: ~20-30 times the number of jobs (one result per reference model per job)
# For 1600 jobs with 20 reference models: ~32,000 results
```

---

## Generate Figures

All figures can be generated independently. Run in any order.

### Main Figures

#### Figure 1 & 3: Model Recovery Analysis

```bash
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W"
```

**Outputs**:
- `Plots/full_W/Figure_1.pdf` - Recovery accuracy curves + confusion matrices
- `Plots/full_W/Figure_3.pdf` - Ranking analysis (correct vs. incorrect)

**Expected run time**: 2-3 minutes

**What it shows**:
- Panel A: Recovery accuracy vs. number of training triplets
- Panels B-D: Confusion matrices at 400, 25.6K, 1.6M triplets
- Figure 3: Mean rank when correct model is recovered vs. when incorrect

**Troubleshooting**:
- **Error**: `FileNotFoundError: Results/Simulations_experiments/full_W/full_W.db`
  - **Fix**: Ensure database is in correct location with exact filename
- **Error**: `sqlite3.OperationalError: no such table: jobs`
  - **Fix**: Database file is corrupted; re-download
- **Empty plot**: Database has no completed jobs; verify with status check above

#### Figure 5: Flexibility-Accuracy Tradeoff

```bash
python figures_scripts/create_Figure_5_tradeoff.py
```

**Outputs**:
- `Plots/Figure_5.pdf`

**Expected run time**: 3-5 minutes

**Dependencies**: Requires multiple experiment databases:
- `Results/Simulations_experiments/full_W/full_W.db`
- `Results/Simulations_experiments/zero_shot/zero_shot.db`
- `Results/Simulations_experiments/diagonal/diagonal.db`
- `Results/models_accuracy_analysis_THINGS_accuracy/` (THINGS evaluation results)

**Note**: If you only have the full_W database, this figure will fail. Download other experiment databases or comment out missing experiments in the config.

### Supplementary Figures

#### Figure S1: Accuracy Distribution Violin Plots

```bash
python figures_scripts/create_Figure_S1_violin_plot.py
```

**Outputs**:
- `Plots/Figure_S1.pdf`

**Expected run time**: 1-2 minutes

#### Figure S3: Model Recovery with PCA-Reduced Features

```bash
python figures_scripts/create_Figures_1_and_3.py experiment_name="full_W_PCA_500"
```

**Outputs**:
- `Plots/full_W_PCA_500/Figure_1.pdf`
- `Plots/full_W_PCA_500/Figure_3.pdf`

**Dependencies**: Requires `Results/Simulations_experiments/full_W_PCA_500/full_W_PCA_500.db`

### RSA and MDS Figures (Require Fitted Models)

⚠️ **These figures require fitted data-generating models**, not just databases.

#### Figure 4: MDS Projection of Model Similarities

```bash
# First, check if RSA matrix exists (may be pre-computed)
ls Results/rsa/rsa_matrix.npy

# If missing, generate it (requires fitted models):
python analysis_scripts/save_transformed_features.py
python analysis_scripts/build_rsa_sim_matrix.py

# Then generate figure:
python figures_scripts/create_Figure_4_MDS.py
```

**Outputs**:
- `Plots/Figure_4.pdf`

**Expected run time**: 
- RSA computation: ~10-20 minutes
- Figure generation: ~1 minute

**Dependencies**:
- `Data/models_data_generating/full_W/*.pt` (fitted CogModel modules)
- OR pre-computed `Results/rsa/rsa_matrix.npy`

#### Figure S4: RSA Heatmap

```bash
# Same dependencies as Figure 4
python figures_scripts/create_Figure_S4_RSA.py
```

**Outputs**:
- `Plots/Figure_S4.pdf`

**Expected run time**: 1-2 minutes (after RSA matrix is computed)

---

## Generate Statistical Tables

### Table S3: Regression Analysis

```bash
python analysis_scripts/Table_S3_regression.py
```

**Outputs**:
- `Results/regression_analysis/regression_table.csv`
- `Results/regression_analysis/regression_table.tex` (LaTeX format)

**Expected run time**: 5-10 minutes (bootstrap regression)

**Dependencies**:
- `Results/Simulations_experiments/full_W/full_W.db`
- `Data/models_n_params_table/models_properties.csv`
- `Results/effective_dimensionality_score/` (pre-computed or generated on-the-fly)

**What it computes**: Bootstrap regression to identify predictors of model recovery accuracy (parameters, layers, effective dimensionality)

---

## Customization and Exploration

### Query Databases Directly

You can query the databases using SQL to explore the data:

```bash
# Open database in SQLite
sqlite3 Results/Simulations_experiments/full_W/full_W.db

# List all tables
.tables

# Show schema
.schema jobs
.schema reference_model_results

# Example queries:

# 1. Get all unique data-generating models
SELECT DISTINCT data_generating_model FROM jobs;

# 2. Get mean recovery accuracy for a specific model at different training sizes
SELECT 
    j.n_train_triplets,
    AVG(r.test_accuracy) as mean_accuracy,
    COUNT(*) as n_simulations
FROM jobs j
JOIN reference_model_results r ON j.id = r.job_id
WHERE j.data_generating_model = 'ResNet50'
  AND r.reference_model = 'ResNet50'  -- Correct recovery
GROUP BY j.n_train_triplets
ORDER BY j.n_train_triplets;

# 3. Find top-performing reference models for a specific data-generating model
SELECT 
    r.reference_model,
    AVG(r.test_accuracy) as mean_accuracy
FROM jobs j
JOIN reference_model_results r ON j.id = r.job_id
WHERE j.data_generating_model = 'ViT_Large'
  AND j.n_train_triplets = 100000
GROUP BY r.reference_model
ORDER BY mean_accuracy DESC
LIMIT 10;

# Exit SQLite
.quit
```

### Create Custom Analyses

You can write custom scripts using the database analysis tools:

```python
# custom_analysis.py
import hydra
from omegaconf import DictConfig
from tools.db_analysis import DBResultsAnalysis

@hydra.main(version_base=None, 
            config_path='scripts_configurations', 
            config_name="figures_1_and_3")
def analyze_custom(cfg: DictConfig):
    # Create analyzer instance
    analyzer = DBResultsAnalysis(
        db_path=cfg['full_W']['db_path'],
        data_generating_models_list=cfg.data_generating_models,
        reference_models_list=cfg.data_generating_models + cfg.new_models,
        n_train_triplets_list=cfg['full_W']['n_train_triplets_list'],
    )
    
    # Example: Get confusion matrix for largest training set
    data_gen_models, ref_models, acc_matrix = analyzer.confusion_matrix(
        n_train_triplets=100000
    )
    
    # Print results
    import pandas as pd
    df = pd.DataFrame(acc_matrix, index=data_gen_models, columns=ref_models)
    print(df)
    
    # Calculate precision/recall/F1 for each model
    metrics = analyzer.get_model_precision_recall_f1_accuracy(n_train_triplets=100000)
    for model, vals in metrics.items():
        print(f"{model}: Precision={vals['precision']:.3f}, Recall={vals['recall']:.3f}, F1={vals['f1']:.3f}")

if __name__ == "__main__":
    analyze_custom()
```

Run with:
```bash
python custom_analysis.py experiment_name="full_W"
```

### Modify Figure Appearance

Edit figure generation scripts to customize appearance:

```python
# Example: Change colormap in create_Figures_1_and_3.py

# Find the confusion matrix drawing section (~line 150):
# Original:
cmap = plt.cm.RdYlGn

# Change to:
cmap = plt.cm.viridis  # or 'plasma', 'inferno', etc.

# Adjust font sizes:
plt.rcParams['font.size'] = 14  # Increase from default 12
```

---

## Troubleshooting

### Common Errors

#### 1. Database File Not Found

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'Results/Simulations_experiments/full_W/full_W.db'
```

**Fix**:
- Verify database is downloaded and placed in correct location
- Check filename exactly matches (case-sensitive)
- Run `ls Results/Simulations_experiments/full_W/` to verify

#### 2. Empty or Incomplete Results

**Error**:
```
ValueError: No data found for experiment 'full_W'
```

**Fix**:
- Database may be empty or have no completed jobs
- Check job status: `sqlite3 <db_path> "SELECT status, COUNT(*) FROM jobs GROUP BY status;"`
- If jobs are 'pending' or 'running', database is incomplete; re-download

#### 3. Missing Hydra Config

**Error**:
```
ConfigKeyError: Missing key 'full_W' in config
```

**Fix**:
- Check that you're using correct experiment_name
- Valid names defined in `scripts_configurations/figures_1_and_3.yaml`
- Run with `--help` to see available options:
  ```bash
  python figures_scripts/create_Figures_1_and_3.py --help
  ```

#### 4. Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'hydra'
```

**Fix**:
- Ensure conda environment is activated:
  ```bash
  conda activate rethinking_alignment
  ```
- Or reinstall dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### 5. Memory Errors

**Error**:
```
RuntimeError: CUDA out of memory
```

**Fix**:
- Figure generation scripts don't use GPU; error suggests different script is running
- If modifying scripts, avoid loading large tensors unnecessarily
- Close other GPU processes

---

## Configuration Reference

### Experiment Names

Valid experiment names for `create_Figures_1_and_3.py`:
- `full_W` - Main experiment (unconstrained W matrix)
- `full_W_PCA_500` - With PCA-reduced features (Figure S3)
- `diagonal` - Diagonal constraint
- `zero_shot` - Zero-shot (identity matrix)

Check `scripts_configurations/figures_1_and_3.yaml` for full list.

### Database Paths

Default paths defined in configurations:
- `Results/Simulations_experiments/full_W/full_W.db`
- `Results/Simulations_experiments/full_W_PCA_500/full_W_PCA_500.db`
- etc.

Override via command line:
```bash
python figures_scripts/create_Figures_1_and_3.py \
    experiment_name=full_W \
    full_W.db_path="path/to/custom/database.db"
```

---

## Expected Outputs

After running all figure scripts, your `Plots/` directory should contain:

```
Plots/
├── full_W/
│   ├── Figure_1.pdf              # Main model recovery results
│   ├── Figure_3.pdf              # Ranking analysis
│   └── [additional diagnostic plots]
├── full_W_PCA_500/
│   ├── Figure_1.pdf              # Figure S3
│   └── Figure_3.pdf
├── Figure_2.pdf                   # (Requires additional data)
├── Figure_4.pdf                   # MDS projection
├── Figure_5.pdf                   # Flexibility-accuracy tradeoff
├── Figure_S1.pdf                  # Violin plots
└── Figure_S4.pdf                  # RSA heatmap
```

---

## Next Steps

- **Explore the data**: Query databases directly with SQL
- **Create custom analyses**: Use `DBResultsAnalysis` class in your own scripts
- **Modify figures**: Edit figure scripts to customize appearance
- **Full replication**: See `GUIDE_full_rerun_from_model_features.md` to rerun entire pipeline

---

