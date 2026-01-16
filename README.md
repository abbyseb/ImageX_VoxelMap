# Voxelmap Tutorial - Patient P1 (SPARE Dataset)

## Overview

This notebook is a tutorial which implements a deep learning-based motion tracking system for radiotherapy treatment. It trains **Network A** to predict deformation vector fields (DVFs) from X-ray projections and CT volumes, enabling real-time tumor tracking during radiation therapy. This work and code is from Dr Nicholas Hindley's VoxelMap : https://github.com/Image-X-Institute/Voxelmap

**Key Features:**
- Train a neural network to predict 3D motion from 2D projections
- Generate target tumor volumes (ITVs/PTVs) for all treatment projections
- Evaluate motion tracking accuracy across multiple metrics
- Validate prediction quality using Dice scores, Jacobian determinants, and image similarity

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Training Workflow](#training-workflow)
3. [Testing Workflow](#testing-workflow)
4. [File Structure](#file-structure)
5. [Code Sections](#code-sections)
6. [Dependencies](#dependencies)
7. [Expected Outputs](#expected-outputs)

---

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX A6000)
- Python 3.8+
- PyTorch with CUDA
- Access to SPARE dataset directory structure

### Quick Start

1. **Update paths** in the notebook:
   - Set your SPARE dataset path (default: `/srv/shared/SPARE`)
   - Set your working directory 

2. **Configure patient and dataset**:
   - Patient: `P1`
   - Dataset: `SPARE`
   - Training: Day 1, No Scatter
   - Testing: Day 2, With Scatter

3. **Run training** (approx 17 min for 1 epoch)

4. **Run testing** to evaluate performance

---

## Training Workflow

### 1. Imports
**Find:** Search for `# imports` in the notebook

Loads essential libraries for neural network training, data processing, and visualization.

**Key imports:**
- `network_a`: Custom neural network architecture
- `losses`: Custom loss functions (flow mask, centroid, Dice, Jacobian)
- `spatialTransform`: DVF-based spatial transformation utilities
- `utilities.helper`: Helper functions 

### 2. File Names Configuration
**Find:** Search for `Patient_number = "P1"` (first occurrence)

Sets up data paths and experiment naming conventions.

**Configuration:**
- `Patient_number = "P1`: Patient identifier
- `NoScatter = True`: Uses scatter-free projection data
- `DayofTreatment = 1`: Training on day 1 data
- `NetworkType = "A"`: Network architecture variant

**Generated paths:**
- Input: `/srv/shared/SPARE/MC_V_P1_NS_01`
- Weights: `SPARE_A_P1_weights/`
- Plots: `SPARE_A_P1_plots/`

### 3. Dataset Setup
**Find:** Search for `class SupervisedDataset(Dataset):`

Implements `SupervisedDataset` class for loading training data.

**Data loading per sample:**
- **Source projection** (phase 06 reference): 2D X-ray image `06_Proj_XXXXX_bin.npy`
- **Target projection** (variable phase): 2D X-ray image `XX_Proj_XXXXX_bin.npy`
- **Source volume** (phase 06): 3D CT volume `subCT_06_mha.npy`
- **Source abdomen mask**: Thoracoabdominal region `sub_Abdomen_mha.npy`
- **Target DVF**: Ground-truth deformation field `DVF_XX_mha.npy`

**Normalization:** All images normalized to [0, 1] range using min-max scaling.

### 4. Data Visualization
**Find:** Search for `plt.subplot(1,3,1)` (first visualization)

Displays example data to verify loading:
- Source projection (2D)
- Source volume (mid-axial slice)
- Target DVF (SI component, mid-axial slice)

### 5. Projection Similarity Check
**Find:** Search for `def get_numpy_similarity`

Computes cosine similarity between no-scatter and scatter projections to quantify anatomical/scatter differences.


### 6. Training Loop
**Find:** Search for `for epoch in range(start_epoch, epoch_num + 1):`

Implements resumable training with automatic checkpointing.

**Training parameters:**
- Batch size: 8
- Learning rate: 1e-5
- Optimizer: Adam
- Train/Val split: 90/10
- Target epochs: 60

**Saved outputs:**
- `SPARE_A_P1_ckpt.pth`: Resume checkpoint (full state)
- `SPARE_A_P1_best.pth`: Best validation weights (inference)
- `SPARE_A_P1_TRAINvsVAL.png`: Loss curve visualization

**Resume capability:** Automatically loads checkpoint if training interrupted.

---

## Testing Workflow

### 7. Testing File Configuration
**Find:** Search for `DayofTreatment = 2` (second occurrence)

Switches to **treatment day 2 with scatter** for out-of-domain testing.

**Key changes:**
- `DayofTreatment = 2`
- `Scatter = True` (tests generalization to scatter-corrupted data)

### 8. Target ITV/PTV Generation
**Find:** Search for `def generate_target_itv_ptv_10bins`

Generates ground-truth tumor volumes for all 680 treatment projections.

**Process:**
1. Load source PTV (`sub_PTV_mha.npy`)
2. Load DVF for each breathing phase (bins 1-10)
3. Warp source PTV using spatial transformer
4. Map breathing phases to projections via `RespBin.csv`

**Output:** `Target_ITVs_ALL_SPARE_A_P1/Target_XXXXX_ITV_PTV.npy` (680 files)

**Why needed:** Provides ground-truth for Dice score and centroid tracking evaluation.

### 9. Testing Dataset Class
**Find:** Search for `class validateSPAREDataset(Dataset):`

Implements `validateSPAREDataset` for test-time data loading.

**Key differences from training:**
- Loads data from day 2 (scatter)
- Uses `RespBin.csv` to select correct breathing phase volume
- Includes ground-truth target PTVs for evaluation
- Returns gantry angles for temporal analysis

### 10. Model Loading
**Find:** Search for `model.load_state_dict(torch.load(PATH`

Loads trained Network A weights for inference.

**Configuration:**
- Device: `cuda:1`
- Weights: `SPARE_A_P1_weights/SPARE_A_P1_best.pth`
- Mode: `eval()` (disables dropout/batch norm training behavior)

### 11. Testing Loop
**Find:** Search for `for i, data in enumerate(testloader, 0):`

Evaluates model on all 680 treatment projections.

**Computed metrics:**

#### Motion Tracking (Primary):
- **Centroid shifts** (LR/SI/AP in mm): Ground-truth vs predicted tumor displacement
- Uses `centroid_shift_mm` utility to convert voxel centroids to physical coordinates

#### Overlap Quality:
- **Dice coefficient**: Volumetric overlap between ground-truth and predicted PTV
- Range: [0, 1], higher is better (1 = perfect overlap)

#### Deformation Regularity:
- **Jacobian determinant violations**: Percentage of voxels with det(J) ≤ 0
- Indicates non-physical folding in predicted DVF
- Lower is better (0% = fully diffeomorphic)

#### Image Quality (Secondary):
- **RMSE**: Root mean squared error of warped volume
- **SSIM**: Structural similarity index (perceptual quality)
- **PSNR**: Peak signal-to-noise ratio

### 12. Results Saving and Visualization
**Find:** Search for `np.save(os.path.join(out_dir, "test_angles.npy")`

Saves metrics and generates motion tracking plots.

**Saved files:**
- `test_angles.npy`: Gantry angles for each projection
- `tar_lr_mm.npy`, `tar_si_mm.npy`, `tar_ap_mm.npy`: Ground-truth motion
- `pred_lr_mm.npy`, `pred_si_mm.npy`, `pred_ap_mm.npy`: Predicted motion
- `test_dice.npy`: Dice scores
- `test_detJ.npy`: Jacobian violation ratios
- `test_mse.npy`, `test_ssim.npy`, `test_psnr.npy`: Image quality metrics

**Visualization:**
- 3-panel plot showing LR/SI/AP displacement vs gantry angle
- Compares ground-truth (solid) vs prediction (dashed)
- Saved as `SPARE_A_P1_trace.png`

---

## File Structure

### Input Data Files

#### Training Data (Day 1, No Scatter)
Located in: `/srv/shared/SPARE/MC_V_P1_NS_01/`

| File Pattern | Description | Dimensions |
|-------------|-------------|------------|
| `XX_Proj_XXXXX_bin.npy` | X-ray projections (phases 01-10) | 128×128 |
| `06_Proj_XXXXX_bin.npy` | Reference projection (phase 06) | 128×128 |
| `subCT_XX_mha.npy` | CT volumes (phases 01-10) | 128×128×128 |
| `subCT_06_mha.npy` | Reference CT volume (phase 06) | 128×128×128 |
| `DVF_XX_mha.npy` | Ground-truth deformation fields | 128×128×128×3 |
| `sub_Abdomen_mha.npy` | Thoracoabdominal mask | 128×128×128 |
| `sub_PTV_mha.npy` | Planning target volume (source) | 128×128×128 |

#### Testing Data (Day 2, With Scatter)
Located in: `/srv/shared/SPARE/MC_V_P1_SC_02/`

| File Pattern | Description | Purpose |
|-------------|-------------|---------|
| `XX_Proj_XXXXX_bin.npy` | Treatment projections (680 total) | Network input |
| `06_Proj_XXXXX_bin.npy` | Reference projection (in `source/`) | Network input |
| `subCT_XX_mha.npy` | Treatment day CT phases | Ground-truth volume |
| `subCT_06_mha.npy` | Reference CT volume | Network input |
| `source/itv_PTV_mha.npy` | Planning target volume | Deformation source |
| `RespBin.csv` | Breathing phase per projection (680×1) | Phase mapping |
| `Angles.csv` | Gantry angles per projection (680×1) | Temporal analysis |

### Output Files

#### Training Outputs
Located in: `SPARE_A_P1_weights/` and `SPARE_A_P1_plots/`

| File | Description | Size |
|------|-------------|------|
| `SPARE_A_P1_ckpt.pth` | Full checkpoint (resumable) | ~200 MB |
| `SPARE_A_P1_best.pth` | Best validation weights | ~100 MB |
| `SPARE_A_P1_TRAINvsVAL.png` | Loss curve plot | ~50 KB |

#### Testing Outputs
Located in: `SPARE_A_P1_plots/SPARE_A_P1_Experiment/`

| File | Description | Shape |
|------|-------------|-------|
| `test_angles.npy` | Gantry angles | (680,) |
| `tar_lr_mm.npy` | Ground-truth LR motion | (680,) |
| `tar_si_mm.npy` | Ground-truth SI motion | (680,) |
| `tar_ap_mm.npy` | Ground-truth AP motion | (680,) |
| `pred_lr_mm.npy` | Predicted LR motion | (680,) |
| `pred_si_mm.npy` | Predicted SI motion | (680,) |
| `pred_ap_mm.npy` | Predicted AP motion | (680,) |
| `test_dice.npy` | Dice coefficients | (680,) |
| `test_detJ.npy` | Jacobian violations | (680,) |
| `test_mse.npy` | RMSE values | (680,) |
| `test_ssim.npy` | SSIM values | (680,) |
| `test_psnr.npy` | PSNR values | (680,) |
| `SPARE_A_P1_trace.png` | Motion tracking plot | Image |

#### Intermediate Files
Located in: `Target_ITV_SPARE_A_P1/` and `Target_ITVs_ALL_SPARE_A_P1/`

| File Pattern | Description | Usage |
|-------------|-------------|-------|
| `Target_ITV_PTV_XX_mha.npy` | Target PTVs for phases 01-10 | Intermediate |
| `Target_XXXXX_ITV_PTV.npy` | Target PTVs for all 680 projections | Testing ground-truth |

---

## Dependencies

### Custom Modules

Located in `utilities/` directory (not shown in notebook):

| Module | Key Components | Purpose |
|--------|---------------|---------|
| `network_a.py` | `model()` class | Network A architecture (3D U-Net + VoxelMorph) |
| `losses.py` | `flow_mask()`, `centroid_ptv()`, `dice()`, `jacobian_determinant()` | Custom loss and metric functions |
| `spatialTransform.py` | `Network()` class | DVF-based spatial transformation (warping) |
| `helpers.py` | `centroid_shift_mm()` | Coordinate conversion utilities |
| `continue_training.py` | `save_checkpoint()`, `load_checkpoint_if_available()`, `plot_losses()` | Training state management |

---

## Expected Outputs

### Training (15hours on RTX A6000)

**Loss curves should show:**
- Steady decrease in training loss
- Validation loss plateaus around epoch 40-50
- Typical final losses after 50 epochs: Train ~0.0119, Val ~0.0101

---

---

## Coordinate System

**Anatomical directions:**
- **LR (Left-Right)**: X-axis, lateral motion
- **SI (Superior-Inferior)**: Y-axis, breathing motion (largest displacement)
- **AP (Anterior-Posterior)**: Z-axis, forward-backward motion
