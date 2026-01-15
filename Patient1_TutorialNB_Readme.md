# Voxelmap Tutorial - Patient P1 (SPARE Dataset)

## Overview

This notebook is a tutorial which implements a deep learning-based motion tracking system for radiotherapy treatment. It trains **Network A** to predict deformation vector fields (DVFs) from X-ray projections and CT volumes, enabling real-time tumor tracking during radiation therapy.

**Key Features:**
- Train a neural network to predict 3D motion from 2D projections.
- Generate target tumor volumes (ITVs/PTVs) for all treatment projections.
- Evaluate motion tracking accuracy across multiple metrics.
- Validate prediction quality using Dice scores, Jacobian determinants, and image similarity.

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

- NVIDIA GPU with CUDA support (tested on RTX A6000).
- Python 3.8+.
- PyTorch with CUDA.
- Access to SPARE dataset directory structure.

### Quick Start

1. **Update paths** in the notebook:
   - Set your SPARE dataset path (default: `/srv/shared/SPARE`).
   - Set your working directory (default: `/home/saatwik/Desktop/voxelmap/Voxelmap-main/`).

2. **Configure patient and dataset**:
   - Patient: `P1`.
   - Dataset: `SPARE`.
   - Training: Day 1, No Scatter (`MC_V_P1_NS_01`).
   - Testing: Day 1, No Scatter.

3. **Run the cells sequentially** to train and evaluate the model.

---

## Training Workflow

The training process involves:
1. Loading 3D CT volumes and 2D X-ray projections.
2. Training Network A (3D U-Net base) to predict the DVF that maps the source volume to the target state.
3. Saving model checkpoints and training logs.

---

## Testing Workflow

The testing process involves:
1. Loading the trained model checkpoint.
2. Predicting DVFs for the testing projection set.
3. Warping the planning PTV/ITV using the predicted DVF.
4. Calculating metrics: Dice Coefficient, Centroid Shift, and Jacobian Determinant.

---

## File Structure

### Data Inputs (SPARE Dataset)
Located in: `/srv/shared/SPARE/MC_V_P1_NS_01/`

| File Pattern | Description |
|--------------|-------------|
| `subCT_06_mha.npy` | Source CT Volume (Reference) |
| `sub_Abdomen_mha.npy` | Patient Abdomen Mask |
| `XX_bin.npy` | 2D X-ray Projections (Phases 01-10) |
| `DVF_XX_mha.npy` | Ground Truth 3D DVFs |

### Intermediate Files
Located in: `Target_ITV_SPARE_A_P1/` and `Target_ITVs_ALL_SPARE_A_P1/`

| File Pattern | Description | Usage |
|-------------|-------------|-------|
| `Target_ITV_PTV_XX_mha.npy` | Target PTVs for phases 01-10 | Intermediate |
| `Target_XXXXX_ITV_PTV.npy` | Target PTVs for all projections | Testing ground-truth |

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

### Training (Estimated 15 hours on RTX A6000)

**Loss curves should show:**
- Steady decrease in training loss.
- Validation loss plateaus around epoch 40-50.
- Typical final losses after 50 epochs: Train ~0.0119, Val ~0.0125.

### Evaluation Metrics
- **Dice Score:** Expected > 0.85 for successful tracking.
- **Centroid Shift:** Should be minimized (measured in mm).
- **Jacobian Determinant:** Should remain positive (0.5 to 1.5) to ensure topology preservation.
