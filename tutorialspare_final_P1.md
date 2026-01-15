# Voxel Map Training/Testing Model A Pipeline for SPARE DATA: Patient P1 

This repository contains the `tutorialspare_final_P1.ipynb` notebook, which implements a deep learning workflow for 3D medical image analysis. The pipeline is designed specifically for **Patient P1** and handles data loading, model training (with resume capabilities), and comprehensive evaluation using medical imaging metrics.

---

## Table of Contents
- Prerequisites  
- Project Structure  
- Configuration  
- Step 1: Training the Model  
- Step 2: Testing and Evaluation  
- Outputs and Metrics  

---

## Prerequisites

### Python Environment
Ensure the following libraries are installed:

- torch (PyTorch)  
- numpy  
- matplotlib  
- scikit-image (skimage)  
- importlib  

### Custom Modules
The notebook relies on the following local modules, which must be available in the working directory:

- `network_a.py`  
  Defines the neural network architecture.

- `continue_training.py`  
  Implements the training loop, loss computation, checkpointing, and plotting utilities.

- `utilities/` (package)  
  - `spatialTransform`: 3D spatial transformation logic  
  - `losses`: Custom loss functions (Dice, Centroid PTV, Jacobian)  
  - `helpers`: Utility functions such as `centroid_shift_mm`  

### Data Directory
By default, the notebook expects data to be located at:

```
/srv/shared/SPARE/
```

Ensure the dataset (for example, `MC_V_P1_NS_01`) is available at this location, or update the `im_dir` variables in the notebook accordingly.

---

## Project Structure

- **Training Data**  
  Patient P1, Day 1, No Scatter  
  `MC_V_P1_NS_01`

- **Testing Data**  
  Patient P1, Day 2, With Scatter  
  `MC_V_P1_SC_02`

- **Model Checkpoints**  
  Saved to:  
  ```
  {DatasetType}_{NetworkType}_{Patient_number}_weights/
  ```

---

## Configuration

Before running the notebook, verify the global variables defined in **Cell 4**.

### Key Variables

- `Patient_number = "P1"`  
- `DatasetType = "SPARE"`  
- `NetworkType = "A"`  

#### Scatter / No Scatter
- Training:  
  ```
  NoScatter = True
  Scatter = False
  ```
- Testing:  
  ```
  Scatter = True
  ```

#### Day of Treatment
- Training: `DayofTreatment = 1`  
- Testing: `DayofTreatment = 2`  

---

## Step 1: Training the Model

The training pipeline supports interruption and resumption via checkpoints.

### 1.1 Data Setup
Run **Cells 1–15**.

- Initializes the `SupervisedDataset` class  
- Loads paired 3D volumes  

**Parameters**
- Batch size: 8  
- Image size: 128 × 128 × 128  

### 1.2 Model Initialization
Run **Cell 16**.

- Builds the network architecture  
- Moves the model to GPU (`cuda:1`) if available  
- Uses `int_steps = 10` for the integration layer  

### 1.3 Training Execution
Run **Cell 19**.

- Starts the training loop  
- Supports resume training via:
  ```
  continue_Training = True
  ```

**Checkpointing**
- Latest checkpoint: `*_ckpt.pth`  
- Best model (lowest validation loss): `*_best.pth`  

**Outputs**
- Loss curves saved to:
  ```
  *_plots/
  ```

---

## Step 2: Testing and Evaluation

Testing is performed on an independent dataset to evaluate generalization.

### 2.1 Update Configuration for Testing
Go to **Cell 22** (Header: *Setting File Names*).

The notebook automatically updates:
- `DayofTreatment = 2`  
- `Scatter = True`  

Run this cell to update the testing paths.

### 2.2 Generate Ground Truth PTV
Run **Cells 24–25**.

- Pre-computes the Target Planning Target Volume (PTV)  
- Processes all 680 projections  
- Uses deformation fields to warp the source PTV into target space  

### 2.3 Run Inference
Run **Cells 29–31**.

- **Cell 29**: Re-initializes the test dataloader (`batch_size = 1`)  
- **Cell 30**: Loads the spatial transformation network  
- **Cell 31**: Main inference loop  
  - Predicts deformation fields  
  - Warps source images and PTVs  
  - Computes evaluation metrics  

---

## Outputs and Metrics

During testing (Cell 31), the following metrics are computed per projection.

### Geometric Accuracy
- **Centroid Shift (mm)**  
  Measures the physical displacement between predicted and ground truth tumor centroids.
  - Left–Right (LR)  
  - Superior–Inferior (SI)  
  - Anterior–Posterior (AP)  

- **Jacobian Determinant**  
  Detects non-physical deformations.  
  Values below zero indicate foldings in the deformation field.

### Overlap and Image Quality
- **Dice Coefficient**  
  Measures volumetric overlap between predicted and ground truth PTVs (0 to 1).

- **SSIM**  
  Structural Similarity Index for image quality assessment.

- **PSNR**  
  Peak Signal-to-Noise Ratio for reconstruction fidelity.

---

## Final Outputs

- **CSV Files**  
  Metric arrays such as:
  - `test_dice`  
  - `pred_lr`, `pred_si`, `pred_ap`  

  These are exported for downstream statistical analysis.

- **Saved Weights**  
  Final and intermediate model weights are stored in:
  ```
  {DatasetType}_{NetworkType}_P1_weights/
  ```

---

