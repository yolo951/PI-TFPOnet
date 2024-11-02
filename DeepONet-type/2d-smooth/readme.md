
# Deep Learning Project for Matrix Analysis

This repository contains the implementation of a deep learning model for matrix analysis, including data generation, training, and visualization components.

## Project Structure

### Data Generation and Storage
- `generate_data.py`: Script for generating all required data, including:
  - Sparse matrix U components (`index_of_u.npy`, `val_of_u.npy`)
  - Vector data files (`vectorB.npy`, `vectorC.npy`)
  - Matrix data files (`matrixf.npy`, `matrixU.npy`, `matrixup.npy`)

### Model Components
- `deeponet.py`: Implementation of the deep neural network architecture
- `ionet.py`: Input-output network implementation
- `dim2_cnn.py`: 2D CNN model implementation

### Training and Testing
- `train.py`: Main training script that:
  - Trains the model
  - Generates visualization plots
  - Saves trained model parameters
  - Outputs figures used in the paper
- `test.py`: Testing and evaluation script

### Output Files
- Training visualizations:
  - `2d_smooth_error.png`
  - `2d_smooth_ground.png`
  - `2d_smooth_l2.png`
  - `2d_smooth_loss.png`
  - `2d_smooth_refine.png`

### Model and Training History
- `model_state.pt`: Saved model parameters
- `loss_history.npy`: Training loss history
- `rel_l2_history.npy`: Relative L2 error history

## Usage

1. Generate required data:
```bash
python generate_data.py
```
This will create all necessary numpy arrays for training.

2. Train the model:
```bash
python train.py
```
This will:
- Train the model using the generated data
- Save training progress
- Generate visualization plots
- Save the trained model

3. The training process will automatically generate visualization plots that were used in the paper.

## Data Description

- `index_of_u.npy` and `val_of_u.npy`: Store the sparse matrix U in COO format
- Matrix files store different components of the problem setup
- Vector files contain related vector data for the computations

## Notes

- All PNG files are visualization outputs used in the research paper
- The model state is automatically saved during training
- Training history and metrics are preserved in numpy files for future reference

