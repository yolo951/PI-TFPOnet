
# implementation

This repository contains the implementation of models (our model and baseline models) in our paper.

## Project Structure
### Data Generation and Storage
- `generate_data.py`: Script for generating all required data, including:
  - Input $f$ on the sparse grid (`data['f_total']`, `f.npy`, `f_centor.npy`) and on the fine grid (`data_fno['f_test_fine']`)
  - Ground truth $u$ on the sparse grid (`data['up_total']`, `data_fno['u_train_sparse']`) and on the fine grid (`data['u_test_fine']`)
  - Sparse matrix U components (`data['index']`, `data['val']`)
  - Vector data files (`data['B_total']`, `data['C_total']`)

### Baseline Methods
- `deeponet.py`: Implementation of DeepONet and PI-DeepONet (baseline method)
- `ionet.py`: Implementation of IO-Net and PI-IONet (baseline method)
- `fno.py`: Implementation of FNO and PINO (baseline method)
- `FNO1d.py`: Core components of the one-dimensional FNO model
- `FNO2d.py`: Core components of the two-dimensional FNO model
- `GeoFNO2d.py`: Core components of the two-dimensional Geo-FNO model

### Our Model
- `train.py`(1d-smooth, 1d-singular, 1d-high-constrast, 2d-smooth) or `tfponet_fno.py` and `tfponet_mlp.py`(2d-singular and 2d-L-shaped): Main training script that:
  - Trains the model
  - Generates visualization plots
  - Saves trained model parameters
  - Outputs figures used in the paper

#### Output Files
- Training visualizations (以2d-smooth为例):
  - `2d_smooth_compare.png`
  - `2d_smooth_l2.png`
  - `2d_smooth_loss.png`
  - `2d-smooth_all_rel_l2_history.png`
  - `2d-smooth_all_rel_l2_time.png`

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

