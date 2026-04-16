# Quantum state tomography with gradient descent 

This repository contains the code required to reproduce results for bound entanglement optimization and analysis. It heavily leverages Quantum State Tomography concepts optimized via Gradient Descent, adapting methods from https://arxiv.org/abs/2503.04526.

## Modern Installation & Usage

This project has been updated to use modern Python packaging (PEP 621). 

1. Create a new virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the package in editable mode (this installs `qst_tec` and all dependencies):
   ```bash
   pip install -e .
   ```

## Project Structure
- `src/qst_tec/`: Core python library for gradient descent-based tomography and optimization.
- `notebooks/`: Jupyter Notebooks demonstrating bound entanglement state generation (`BEState1`, `BEState2`, `Bound`, `Bound UPB`).
- `data/`: Results and saved numpy states.
- `docs/papers/`: Reference literature.

## Tested Environment
- Python >= 3.10
- numpy
- qutip (5.0.0+)
- scipy
- seaborn
- torch
- jax
- cvxpy
This repository contains the code required to reproduce the plots from https://arxiv.org/abs/2503.04526  

## Installation and use

Install the myenvironment.yml or you can install the libraries by yourself, but be careful with the version of QuTip, in this project qutip==5.0.0 is used.

```python
conda env create -f environment.yml
conda activate applesilicon_gd_qst_env
```
The notebooks in the `examples` folder are tutorials on how to do the quantum state tomography with the different methods of gradient descent. 

The `data_and_paper-plots` folder contains a Jupyter Notebook (`.ipynb`) that runs the methods used to generate the plots presented in the paper, along with a ZIP file containing the corresponding data.
