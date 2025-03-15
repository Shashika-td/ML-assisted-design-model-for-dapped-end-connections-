# ML-Assisted Design Model for Dapped-End Connections

## Overview
This repository contains a Python script for an ML-assisted design model for dapped-end connections. The model takes target capacity, error margin, dapped-end depth, and span-to-effective-depth ratio as inputs and predicts the feasible reinforcement configuration along with the compressive strength of concrete required to achieve the target capacity.

The script generates random input data, applies engineering constraints, scales the inputs (if required), predicts the connection capacity using the trained ML model, and filters results based on an error threshold.

## Requirements

### Python Version:
- Python 3.12 or higher

### Libraries:

The following libraries are required to run the Python script. Each library can be installed by running the following commands in your command prompt:

- `joblib`: Install using
  ```bash
  pip install joblib
  ```
- `numpy`: Install using
  ```bash
  pip install numpy
  ```
  - `pandas`: Install using
  ```bash
  pip install pandas
  ```
