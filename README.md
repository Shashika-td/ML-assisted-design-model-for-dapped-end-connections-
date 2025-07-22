# ML-Assisted Design Model for Dapped-End Connections

## Overview
This repository contains a Python script for an ML-assisted design model for dapped-end connections. The model takes target capacity, error margin, dapped-end depth, and span-to-effective-depth ratio as inputs and predicts the feasible reinforcement configuration along with the compressive strength of concrete required to achieve the target capacity.

The script generates random input data, applies engineering constraints, scales the inputs (if required), predicts the connection capacity using the trained ML model, and filters results based on an error threshold.

## Requirements

### Python Version:
- Python 3.12 or higher

### Libraries:-

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
## How to use

### 1. Save the trained ML model
First, you need to save the trained ML model to your local disk using the following code.
  ```bash
  import joblib

  # Save trained model
  joblib.dump(model, "trained_model.pkl")
  ```
If the Model is an ANN or Hybrid ANN:-

If your model is an Artificial Neural Network (ANN) or a Hybrid ANN, it was likely trained on scaled data. In this case, you must also save the corresponding scaler used to scale the dataset:
  ```bash
  # Save the scaler used during training
  joblib.dump(scaler, "scaler.pkl")
  ```
If the Model is a Classical ML Model:-

If you are using a classical ML model such as Extreme Gradient Boosting, Random Forest, or Decision Trees, and it was trained on non-scaled parameters, then there is no need to scale the input data before making predictions.

Note: Ensure that you update the model filename in the script if you use a different name than `trained_GA_ANN_model.pkl`.

### 2. Run the Script
`ML-Assisted Design Model for DE Connections.py`

Example output:-

```bash
Model loaded successfully!
Scaler loaded successfully!
10 design combinations found within the threshold range
Filtered results saved to Predicted_parameters.csv
```

## Contact

For any questions, please contact me via email at:- [dharmawanshadast.24@uom.lk](mailto:dharmawanshadast.24@uom.lk)
