### README Documentation for Package Drop Prediction Model

#### Introduction
This documentation describes the implementation of a package drop prediction model, used for determining the landing point of a package dropped from an aircraft. The model takes into account various flight and environmental parameters to estimate the final position of the package relative to the aircraft's position at the time of release.

#### File Structure
The implementation is organized into three main files:
1. `main.py`: The primary script for model usage and testing.
2. `model.py`: Contains the `Model` class, encapsulating the physics-based prediction model.
3. `drop_dynamics.py`: Provides functions for calculating the dynamics of the dropping package and includes commented sections detailing the methodology of the approach.

#### `main.py` Overview
- **Model Initialization**: The `Model` class from `model.py` is instantiated with parameters including package mass and gravitational acceleration.
- **Parameter Estimation**: The script initially includes code (commented out) for parameter estimation using the provided dataset. This process involves loading the data, preprocessing it, and estimating model parameters through non-linear least squares fitting.
- **Model Evaluation**: The estimated parameters and their covariance are displayed, followed by an evaluation of the model's accuracy using an R-squared score.
- **Example Usage**: The script demonstrates how to use the model with a set of initial conditions, predicting the final north and east positions of the package. It also compares the predicted positions with actual data.

#### `model.py` Overview
This file defines the `Model` class, which includes the following key components:
- **Data Loading and Preprocessing**: Functions for loading the dataset and preprocessing it for model fitting.
- **Parameter Estimation**: Implements the parameter estimation using non-linear least squares.
- **Prediction Method**: A method for predicting the final position of the package given initial conditions and estimated parameters.

#### `drop_dynamics.py` Overview
Contains functions that calculate the dynamics of the package during its drop. This includes accounting for factors like drag, wind velocity and package ejection delay. The methodology of the approach is detailed in commented sections within this file.

#### Usage
To use this model:
1. Instantiate the `Model` class.
2. Optionally perform parameter estimation using your data.
3. Use the `predict` method of the `Model` class to estimate the final position of a package given a set of initial conditions.

---

Feel free to reach out with any questions or if further clarification is needed on any part of this implementation.
