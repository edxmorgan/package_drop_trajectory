from model import Model
import numpy as np


model = Model(package_mass=1, g=9.81)

# model parameters identification
df = model.load_data("Package_drop_take_home_.numbers")
data = model.preprocess_data(df)
model.estimate_parameters(data)

####### RESPONSE FOR PARAMETER ESTIMATION ####################################

# performing non-linear least squares to fit physics dynamics to data.
# estimating...
# estimation done
# popt: [4.35458957 8.79336404 0.61662788 0.5       ]
# pcov: [[ 2.94884772e+00 -6.02302268e-01 -1.83351186e-03  0.00000000e+00]
#  [-6.02302268e-01  1.98475854e+01  3.56022669e-03  0.00000000e+00]
#  [-1.83351186e-03  3.56022669e-03  7.60176136e-03  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]]
# Evaluating the model on test data
# Model r2_score: 83.71%

################################################################################


## Example usage with multiple initial conditions
popt = np.array([4.35458957, 8.79336404, 0.61662788, 0.5])
pcov: np.array([[2.94884772e+00, -6.02302268e-01, -1.83351186e-03, 0.00000000e+00],
                [-6.02302268e-01,  1.98475854e+01,  3.56022669e-03, 0.00000000e+00],
                [-1.83351186e-03,  3.56022669e-03,  7.60176136e-03, 0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00]])

ics = np.array([37.9, 14.75,  -14.49,  -1.05, 0.85,    1.54,     0.,      1.0703])
final_positions = model.predict(ics, popt[0], popt[1], popt[2], popt[3])
print("Predicted Final Positions (North, East):", final_positions)
print("Actual Final Positions (North, East):", [14.21, 4.7])

# uncertainty of the prediction can be computed using the formulation below
# SIGMA_F = J @ SIGMA_x @ J^T

# where the J is the jacobian of the system wrt the parameters
# SIGMA_x is the covariance matrix of the parameters

# would require linearizing the model to x_d = Ax to get J from A