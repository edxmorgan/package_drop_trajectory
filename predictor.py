from scipy.optimize import least_squares
import numpy as np
from drop_dynamics import Dynamics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numbers_parser import Document

class Model():
    def __init__(self,package_mass, g, Ve):
        self.dynamics = Dynamics(package_mass, g, Ve)

    def load_data(self, file="Package_drop_take_home_.numbers"):
        """ Load data from a file """
        doc = Document(file)
        sheets = doc.sheets
        tables = sheets[0].tables
        data = tables[0].rows(values_only=True)
        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    def preprocess_data(self, df:pd.DataFrame):
        """ Preprocess the data """
        df.columns = ['height','vel_n','vel_e','vel_d','wind_n','wind_e','wind_d','air_density','final_pos_n','final_pos_e']
        # Split data into training and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        # Prepare data for curve_fit
        # Example: experimental_data = [[height, aircraft_velocity, wind_velocity, air_density, package_mass, g], ...]
        # observed_positions = [[final_north, final_east], ...]

        train_experimental_data = train_df[['height','vel_n','vel_e','vel_d','wind_n','wind_e','wind_d', 'air_density']].to_numpy()
        train_observed_positions = train_df[['final_pos_n', 'final_pos_e']].to_numpy()

        test_experimental_data = test_df[['height','vel_n','vel_e','vel_d','wind_n','wind_e','wind_d', 'air_density']].to_numpy()
        test_observed_positions = test_df[['final_pos_n', 'final_pos_e']].to_numpy()

        return train_experimental_data, train_observed_positions, test_experimental_data, test_observed_positions
    
    def estimate_parameters(self, data):
        """ Estimate model parameters using least squares optimization """
        # Define the objective function for least squares optimization
        # for better optimisation methods, the jacobian information about the parameters can be used.
        def objective_function(x, exp, obs):
            # Predict using the dynamics model
            predictions = self.predict(exp, x)
            # Calculate residuals (difference between predictions and actual values)
            residuals = (obs[:,0]-predictions[:,0])**2 + (obs[:,1]-predictions[:,1])**2
            mean_residual_0 = np.mean(obs[:,0]-predictions[:,0], axis=0)
            mean_residual_1 = np.mean(obs[:,1]-predictions[:,1], axis=0)
            print(f"error mean 0:{mean_residual_0} 1:{mean_residual_1} ")
            return residuals
        
        x0 = np.array([1.0, 1.0]) #randomly chosen
        test_predictions = self.predict(data[2], x0)
        mse = mean_squared_error(data[3], test_predictions)
        print(f"initial Test MSE: {mse}")

        result = least_squares(fun=objective_function, x0=x0, args=(data[0], data[1]), bounds=[(0.1, 0.1), (10, 10)])
        optimized_params = result.x

        # Calculate the covariance matrix
        jacobian = result.jac
        residual_variance = np.var(result.fun)
        covariance_matrix = np.linalg.inv(jacobian.T.dot(jacobian)) * residual_variance

        # Evaluate the model on test data
        test_predictions = self.predict(data[2], optimized_params)
        mse = mean_squared_error(data[3], test_predictions)

        print(f"residual_variance: {residual_variance}")
        print(f"Optimized Parameters: {optimized_params}")
        print(f"Covariance Matrix: \n{covariance_matrix}")
        print(f"Test MSE: {mse}")

        return optimized_params, covariance_matrix


    def predict(self, ics, drag_area):
        """
        Predict the final positions for multiple initial conditions.
        
        Parameters:
        ics (numpy.ndarray): array where each row represents a set of initial conditions.
        drag_area (float): The drag area value to use for all predictions.

        Returns:
        numpy.ndarray: A 2D array where each row represents the final position (North, East) for the corresponding set of initial conditions.
        """
        # print(drag_area)
        vectorized_simulate = np.vectorize(self.dynamics.simulate_package_drop, signature='(n),(k)->(m)')
        final_positions = vectorized_simulate(ics, drag_area)
        return final_positions


if __name__ == "__main__":
    model = Model(package_mass=1, g=9.81, Ve=5.0) #Ve = descent velocity
    df = model.load_data()
    data = model.preprocess_data(df)
    model.estimate_parameters(data)
    
    #Example usage with multiple initial conditions
    # ics = np.array([
    #     [39.5   , -14.55  ,  14.47    , 1.32,     0.18 ,   -1.64  ,   0.  ,     1.1441]
    # ])
    # optimised_drag_area = [0.254793, 0.22632925]
    # final_positions = model.predict(ics, optimised_drag_area)
    # print("Final Positions (North, East):", final_positions)