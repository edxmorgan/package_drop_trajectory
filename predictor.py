from scipy.optimize import least_squares
import numpy as np
from drop_dynamics import Dynamics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numbers_parser import Document
from scipy.optimize import curve_fit

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
        #Use non-linear least squares to fit a function, f, to data.
        xdata = data[0]
        ydata = data[1].flatten()
        f = lambda x, theta0, theta1, Ve : self.predict(x, theta0, theta1, Ve).flatten()
        popt, pcov = curve_fit(f, xdata, ydata, bounds=[(0.1,0.1, 0.1),(10,10,10)])
        print(f"popt: {popt}")
        print(f"pcov: {pcov}")
        # Evaluate the model on test data
        test_predictions = self.predict(data[2], popt[0], popt[1], popt[2])
        mse = mean_squared_error(data[3], test_predictions)
        print(f"Test MSE: {mse}")


    def predict(self, ics, drag_area0, drag_area1, Ve):
        """
        Predict the final positions for multiple initial conditions.
        
        Parameters:
        ics (numpy.ndarray): array where each row represents a set of initial conditions.
        drag_area0 (float): The drag area value to use for N.
        drag_area1 (float): The drag area value to use for E.
        Ve (float): steady state descent velocity.

        Returns:
        numpy.ndarray: A 2D array where each row represents the final position (North, East) for the corresponding set of initial conditions.
        """
        # print(drag_area)
        drag_area = [drag_area0, drag_area1]
        vectorized_simulate = np.vectorize(self.dynamics.simulate_package_drop, signature='(n),(k),()->(m)')
        final_positions = vectorized_simulate(ics, drag_area, Ve)
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
    # optimised_drag_area_0 = 0.254793
    # optimised_drag_area_1 =  0.22632925
    # final_positions = model.predict(ics, optimised_drag_area_0, optimised_drag_area_1)
    # print("Final Positions (North, East):", final_positions)

# popt: [0.28564251 0.29699216 5.54636804]
# pcov: [[ 1.54661599e-04  7.35457580e-05 -2.26487487e-03]
#         [ 7.35457580e-05  1.63185816e-04 -2.26559741e-03]
#         [-2.26487487e-03 -2.26559741e-03  1.36998541e-01]]
# Test MSE: 41.588051680346915