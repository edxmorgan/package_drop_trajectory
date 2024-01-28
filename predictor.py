import numpy as np
from drop_dynamics import Dynamics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from numbers_parser import Document
from scipy.optimize import curve_fit

class Model():
    def __init__(self,package_mass, g):
        self.dynamics = Dynamics(package_mass, g)

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
        ydata = data[1].reshape(-1)
        f = lambda x, theta0, theta1, theta2 ,delay : self.predict(x, theta0, theta1, theta2, delay).reshape(-1)
        popt, pcov = curve_fit(f, xdata, ydata, bounds=[(0.1,0.1, 0.1, 0),(10,10,10, 1)])
        print(f"popt: {popt}")
        print(f"pcov: {pcov}")
        # Evaluate the model on test data
        test_predictions = self.predict(data[2], popt[0], popt[1], popt[2], popt[3])

        # Calculate R^2 score
        r2 = r2_score(data[3], test_predictions)
        accuracy_percentage = r2 * 100
        print(f"Model Accuracy: {accuracy_percentage:.2f}%")

        return popt, pcov, accuracy_percentage


    def predict(self, ics, drag_area0, drag_area1, drag_area2, delay):
        """
        Predict the final positions for multiple initial conditions.
        
        Parameters:
        ics (numpy.ndarray): array where each row represents a set of initial conditions.
        drag_area0 (float): The drag area value to use for N.
        drag_area1 (float): The drag area value to use for E.
        drag_area2 (float): The drag area value to use for D.
        delay (float): how long it takes the package to come out of the plane.

        Returns:
        numpy.ndarray: A 2D array where each row represents the final position (North, East) for the corresponding set of initial conditions.
        """
        drag_area = [drag_area0, drag_area1, drag_area2]
        vectorized_simulate = np.vectorize(self.dynamics.simulate_package_drop, signature='(n),(k),()->(m)')
        final_positions = vectorized_simulate(ics, drag_area, delay)
        return final_positions


if __name__ == "__main__":
    model = Model(package_mass=1, g=9.81)
    df = model.load_data()
    data = model.preprocess_data(df)
    # model.estimate_parameters(data)
    
    #Example usage with multiple initial conditions
    i = 90
    ics = np.array([
        data[0][i,:],
        data[0][i-32,:]
    ])
    optimised_drag_area_0 = 0.28478375
    optimised_drag_area_1 = 0.29638052
    optimised_drag_area_2 = 0.57881585
    optimised_delay = 0.4
    final_positions = model.predict(ics, optimised_drag_area_0, optimised_drag_area_1, optimised_drag_area_2, optimised_delay)
    print("Predicted Final Positions (North, East):", final_positions)
    print("Actual Final Positions (North, East):", data[1][i,:],data[1][i-32,:])