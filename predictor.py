from scipy.optimize import curve_fit
import numpy as np
from dynamics import simulate_package_drop
import pandas as pd
from numbers_parser import Document

class Model():
    def __init__(self,package_mass, g, drag_area):
        self.package_mass = package_mass
        self.g = g
        self.drag_area = drag_area

    def estimate_parameters(self):
        """
            estimates drag_area
        """
        doc = Document("Package_drop_take_home_.numbers")
        sheets = doc.sheets
        tables = sheets[0].tables
        data = tables[0].rows(values_only=True)
        df = pd.DataFrame(data[1:], columns=data[0])
        df.columns = ['height','vel_n','vel_e','vel_d','wind_n','wind_e','wind_d','air_density','final_pos_n','final_pos_e']
        df.loc[df.n == "d", ['height','b']].values.flatten().tolist()

        # Prepare your data for curve_fit
        # Example: experimental_data = [[height, aircraft_velocity, wind_velocity, air_density, package_mass, g], ...]
        # observed_positions = [[final_north, final_east], ...]

        # Use curve_fit to estimate the drag coefficient
        # popt, pcov = curve_fit(model_function, experimental_data, observed_positions, bounds=[(0.1,0.1),(10, 10)])

        # Estimated drag coefficient
        # estimated_drag_coef = popt[0]

    def predict(self, height, aircraft_velocity, wind_velocity, air_density):
        try:
            ic = height, aircraft_velocity, wind_velocity, air_density, self.package_mass, self.g 
            final_position = simulate_package_drop(ic, self.drag_area)
            return final_position
        except ValueError as e:
            print(e)
        return None


if __name__ == "__main__":
    model = Model(package_mass=1, g=9.81, drag_area=0.5)
    height = 100
    aircraft_velocity = [0, 0, 0]
    wind_velocity = [1, 1, 0]
    air_density = 1.225
    final_position = model.predict(height, aircraft_velocity, wind_velocity, air_density)
    print("Final Position (North, East):", final_position)