from scipy.optimize import curve_fit
import numpy as np
from dynamics import simulate_package_drop
import pandas as pd
from numbers_parser import Document

class Model():
    def __init__(self):
        pass

    def identificate(self):
        doc = Document("Package_drop_take_home_.numbers")
        sheets = doc.sheets
        tables = sheets[0].tables
        data = tables[0].rows(values_only=True)
        df = pd.DataFrame(data[1:], columns=data[0])
        df.columns = ['height','vel_n','vel_e','vel_d','wind_n','wind_e','wind_d','air_density','final_pos_n','final_pos_e']
        print(df)

    def predict(self):
        # Example usage
        try:
            height = 100
            aircraft_velocity = [0, 0, 0]
            wind_velocity = [1, 1, 0]
            air_density = 1.225
            drag_coef = 0.5

            final_position = simulate_package_drop(height, aircraft_velocity, wind_velocity, air_density, drag_coef)
            print("Final Position (North, East):", final_position)
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    model = Model()
    model.predict()