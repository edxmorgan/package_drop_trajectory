# assumption 1: The aircraft height, velocity, and air density are perfectly known

# assumption 2: The ground is flat near delivery sites.

# assumption 3: package is a point mass

# assumption 4:  wind estimate is typically off by ~1m/s. 

# assumption 5:  the time delay for the package release is less than 1 second

#to solve this problem using classical physics, consider the dynamics of the package (point mass) as it falls through the air,
#influenced by gravity, drag and wind. The challenge is to predict the final landing position (North-East coordinates)
#relative to the point of release of the aircraft.


# initial conditions : when released, the package initially has the aircraft's velocity.

# forces acting on the package: 
# 1. gravity pulls the package downwards.
# 2. drag force opposes the motion of the package. 
# 3. wind effect: wind adds or substract from the package's velocity in each direction (north, east and down)

# 4.equations of motion: the motion of the package can be described using the equations of motion under constant acceleration,
#   modified for drag and wind.

# preliminaries
# Delay differential equations are used in the mathematical modeling of systems where the reactions 
# to the stresses occur not immediately but after a certain non-negligible period of time.

# to tackle the final complication of delayed package ejection, the problem can be simulated using
#  a parametric system of DDE(delay differential equations) with one constant delay and a constant initial history function
# that can be solved numerically, by including a parameter representing the 
# delay time between the release command and the actual release of the package in a dde solver.
# This parameter will influence the initial conditions of your package's trajectory.
#  An optimization techniques can be applied to solver and the collected data to estimate 
# the delay parameter along with the other parameters of the model.

import numpy as np
from scipy.integrate import solve_ivp

class Dynamics():
    def __init__(self,package_mass, g):
        self.package_mass = package_mass
        self.g = g
    
    def package_dynamics(self, t, state, air_density, drag_area, package_mass, wind_velocity, g, delay):
        # Unpack the current state
        x, y, z, vx, vy, vz = state

        # t --> inf , vz --> Ve (steady state descent velocity)
        # print(vz) 

        # Check if the delay period has passed
        if t < delay:
            # During the delay, the package has not yet started moving
            # Its velocity remains unchanged, and there are no accelerations
            # constant initial history function
            return [vx, vy, vz, 0, 0, 0]

        # Adjust velocity for wind effect
        relative_velocity = np.array([vx, vy, vz]) - np.array(wind_velocity)
        relative_velocity_mag = np.linalg.norm(relative_velocity)

        # Calculate drag force based on relative velocity
        drag_force_x = -0.5 * drag_area[0] * air_density * relative_velocity_mag * relative_velocity[0]
        drag_force_y = -0.5 * drag_area[1] * air_density * relative_velocity_mag * relative_velocity[1]
        drag_force_z = -0.5 * drag_area[2] * air_density * relative_velocity_mag * relative_velocity[2]

        # Acceleration due to gravity and drag
        ax = drag_force_x / package_mass
        ay = drag_force_y / package_mass
        az = drag_force_z / package_mass

        az += g  # Gravity, positive in the down direction

        # Return derivatives of state
        return [vx, vy, vz, ax, ay, az]

    def hit_ground_event(self, t, state, air_density, drag_area, package_mass, wind_velocity, g, delay):
        # Event function to stop the integration when the package hits the ground
        return state[2]  # Z-position
    
    # Set properties for the event function
    hit_ground_event.terminal = True
    hit_ground_event.direction = 1

    def simulate_package_drop(self, initial_conditions, drag_area, delay):
        # Unpack initial conditions
        height, aircraft_vel_n, aircraft_vel_e, aircraft_vel_d, wind_vel_n, wind_vel_e, wind_vel_d, air_density = initial_conditions

        aircraft_velocity = list([aircraft_vel_n, aircraft_vel_e, aircraft_vel_d])
        wind_velocity = list([wind_vel_n, wind_vel_e, wind_vel_d])

        # Initial states: x, y, z, vx, vy, vz
        # GROUND AS REFERENCE
        initial_state = [0, 0, -height] + aircraft_velocity

        # Max tme span for the simulation
        t_span = (0, 100)  # 100 seconds max

        # Solve the differential equations
        sol = solve_ivp(self.package_dynamics, t_span, initial_state, 
                        args=(air_density, drag_area, self.package_mass, wind_velocity, self.g, delay), 
                        events=self.hit_ground_event)

        # Check if the package reached the ground
        if sol.status == 1 and sol.t_events[0].size > 0:
            # Get the final North and East position at the ground hit time
            final_north = sol.y[0, -1]
            final_east = sol.y[1, -1]
            return np.array([final_north, final_east])
        else:
            raise ValueError("Package did not reach the ground within the simulation time frame.")
