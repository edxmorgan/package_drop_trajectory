# assumption 1: The aircraft height, velocity, and air density are perfectly known

# assumption 2: The ground is flat near delivery sites.

# assumption 3: flight path angle is zero thus the aircraft is travelling horizontally

# assumption 4: package is a point mass

# assumption 5:  wind estimate is typically off by ~1m/s. 

# assumption 6:  the time delay for the package release is less than 1 second

#to solve this problem using classical physics, consider the dynamics of the package (point mass) as it falls through the air,
#influenced by gravity, drag and wind. The challenge is to predict the final landing position (North-East coordinates)
#relative to the point of release of the aircraft.


# initial conditions : when released, the package initially has the aircraft's velocity and is affected by wind.

# forces acting on the package: 
# 1. gravity pulls the package downwards.
# 2. drag force opposes the motion of the package. 
# 3. wind effect: wind adds or substract from the package's velocity in each direction (north, east and down) #for kinematics

# 4.equations of motion: the motion of the package can be described using the equations of motion under constant acceleration, modified
#   modified for drag and wind.

import numpy as np
from scipy.integrate import solve_ivp

def package_dynamics(t, state, air_density, drag_coef, package_mass, wind_velocity, g):
    # Unpack the current state
    x, y, z, vx, vy, vz = state

    # Adjust velocity for wind effect
    relative_velocity = np.array([vx, vy, vz]) - np.array(wind_velocity)
    relative_velocity_mag = np.linalg.norm(relative_velocity)

    # Calculate drag force based on relative velocity
    drag_force = -0.5 * drag_coef * air_density * relative_velocity_mag * relative_velocity

    # Acceleration due to gravity and drag
    ax, ay, az = drag_force / package_mass
    az -= g  # Gravity, negative in the down direction

    # Return derivatives of state
    return [vx, vy, vz, ax, ay, az]



def hit_ground_event(t, state, air_density, drag_coef, package_mass, wind_velocity, g):
    # Event function to stop the integration when the package hits the ground
    return state[2]  # Z-position

# Set the event function to only look for zero-crossings from positive to negative (package hitting the ground)
hit_ground_event.terminal = True
hit_ground_event.direction = -1


def simulate_package_drop(height, aircraft_velocity, wind_velocity, air_density, drag_coef, package_mass=1.0, g=9.81):
    # Initial conditions: x, y, z, vx, vy, vz
    initial_state = [0, 0, height] + list(aircraft_velocity)
    
    # Time span for the simulation
    t_span = (0, 30)  # 30 seconds max, adjust as needed

    # Solve the differential equations
    sol = solve_ivp(package_dynamics, t_span, initial_state, 
                    args=(air_density, drag_coef, package_mass, wind_velocity, g), 
                    events=hit_ground_event)

    # Check if the package reached the ground
    if sol.status == 1 and sol.t_events[0].size > 0:
        # Get the final North and East position at the ground hit time
        final_north = sol.y[0, -1]
        final_east = sol.y[1, -1]
        return final_north, final_east
    else:
        raise ValueError("Package did not reach the ground within the simulation time frame.")
