mass = 1
# assumption 1: The aircraft height, velocity, and air density are perfect known

# assumption 2: The ground is flat near delivery sites.

# assumption 3: flight path angle is zero

# assumption 4: package is a point mass

# wind estimate is typically wrong by ~1m/s. 
# wind velocity can be modelled stochastically using a gaussian distribution with mean = "{wind velocity}" and variance = 1


def drag(coef, rho, v):
    """
    This function computes the drag force on the package in the direction of the velocity wrt the air.
 
    Parameters:
    coef (float): the drag coefficient and area. (0.1 to 10.)
    rho (float): the air density.
    v (float): the magnitude of the velocity wrt the air.
 
    Returns:
    float: the quadratic drag formulation 
    """
    return 0.5 * coef * rho * v^2 