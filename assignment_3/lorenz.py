"""
lorenz.py
Function defining the Lorenz system of ODEs (Lorenz 1963, equations 25-27).
"""


def lorenz(t, W, sigma=10., r=28., b=8./3.):
    """
    Compute the time derivatives of the Lorenz system.

    The Lorenz equations (from Lorenz 1963, eqs. 25-27) are:
        X' = -sigma * (X - Y)
        Y' = r*X - Y - X*Z
        Z' = -b*Z + X*Y

    Parameters
    ----------
    t : float
        Current time (not used explicitly; required by ODE solver interface).
    W : array-like, shape (3,)
        State vector [X, Y, Z].
    sigma : float, optional
        Prandtl number (default 10).
    r : float, optional
        Rayleigh number (default 28).
    b : float, optional
        Dimensionless length scale (default 8/3).

    Returns
    -------
    dW : list of float
        Time derivatives [dX/dt, dY/dt, dZ/dt].
    """
    X, Y, Z = W
    dX = -sigma * (X - Y)
    dY = r * X - Y - X * Z
    dZ = -b * Z + X * Y
    return [dX, dY, dZ]
