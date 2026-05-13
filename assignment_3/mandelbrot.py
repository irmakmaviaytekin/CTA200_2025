"""
mandelbrot.py
Functions for computing the Mandelbrot set via iteration of z_{i+1} = z_i^2 + c.
"""

import numpy as np


def mandelbrot_iterate(c, max_iter=100):
    """
    Iterate z_{i+1} = z_i^2 + c starting from z_0 = 0 for each point c.

    Parameters
    ----------
    c : np.ndarray (complex)
        2D array of complex numbers representing points in the complex plane.
    max_iter : int, optional
        Maximum number of iterations before declaring a point bounded (default 100).

    Returns
    -------
    diverge_iter : np.ndarray (int)
        2D array where each entry is the iteration number at which |z|^2 > 4
        (i.e., the point diverged). Points that never diverge are assigned max_iter.
    diverged : np.ndarray (bool)
        2D boolean array; True where the point diverged before max_iter.
    """
    z = np.zeros_like(c, dtype=complex)
    diverge_iter = np.full(c.shape, max_iter, dtype=int)
    diverged = np.zeros(c.shape, dtype=bool)

    for i in range(max_iter):
        # Only iterate points that haven't diverged yet
        mask = ~diverged
        z[mask] = z[mask] ** 2 + c[mask]

        # Check which newly diverged (|z|^2 > 4)
        newly_diverged = mask & (z.real**2 + z.imag**2 > 4)
        diverge_iter[newly_diverged] = i
        diverged[newly_diverged] = True

    return diverge_iter, diverged
