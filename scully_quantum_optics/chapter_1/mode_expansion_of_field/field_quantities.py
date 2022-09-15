from typing import List
import numpy as np

ModeIndex3D = List[int,int,int]
"""
Type alias for 3D quantized mode index.
"""


def eigen_frequency_1D(j:int,c:float,L:float) -> float:
    """Eigenfrequency of 1D cavity mode with index j

    .. role:: m(math)
    
    Refers to quatity :m:`\nu_j` in Equation 1.1.16. Explicitly
    this is:

    .. math:
       name: Equation 1.1.16

       \nu_j = j \pi c/L

    Args:
        j: Mode index.
        c: Speed of light.
        L: Length of the 1D cavity

    Returns:
        The cavity eigenfrequency of mode with index j.
    
    """
    return j*np.pi*c/L

def eigen_frequency_3D(n_vec:ModeIndex3D,L:float,c:float) -> float:
    """Eigen frequency of 3D cubic cavity for a given eigenmode index k.

    Generalization of eigen_frequency_1D. Given by:

    .. math::
       name: Eigen Frequency 3D

       \nu_k = k \pi c/L
    
    where :math:`k=|\mathbf{k}|^2`.

    Args:
        n_vec: Mode index. Should be tuple of 3 integers.
        L: Side length of cubic resonator.
        c: Speed of light.

    Returns:
        Eigenfrequency of eigenmode n_vec.
    
    """

    n_norm = np.sqrt(n_vec[0]**2+n_vec[1]**2+n_vec[2]**2)
    return n_norm*np.pi*c/L


