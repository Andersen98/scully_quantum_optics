import numpy as np
from ..mode_expansion_of_field.field_quantities import ModeIndex3D,eigen_frequency_3D

def electric_field_coefficient(n_vec:ModeIndex3D,cube_length:float,epsilon_naught:float,c:float,hbar:float) -> float:
    """Electric field coefficient that shows up when you expand in ladder operators.

    Given by equation 1.1.20 of Scully and Zubairy, Quantum Optics. Has units of electric field.
    .. role:: m(math)

    .. m::
       :name: Equation 1.1.20

       \\mathcal{E}_{\\mathbf{k}}
        = \\left ( \\frac{\\hbar \\nu_k}{2\\epsilon_0 V}\\right)^{1/2}
    
    Args:
        n_vec: A list of 3 integers which determine the eigenmode of interest.
        cube_legth: Side length of the cubic resonator.
        epsiolon_naught: Permittivity of free space for whatever units you are using.
        c: Speed of light for whatever units you are using
        hbar: The reduced Planck's constant in whatever units you are using.
    Returns:
        float: Electric field coefficient for eigen mode k.
        
    """
    if(c <=0 or cube_length<=0 or hbar<=0 or epsilon_naught<=0):
        raise NotImplementedError('Function not defined for c,cube_length,epsilon_naugh,hbar <= 0.')
    v_n = eigen_frequency_3D(n_vec,cube_length,c)
    volume = cube_length**3
    return np.sqrt(hbar*v_n/(2*epsilon_naught*volume))