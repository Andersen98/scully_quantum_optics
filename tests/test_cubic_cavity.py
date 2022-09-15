import numpy as np
from scipy import constants
import pytest
from sympy import epath
from scully_quantum_optics.chapter_1.mode_expansion_of_field.field_quantities import ModeIndex3D
from scully_quantum_optics.chapter_1.quantization.cubic_cavity import electric_field_coefficient

def test_electric_field_coefficient_simple():
    L = 1
    c = 1
    epsilon_naught = 1
    hbar = 1
    indices = list(range(20))
    desired_frequencies = [j*np.pi for j in indices]
    desired = np.array([np.sqrt(freq/2) for freq in desired_frequencies])
    n_vecs = [ModeIndex3D(x=idx,y=0,z=0)  for idx in indices]
    actual = np.array([
        electric_field_coefficient(n_vec,L,epsilon_naught,c,hbar) 
        for n_vec in n_vecs])
    np.testing.assert_allclose(actual,desired)
