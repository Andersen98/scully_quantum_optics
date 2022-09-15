import numpy as np
from scipy import constants
import pytest
from scully_quantum_optics.chapter_1.mode_expansion_of_field.field_quantities import ModeIndex3D, eigen_frequency_1D, eigen_frequency_3D

def test_eigenfrequency_3D_rotational_symmetry():
    idx_1: ModeIndex3D = {'x':0,'y':3,'z':4}
    idx_2: ModeIndex3D = {'x':3,'y':4,'z':0}
    idx_3: ModeIndex3D = {'x':3,'y':0,'z':4}
    idx_4: ModeIndex3D = {'x':0,'y':4,'z':3}
    idx_5: ModeIndex3D = {'x':4,'y':3,'z':0}
    idx_6: ModeIndex3D = {'x':4,'y':0,'z':3}
    idx_7: ModeIndex3D = {'x':5,'y':0,'z':0}
    idx_8: ModeIndex3D = {'x':0,'y':5,'z':0}
    idx_9: ModeIndex3D = {'x':0,'y':0,'z':5}
    indices =  [idx_1,idx_2,idx_3,idx_4,idx_5,idx_6,
                idx_7,idx_8, idx_9]
    L = 157
    frequencies = np.array(
                    [eigen_frequency_3D(n_vec,L,constants.c) for n_vec in indices])
    desired_frequencies = (5.0*np.pi*constants.c/L)*np.ones(9)
    np.testing.assert_allclose(frequencies,desired_frequencies)

def test_eigenfrequency_1D_typical():
    L = 405.3
    n_1 = 3
    n_2 = 14
    n_3 = 220
    desired_1 = n_1*np.pi*constants.c/L
    desired_2 = n_2*np.pi*constants.c/L
    desired_3 = n_3*np.pi*constants.c/L
    desired = np.array([desired_1,desired_2,desired_3])
    actual = np.array([n*np.pi*constants.c/L for n in [n_1,n_2,n_3]])
    np.testing.assert_allclose(actual,desired)

def test_raises_ModeIndex3D():
    with pytest.raises(NotImplementedError):
        a: ModeIndex3D = {'x': 1.5, 'y':1, 'z':0}
        eigen_frequency_3D(a,1.0,1.0)

def test_raises_ModeIndex1D():
    with pytest.raises(NotImplementedError):
        eigen_frequency_1D(1.5,1.0,1.0)