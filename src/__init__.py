from ._dw_ssfp_fit import *

unwrapped_fit = fit

def wrapped_fit(
        scheme, non_dw, 
        DW_SSFP, T1_map, T2_map, B1_map, mask, 
        communicator, population, generations,
        return_individuals, return_champions
    ):
    
    import numpy
    
    base_shape = DW_SSFP.shape[:-1]
    
    if communicator.rank == 0:
        # NOTE: don't cast mask to double
        mask = numpy.asarray(mask.dataobj).astype(bool)
        
        DW_SSFP = DW_SSFP.get_fdata()[mask]
        T1_map = T1_map.get_fdata()[mask]
        T2_map = T2_map.get_fdata()[mask]
        B1_map = B1_map.get_fdata()[mask]
    else:
        DW_SSFP = T1_map = T2_map = B1_map = numpy.array([])
    
    individuals, champions = unwrapped_fit(
        scheme, non_dw, DW_SSFP, T1_map, T2_map, B1_map,
        communicator, population, generations,
        return_individuals, return_champions)
    
    if communicator.rank == 0:
        # WARNING F-contiguous vs. C-contiguous
        if return_individuals:
            full = numpy.zeros(base_shape+individuals.shape[1:], order="F")
            full[mask] = individuals
            individuals = full
        if return_champions:
            full = numpy.zeros(base_shape+champions.shape[1:], order="F")
            full[mask] = champions
            champions = full
    
    return individuals, champions

fit = wrapped_fit

from . import diffusion_vectors
from .protocol import Protocol
