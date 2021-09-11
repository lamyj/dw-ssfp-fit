from ._dw_ssfp_fit import *

unwrapped_fit = fit

def wrapped_fit(
        scheme, reference, 
        DW_SSFP, B1_map, mask, T1_map, T2_map,
        communicator, population, generations,
        return_individuals=False):
    
    import numpy
    
    base_shape = DW_SSFP.shape[:-1]
    
    if communicator.rank == 0:
        if base_shape != B1_map.shape:
            raise Exception("Shape mismatch between DW_SSFP and B1_map")
        if base_shape != mask.shape:
            raise Exception("Shape mismatch between DW_SSFP and mask")
        if T1_map is not None and base_shape != T1_map.shape:
            raise Exception("Shape mismatch between DW_SSFP and T1_map")
        if T2_map is not None and base_shape != T2_map.shape:
            raise Exception("Shape mismatch between DW_SSFP and T2_map")
        
        mask = numpy.asarray(mask.dataobj).astype(bool)
        
        # NOTE: slicing by mask returns a C-contiguous, linear, array.
        DW_SSFP = DW_SSFP.get_fdata()[mask]
        B1_map = B1_map.get_fdata()[mask]
        T1_map = T1_map.get_fdata()[mask] if T1_map else None
        T2_map = T2_map.get_fdata()[mask] if T2_map else None
    else:
        DW_SSFP = B1_map = T1_map = T2_map = numpy.array([])
    
    (
        champions_D, champions_T1, champions_T2,
        individuals_D, individuals_T1, individuals_T2
    ) = unwrapped_fit(
        scheme, reference, DW_SSFP, B1_map, T1_map, T2_map,
        communicator, population, generations, return_individuals)
    
    def get_full_from_mask(source, mask):
        full = numpy.zeros(mask.shape+source.shape[1:], order="F")
        full[mask] = source
        return full
    
    if communicator.rank == 0:
        champions_D = get_full_from_mask(champions_D, mask)
        if champions_T1 is not None:
            champions_T1 = get_full_from_mask(champions_T1, mask)
        if champions_T2 is not None:
            champions_T2 = get_full_from_mask(champions_T2, mask)
        
        if return_individuals:
            individuals_D = get_full_from_mask(individuals_D, mask)
            if individuals_T1 is not None:
                individuals_T1 = get_full_from_mask(individuals_T1, mask)
            if individuals_T2 is not None:
                individuals_T2 = get_full_from_mask(individuals_T2, mask)
        
    
    return (
        champions_D, champions_T1, champions_T2,
        individuals_D, individuals_T1, individuals_T2)

fit = wrapped_fit

from . import diffusion_vectors
from .protocol import Protocol
