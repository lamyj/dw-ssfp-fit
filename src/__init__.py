from ._dw_ssfp_fit import *

def get_combined_mask(B1_map, mask, T1_map, T2_map):
    """ Return a mask which is the combination of a binary mask and "usable"
        values in the B1, T1, and T2 maps.
    """
    
    combined_mask = (mask != 0) & (B1_map > 0.5) & (B1_map < 1.5)
    if T1_map is not None:
        combined_mask &= (T1_map > 0) & (T1_map < 5)
    if T2_map is not None:
        combined_mask &= (T2_map > 0) & (T2_map < 5)
    return combined_mask

def add_parser_commands(parser):
    import argparse
    import itertools
    
    subparsers = parser.add_subparsers(help="Available commands")
    
    for T1, T2 in itertools.product(["input", "output"], ["input", "output"]):
        inputs, outputs = [
            [name for name, status in zip(["T1", "T2"], [T1, T2]) if status==x]
            for x in ["input", "output"]]
        
        name = "D"+"".join(f"+{x}" for x in outputs)
        
        description = (
            "Fit the diffusion tensor"
            +"".join(
                f", {name}" if index != len(outputs)-1 else f" and {name}"
                for index, name in enumerate(outputs))
            +" from the DW-SSFP data, the B1 map" 
            + ("," if inputs else " and") + " the mask"
            +"".join(
                f", the {name} map" if index != len(inputs)-1 else f" and {name} map"
                for index, name in enumerate(inputs))
        )
        
        help = "Fit the diffusion tensor"+"".join(
            f", {name}" if index != len(outputs)-1 else f" and {name}"
            for index, name in enumerate(outputs))
        
        subparser = subparsers.add_parser(
            name, description=description, help=help,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        subparser.add_argument("scheme", help="Acquisition scheme")
        subparser.add_argument("DW_SSFP", help="DW-SSFP image")
        subparser.add_argument("B1_map", help="Relative B1 map")
        subparser.add_argument("mask", help="Binary mask")
        
        for input_ in inputs:
            subparser.add_argument(
                f"{input_}_map", help=f"Input {input_} map, in seconds")
        
        subparser.add_argument(
            "D",
            help="Output diffusion tensor image, following the MRtrix format")
        
        for output in outputs:
            subparser.add_argument(
                f"{output}_map", help=f"Output {output} map, in seconds")
        
        subparser.add_argument(
            "--reference", "-r", type=int, default=0,
            help="Number of non-diffusion-weighted (or low-diffusion-weighted) "
                "acquisition")
        subparser.add_argument(
            "--population", "-p", type=int, default=10,
            help="Number of individuals")
        subparser.add_argument(
            "--generations", "-g", type=int, default=1,
            help="Number of generations")
        subparser.set_defaults(name=name, inputs=inputs, outputs=outputs)

unwrapped_fit = fit

def wrapped_fit(
        scheme, non_dw, 
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
        scheme, non_dw, DW_SSFP, B1_map, T1_map, T2_map,
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
