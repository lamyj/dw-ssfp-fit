import argparse
import json
import pathlib
import sys

# WARNING: import nibabel *before* mpi4py to avoid a fork-related warning when
# running with mpirun.
import nibabel
from mpi4py import MPI
import numpy
from sycomore.units import *

import dw_ssfp_fit

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("scheme", help="Path to the acquisition scheme")
    parser.add_argument("DW_SSFP", help="Path to the DW-SSFP image")
    parser.add_argument(
        "T1_map", help="Path to the T1 map image, must be in seconds")
    parser.add_argument(
        "T2_map", help="Path to the T2 map image, must be in seconds")
    parser.add_argument("B1_map", help="Path to the relative B1 map image")
    parser.add_argument("mask", help="Path to the mask image")
    parser.add_argument("D", help="Path to the output diffusion tensor image")
    
    parser.add_argument(
        "--reference", "-r", type=int, default=0,
        help="Number of non-diffusion-weighted (or low-diffusion-weighted) "
            "acquisition")
    parser.add_argument(
        "--population", "-p", type=int, default=10,
        help="Number of individuals")
    parser.add_argument(
        "--generations", "-g", type=int, default=1,
        help="Number of generations")
    arguments = parser.parse_args()
    
    communicator = MPI.COMM_WORLD
    
    scheme, DW_SSFP, T1_map, T2_map, B1_map, mask = load(
        communicator, arguments.scheme, arguments.DW_SSFP,
        arguments.T1_map, arguments.T2_map, arguments.B1_map,
        arguments.mask)
    D = fit(
        communicator, scheme, arguments.reference,
        DW_SSFP, T1_map, T2_map, B1_map, mask,
        arguments.population, arguments.generations)
    if communicator.rank == 0:
        save(communicator, D, DW_SSFP.affine, arguments.D)
    
def load(communicator, scheme, DW_SSFP, T1_map, T2_map, B1_map, mask):
    # Load the scheme on all ranks to avoid synchronizing a data structure
    with open(str(scheme)) as fd:
        scheme = [dw_ssfp_fit.Acquisition(**x) for x in json.load(fd)]
    
    # Load the images only on rank 0 as blocks are dispatched to other rank
    # during fit.
    if communicator.rank == 0:
        DW_SSFP = nibabel.load(str(DW_SSFP))
        T1_map = nibabel.load(str(T1_map))
        T2_map = nibabel.load(str(T2_map))
        B1_map = nibabel.load(str(B1_map))
        
        mask = nibabel.load(mask)
        # Update mask to discard invalid T1, T2, B1 values
        ROI = (
            (mask.get_fdata() != 0)
            & (T1_map.get_fdata() > 0) & (T1_map.get_fdata() < 2) 
            & (T2_map.get_fdata() > 0) & (T2_map.get_fdata() < 1) 
            & (B1_map.get_fdata() > 0.5) & (B1_map.get_fdata() < 1.5))
        mask = nibabel.Nifti1Image(ROI.astype(int), mask.affine)
    else:
        DW_SSFP = numpy.array(())
        T1_map = numpy.array(())
        T2_map = numpy.array(())
        B1_map = numpy.array(())
        mask = numpy.array(())
    
    return scheme, DW_SSFP, T1_map, T2_map, B1_map, mask

def fit(
        communicator, scheme, non_dw, 
        DW_SSFP, T1_map, T2_map, B1_map, mask,
        population, generations):
    _, D = dw_ssfp_fit.fit(
        scheme, non_dw, DW_SSFP, T1_map, T2_map, B1_map, mask,
        communicator, population, generations, False, True)
    return D

def save(communicator, D, affine, path):
    """ Save a diffusion tensor map to an MRtrix-compatible format.
    """
    
    # https://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2tensor.html
    # https://community.mrtrix.org/t/unit-measure-of-dti-metrics/2401
    # Order of volume is D11, D22, D33, D12, D13, D23
    # b-values in scheme are in s/mm^2, so the diffusion coefficients are
    # implicitly in mm^2/s
    volumes = [[0,0], [1,1], [2,2], [0,1], [0,2], [1,2]]
    scale = (m**2/s).convert_to(mm**2/s)
    D = numpy.asfortranarray(
        1e6*numpy.stack([D[...,x[0], x[1]] for x in volumes], axis=-1))
    nibabel.save(nibabel.Nifti1Image(D, affine), str(path))

if __name__ == "__main__":
    sys.exit(main())