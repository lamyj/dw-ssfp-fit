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
import helpers

def main():
    parser = argparse.ArgumentParser(
        description="Fit the diffusion tensor, and optionally the T1 and T2 "
            "from DW-SSFP data")
    
    helpers.add_parser_commands(parser)
    arguments = parser.parse_args()
    
    communicator = MPI.COMM_WORLD
    
    scheme, DW_SSFP, B1_map, mask, T1_map, T2_map = load(
        communicator, arguments.scheme,
        arguments.DW_SSFP, arguments.B1_map, arguments.mask,
        arguments.T1_map if "T1" in arguments.inputs else None, 
        arguments.T2_map if "T2" in arguments.inputs else None)
    
    D, T1, T2 = fit(
        communicator, scheme, arguments.reference,
        DW_SSFP, B1_map, mask, T1_map, T2_map, B1_map, mask,
        arguments.population, arguments.generations)
    if communicator.rank == 0:
        affine = DW_SSFP.affine
        save_D(communicator, D, affine, arguments.D)
        if "T1" in arguments.outputs:
            nibabel.save(nibabel.Nifti1Image(T1, affine), arguments.T1_map)
        if "T2" in arguments.outputs:
            nibabel.save(nibabel.Nifti1Image(T2, affine), arguments.T2_map)
    
def load(communicator, scheme, DW_SSFP, B1_map, mask, T1_map, T2_map):
    # Load the scheme on all ranks to avoid synchronizing a data structure
    with open(scheme) as fd:
        scheme = [dw_ssfp_fit.Acquisition(**x) for x in json.load(fd)]
    
    # Load the images only on rank 0 as blocks are dispatched to other rank
    # during fit.
    if communicator.rank == 0:
        DW_SSFP = nibabel.load(DW_SSFP)
        B1_map = nibabel.load(B1_map)
        mask = nibabel.load(mask)
        
        T1_map = nibabel.load(T1_map) if T1_map is not None else None
        T2_map = nibabel.load(T2_map) if T2_map is not None else None
        
        # Update mask to discard invalid T1, T2, B1 values
        ROI = helpers.get_combined_mask(
            B1_map.get_fdata(), numpy.asarray(mask.dataobj),
            T1_map.get_fdata() if T1_map is not None else None,
            T2_map.get_fdata() if T2_map is not None else None)
        mask = nibabel.Nifti1Image(ROI.astype(int), mask.affine)
    else:
        DW_SSFP = numpy.array(())
        B1_map = numpy.array(())
        mask = numpy.array(())
        T1_map = numpy.array(()) if T1_map is not None else None
        T2_map = numpy.array(()) if T2_map is not None else None
        
    return scheme, DW_SSFP, B1_map, mask, T1_map, T2_map

def fit(
        communicator, scheme, non_dw, 
        DW_SSFP, B1_map, mask, T1_map, T2_map,
        population, generations):
    D, T1, T2, _, _, _ = dw_ssfp_fit.fit(
        scheme, non_dw, DW_SSFP, B1_map, mask, T1_map, T2_map,
        communicator, population, generations)
    return D, T1, T2

def save_D(communicator, D, affine, path):
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
    nibabel.save(nibabel.Nifti1Image(D, affine), path)

if __name__ == "__main__":
    sys.exit(main())
