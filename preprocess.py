import json
import pathlib

import nibabel
import qMRI_toolbox
import spire
import spire.ants
from sycomore.units import *

import dw_ssfp_fit

root = pathlib.Path(".")

XFL_flip_angle = next(root.glob("*/*/*XFL_B1_Ampli*/1.nii.gz"))
SPGR_PDw = {
    json.load((x.parent/"1.json").open())["ImageType"][2]: x
    for x in root.glob("*/*/*MT map (PDw)*/1.nii.gz")}
SPGR_T1w = {
    json.load((x.parent/"1.json").open())["ImageType"][2]: x
    for x in root.glob("*/*/*MT map (T1w)*/1.nii.gz")}
bSSFP_T2 = sorted(
    root.glob("*/*/*TrueFISP*/1.nii.gz"), 
    key=lambda x: int(x.parts[-2].split("_")[0]))
DW_SSFP = sorted(
    root.glob("*/*/*DW-SSFP*/1.nii.gz"), 
    key=lambda x: int(x.parts[-2].split("_")[0]))
diffusion_vectors = list(root.glob("DiffusionVectors*txt"))

# FIXME? realign SPGR_PDw and SPGR_T1w

brain = qMRI_toolbox.segmentation.BET(
    DW_SSFP[0], "brain.nii.gz", mask="brain_mask.nii.gz")

B1_map = qMRI_toolbox.b1_map.XFL(XFL_flip_angle, "B1_map.nii.gz")
B1_map_in_SPGR_MT = spire.ants.ApplyTransforms(
    B1_map.targets[0], (SPGR_T1w["M"], 0), [], "B1_map_in_SPGR_MT.nii.gz")
B1_map_in_bSSFP_T2 = spire.ants.ApplyTransforms(
    B1_map.targets[0], bSSFP_T2[0], [], "B1_map_in_bSSFP_T2.nii.gz")
B1_map_in_DW_SSFP = spire.ants.ApplyTransforms(
    B1_map.targets[0], (DW_SSFP[0], 0), [], "B1_map_in_DW_SSFP.nii.gz")

T1_map = qMRI_toolbox.t1_map.VFA(
    [x["M"] for x in [SPGR_PDw, SPGR_T1w]], B1_map_in_SPGR_MT.targets[0], 
    "T1_map.nii.gz")
T1_map_in_bSSFP_T2 = spire.ants.ApplyTransforms(
    T1_map.targets[0], bSSFP_T2[0], [], "T1_map_in_bSSFP_T2.nii.gz")

T2_map = qMRI_toolbox.t2_map.bSSFP(
    bSSFP_T2, B1_map_in_bSSFP_T2.targets[0], T1_map_in_bSSFP_T2.targets[0],
    "T2_map.nii.gz")

T1_map_in_DW_SSFP = spire.ants.ApplyTransforms(
    T1_map.targets[0], [DW_SSFP[0], 0], [], "T1_map_in_DW_SSFP.nii.gz")
T2_map_in_DW_SSFP = spire.ants.ApplyTransforms(
    T2_map.targets[0], [DW_SSFP[0], 0], [], "T2_map_in_DW_SSFP.nii.gz")

class merged_DW_SSFP(spire.Task):
    file_dep = DW_SSFP
    targets = ["dw_ssfp.nii.gz"]
    actions = [["mrcat", "-force", *file_dep, *targets]]

class DW_SSFP_denoised(spire.Task):
    file_dep = merged_DW_SSFP.targets
    targets = ["dw_ssfp_denoised.nii.gz"]
    actions = [["dwidenoise", "-force", *file_dep, *targets]]

class DW_SSFP_metadata(spire.Task):
    def create_meta_data(image_paths, diffusion_vectors_paths, output):
        diffusion_vectors = []
        for path in sorted(diffusion_vectors_paths):
            try:
                diffusion_vectors.append(
                    dw_ssfp_fit.diffusion_vectors.read(path.open()))
            except:
                continue
        
        scheme = []
        for path in image_paths:
            image = nibabel.load(path)
            meta_data = json.load((path.parent/"1.json").open())
            protocol = dw_ssfp_fit.Protocol(meta_data)
            directions = diffusion_vectors[protocol.diffusion_vectors][protocol.direction_set]
            
            if len(directions) != protocol.directions_count:
                raise Exception("Directions count mismatch")
            
            for direction in directions:
                scheme.append({
                    "alpha": (meta_data["FlipAngle"][0]*deg).convert_to(rad),
                    "G_diffusion": (protocol.G_diffusion).convert_to(T/m), 
                    "tau_diffusion": (protocol.tau_diffusion).convert_to(s), 
                    "direction": direction, 
                    "TE": (meta_data["EchoTime"][0]*ms).convert_to(s), 
                    "TR": (meta_data["RepetitionTime"][0]*ms).convert_to(s),
                    "pixel_bandwidth": (meta_data["PixelBandwidth"][0]*Hz).convert_to(Hz), 
                    "train_length": protocol.train_length,
                    "shape": protocol.shape,
                    "FOV": [x.convert_to(m) for x in protocol.FOV],
                    "G_max": (25*mT/m).convert_to(T/m)
                })
        
        with open(output, "w") as fd:
            json.dump(scheme, fd)
    
    file_dep = DW_SSFP+diffusion_vectors
    targets = ["dw_ssfp.json"]
    actions = [(create_meta_data, (DW_SSFP, diffusion_vectors, targets[0]))]
