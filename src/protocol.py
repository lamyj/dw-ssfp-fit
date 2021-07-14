import base64
import dicomifier
import numpy
import sycomore
from sycomore.units import *

class Protocol(object):
    def __init__(self, meta_data):
        csa = dicomifier.dicom_to_nifti.siemens.parse_csa(
            base64.b64decode(meta_data["00291020"][0]))
        protocol = dicomifier.dicom_to_nifti.siemens.parse_ascconv(
            csa["MrPhoenixProtocol"][0])
        
        # TODO Phase Correction, Echo Shifting / alFree[5], alFree[6]
        # TODO adFree[0] (tau_diffusion in seconds?)
        
        self.q = protocol["sWiPMemBlock"]["adFree"][2]/cm
        self.G_diffusion = protocol["sWiPMemBlock"]["adFree"][3]*mT/m
        self.tau_diffusion = protocol["sWiPMemBlock"]["adFree"][4]*ms
        self.diffusion_vectors = protocol["sWiPMemBlock"]["alFree"][2]
        self.direction_set = protocol["sWiPMemBlock"]["alFree"][3]
        self.directions_count = protocol["sWiPMemBlock"]["alFree"][4]
        
        computed_q = sycomore.gamma/(2*numpy.pi)*self.G_diffusion*self.tau_diffusion
        if abs((computed_q-self.q)/self.q) > 1e-2:
            raise Exception("q-value does not match Gmax and Grad dur")
        
        self.rf_pulse_duration = protocol["sWiPMemBlock"]["alFree"][7]*us
        self.rf_time_bandwidth_product = protocol["sWiPMemBlock"]["adFree"][5]
        self.global_fft_scale_factor = protocol["sWiPMemBlock"]["adFree"][6]
        self.slew_rate_slow_down = {
            name: protocol["sWiPMemBlock"]["adFree"][7+index]
            for index, name in enumerate(["Diff", "RO", "PE1", "PE2"])}
