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
            help="Index of reference, non-diffusion-weighted "
                "(or low-diffusion-weighted) acquisition")
        subparser.add_argument(
            "--population", "-p", type=int, default=10,
            help="Number of individuals")
        subparser.add_argument(
            "--generations", "-g", type=int, default=1,
            help="Number of generations")
        subparser.set_defaults(name=name, inputs=inputs, outputs=outputs)