"""
Tanh Lookup Table Loader
=========================
This module provides functions to load and apply the tanh lookup table
used in the FPGA hardware implementation.

The LUT maps signed 8-bit input values [-128, 127] to tanh approximations.
"""

import os
import numpy as np


def load_tanh_lut():
    """
    Load the tanh lookup table from tanh_lut.mem file.
    
    Returns:
        np.ndarray: Array of 256 signed 8-bit values representing tanh outputs
                    for inputs from -128 to 127
    """
    # Find the tanh_lut.mem file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lut_path = os.path.join(script_dir, "..", "outputs", "mem", "tanh_lut.mem")
    
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"tanh_lut.mem not found at {lut_path}")
    
    # Read hex values from the .mem file
    lut = []
    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Parse hex value and convert to signed 8-bit
                hex_val = int(line, 16)
                # Convert from unsigned to signed if necessary
                if hex_val >= 128:
                    signed_val = hex_val - 256
                else:
                    signed_val = hex_val
                lut.append(signed_val)
    
    if len(lut) != 256:
        raise ValueError(f"Expected 256 entries in LUT, got {len(lut)}")
    
    return np.array(lut, dtype=np.int8)


def apply_tanh_lut(value, tanh_lut):
    """
    Apply tanh lookup table to a value.
    
    Args:
        value: Input value (should be in range [-128, 127])
        tanh_lut: Tanh lookup table array from load_tanh_lut()
    
    Returns:
        int: Tanh approximation of the input value
    """
    # Clip to valid range
    value = np.clip(value, -128, 127)
    
    # Convert signed value [-128, 127] to LUT index [0, 255]
    # Input -128 maps to index 0, input 127 maps to index 255
    index = int(value + 128)
    
    return int(tanh_lut[index])
