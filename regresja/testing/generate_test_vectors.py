"""
Generate Test Vectors for Isolated Inference Testbench

This script generates preprocessed test vectors for the Verilog testbench
to test inference.v in complete isolation.

Output Files (in regresja/inference/tb/):
  - test_vectors_pixels.mem: Preprocessed pixel data (hex bytes)
  - test_vectors_scores.mem: Expected class scores from simulation (hex int32)
  - test_vectors_meta.mem: Expected predictions (hex nibbles)
  - test_vectors_labels.mem: True labels (hex nibbles)
"""

import numpy as np
import os
import sys
from sklearn.datasets import fetch_openml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate_fpga_inference import (
    fpga_inference_with_overflow,
    load_model_and_scaler,
    preprocess_image_fpga
)

# Configuration
NUM_TEST_VECTORS = 100
OUTPUT_DIR = "../outputs/mem"

def generate_test_vectors():
    """Generate test vectors for Verilog testbench."""
    print("=" * 80)
    print("Generating Test Vectors for Isolated Inference Testbench")
    print("=" * 80)
    print()
    
    # Load model and scaler
    print("Loading model and scaler...")
    W, B, mean, scale = load_model_and_scaler()
    print(f"  [OK] Weights: {W.shape}, dtype: {W.dtype}")
    print(f"  [OK] Biases: {B.shape}, dtype: {B.dtype}")
    print(f"  [OK] Scaler loaded")
    print()
    
    # Load MNIST test data
    print("Loading MNIST test dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X_test = X[60000:]
    y_test = y[60000:].astype(int)
    print(f"  [OK] Loaded {len(X_test)} test images")
    print()
    
    # Prepare output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    
    # Open output files
    f_pixels = open(os.path.join(output_path, "test_vectors_pixels.mem"), "w")
    f_scores = open(os.path.join(output_path, "test_vectors_scores.mem"), "w")
    f_meta = open(os.path.join(output_path, "test_vectors_meta.mem"), "w")
    f_labels = open(os.path.join(output_path, "test_vectors_labels.mem"), "w")
    
    print(f"Generating {NUM_TEST_VECTORS} test vectors...")
    print("-" * 80)
    
    for i in range(NUM_TEST_VECTORS):
        # Get test image and label
        img_raw = X_test[i]
        true_label = y_test[i]
        
        # Preprocess image (exactly as UART sends it)
        img_preprocessed = preprocess_image_fpga(img_raw, mean, scale)
        
        # Run Python simulation to get expected scores
        pred, scores, had_overflow = fpga_inference_with_overflow(img_preprocessed, W, B)
        
        # Write pixel data (784 bytes in hex, ONE PER LINE for Verilog $readmemh)
        img_uint8 = img_preprocessed.view(np.uint8)
        for byte in img_uint8:
            f_pixels.write(f"{byte:02x}\n")
        
        # Write expected scores (10 int32 values in hex, ONE PER LINE)
        # Convert signed int32 to unsigned uint32 view for proper hex representation
        scores_uint32 = scores.view(np.uint32)
        for score in scores_uint32:
            f_scores.write(f"{score:08x}\n")
        
        # Write expected prediction
        f_meta.write(f"{pred:01x}\n")
        
        # Write true label
        f_labels.write(f"{true_label:01x}\n")
        
        # Display progress
        overflow_str = " [OVERFLOW]" if had_overflow else ""
        match_str = "[OK]" if pred == true_label else "[FAIL]"
        print(f"  {i:3d}: Label={true_label}, Pred={pred} {match_str}{overflow_str}")
    
    print("-" * 80)
    print()
    
    # Close files
    f_pixels.close()
    f_scores.close()
    f_meta.close()
    f_labels.close()
    
    print("=" * 80)
    print("Test Vector Generation Complete!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"Files generated:")
    print(f"  - test_vectors_pixels.mem  ({NUM_TEST_VECTORS * 784} lines, flattened)")
    print(f"  - test_vectors_scores.mem  ({NUM_TEST_VECTORS * 10} lines, flattened)")
    print(f"  - test_vectors_meta.mem    ({NUM_TEST_VECTORS} lines, predictions)")
    print(f"  - test_vectors_labels.mem  ({NUM_TEST_VECTORS} lines, labels)")
    print()
    print("Next steps:")
    print("  1. cd regresja/inference/tb/")
    print("  2. iverilog -o tb_inference_isolated tb_inference_isolated.v ../rtl/inference.v")
    print("  3. vvp tb_inference_isolated")
    print("=" * 80)


if __name__ == "__main__":
    generate_test_vectors()