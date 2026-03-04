"""
Generate Test Vectors for Isolated Inference Testbench - 2 Hidden Layer Network

This script generates preprocessed test vectors for the Verilog testbench
to test inference.v in complete isolation.

Model Architecture:
  - Input:  784 pixels (28x28 image, 8-bit signed)
  - Layer 1: 784 → 16 neurons (ReLU activation, right-shift by 7)
  - Layer 2: 16 → 16 neurons (ReLU activation, right-shift by 7)
  - Layer 3: 16 → 10 outputs (no ReLU, no shift)

Output Files (in 2_ukryte/inference/tb/):
  - test_vectors_pixels.mem: Preprocessed pixel data (hex bytes)
  - test_vectors_scores.mem: Expected class scores from simulation (hex int32)
  - test_vectors_meta.mem: Expected predictions (hex nibbles)
  - test_vectors_labels.mem: True labels (hex nibbles)
"""

import numpy as np
import os
import sys

# Configuration
NUM_TEST_VECTORS = 100
OUTPUT_DIR = "../outputs/mem"

# Quantization constants (must match training and hardware)
INPUT_SCALE = 127.0
SHIFT1 = 7  # Right-shift after Layer 1
SHIFT2 = 7  # Right-shift after Layer 2


def load_quantized_model():
    """Load quantized weights and biases from binary files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "..", "outputs", "bin")
    
    # Load weights and biases
    L1_weights = np.fromfile(os.path.join(bin_dir, "L1_weights.bin"), dtype=np.int8).reshape(16, 784)
    L1_biases = np.fromfile(os.path.join(bin_dir, "L1_biases.bin"), dtype=np.int32)
    L2_weights = np.fromfile(os.path.join(bin_dir, "L2_weights.bin"), dtype=np.int8).reshape(16, 16)
    L2_biases = np.fromfile(os.path.join(bin_dir, "L2_biases.bin"), dtype=np.int32)
    L3_weights = np.fromfile(os.path.join(bin_dir, "L3_weights.bin"), dtype=np.int8).reshape(10, 16)
    L3_biases = np.fromfile(os.path.join(bin_dir, "L3_biases.bin"), dtype=np.int32)
    
    return (L1_weights, L1_biases, L2_weights, L2_biases, L3_weights, L3_biases)


def load_norm_params():
    """Load PyTorch normalization parameters saved during training."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_dir = os.path.join(script_dir, "..", "outputs", "npy")
    
    norm_mean = np.load(os.path.join(npy_dir, "norm_mean.npy"))
    norm_std = np.load(os.path.join(npy_dir, "norm_std.npy"))
    return norm_mean, norm_std


def preprocess_image(image_data, norm_mean, norm_std):
    """
    Apply the same preprocessing as during training:
    1. Normalize to [0, 1] (divide by 255)
    2. Apply PyTorch normalization: (x - mean) / std
    3. Quantize to int8: multiply by INPUT_SCALE (127)
    4. Clip to [-128, 127]
    """
    # Convert to float and normalize to [0, 1]
    x = image_data.astype(np.float32) / 255.0
    
    # Apply PyTorch normalization
    x_normalized = (x - norm_mean) / norm_std
    
    # Quantize to int8 range
    x_quantized = np.round(x_normalized * INPUT_SCALE)
    
    # Clip to int8 range and convert
    x_int8 = np.clip(x_quantized, -128, 127).astype(np.int8)
    
    return x_int8


def fpga_inference_simulation(image_int8, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B):
    """
    Simulate FPGA fixed-point inference in Python.
    
    This matches the FPGA implementation with:
    - 32-bit accumulators
    - Right-shifts after Layer 1 and Layer 2
    - ReLU activation after Layer 1 and Layer 2
    - 32-bit overflow/underflow behavior (wrapping)
    """
    
    # Layer 1: 784 -> 16 with ReLU and right-shift
    L1_outputs = np.zeros(16, dtype=np.int32)
    for n in range(16):
        acc = np.int32(0)
        for i in range(784):
            product = np.int32(image_int8[i]) * np.int32(L1_W[n, i])
            acc = np.int32(acc + product)  # Wrap on overflow
        acc = np.int32(acc + L1_B[n])
        # Right-shift by SHIFT1 (arithmetic shift)
        acc = acc >> SHIFT1
        # ReLU
        L1_outputs[n] = max(0, acc)
    
    # Layer 2: 16 -> 16 with ReLU and right-shift
    L2_outputs = np.zeros(16, dtype=np.int32)
    for n in range(16):
        acc = np.int64(0)  # Use int64 to detect overflow
        for i in range(16):
            product = np.int64(L1_outputs[i]) * np.int64(L2_W[n, i])
            acc += product
        acc += np.int64(L2_B[n])
        
        # Clip to int32 range (simulating FPGA 32-bit register overflow)
        if acc > 2147483647:
            acc = 2147483647
        elif acc < -2147483648:
            acc = -2147483648
        
        # Right-shift by SHIFT2
        acc = np.int32(acc) >> SHIFT2
        # ReLU
        L2_outputs[n] = max(0, acc)
    
    # Layer 3: 16 -> 10 (no ReLU, no shift - final outputs)
    L3_outputs = np.zeros(10, dtype=np.int32)
    for c in range(10):
        acc = np.int64(0)  # Use int64 to detect overflow
        for i in range(16):
            product = np.int64(L2_outputs[i]) * np.int64(L3_W[c, i])
            acc += product
        acc += np.int64(L3_B[c])
        
        # Clip to int32 range (simulating FPGA 32-bit register overflow)
        if acc > 2147483647:
            acc = 2147483647
        elif acc < -2147483648:
            acc = -2147483648
        
        L3_outputs[c] = np.int32(acc)
    
    predicted_digit = np.argmax(L3_outputs)
    
    return predicted_digit, L3_outputs, (L1_outputs, L2_outputs, L3_outputs)


def load_mnist_test_data():
    """Load MNIST test dataset using torchvision or sklearn."""
    # Try torchvision first
    try:
        from torchvision import datasets, transforms
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, "..", "..", "data")
        
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        
        images = []
        labels = []
        for i in range(len(mnist_test)):
            img, label = mnist_test[i]
            img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
            images.append(img_np)
            labels.append(label)
        
        return np.array(images), np.array(labels)
        
    except ImportError:
        pass
    
    # Try sklearn as fallback
    try:
        from sklearn.datasets import fetch_openml
        
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X_test = X[60000:].astype(np.uint8)
        y_test = y[60000:].astype(int)
        return X_test, y_test
        
    except ImportError:
        print("ERROR: Neither torchvision nor sklearn is installed!")
        print("Install with: pip install torchvision  OR  pip install scikit-learn")
        sys.exit(1)


def generate_test_vectors():
    """Generate test vectors for Verilog testbench."""
    print("=" * 80)
    print("Generating Test Vectors for 2-Hidden-Layer Isolated Inference Testbench")
    print("=" * 80)
    print()
    
    # Load model
    print("Loading quantized model...")
    L1_W, L1_B, L2_W, L2_B, L3_W, L3_B = load_quantized_model()
    print(f"  [OK] L1 Weights: {L1_W.shape}, Biases: {L1_B.shape}")
    print(f"  [OK] L2 Weights: {L2_W.shape}, Biases: {L2_B.shape}")
    print(f"  [OK] L3 Weights: {L3_W.shape}, Biases: {L3_B.shape}")
    print()
    
    # Load normalization parameters
    print("Loading normalization parameters...")
    norm_mean, norm_std = load_norm_params()
    print(f"  [OK] norm_mean: {norm_mean}, norm_std: {norm_std}")
    print()
    
    # Load MNIST test data
    print("Loading MNIST test dataset...")
    X_test, y_test = load_mnist_test_data()
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
    
    correct_count = 0
    
    for i in range(NUM_TEST_VECTORS):
        # Get test image and label
        img_raw = X_test[i]
        true_label = y_test[i]
        
        # Preprocess image (exactly as UART sends it)
        img_preprocessed = preprocess_image(img_raw, norm_mean, norm_std)
        
        # Run Python simulation to get expected scores
        pred, scores, _ = fpga_inference_simulation(
            img_preprocessed, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B
        )
        
        if pred == true_label:
            correct_count += 1
        
        # Write pixel data (784 bytes in hex, ONE PER LINE for Verilog $readmemh)
        img_uint8 = img_preprocessed.view(np.uint8)
        for byte in img_uint8:
            f_pixels.write(f"{byte:02x}\n")
        
        # Write expected scores (10 int32 values in hex, ONE PER LINE)
        scores_uint32 = scores.view(np.uint32)
        for score in scores_uint32:
            f_scores.write(f"{score:08x}\n")
        
        # Write expected prediction
        f_meta.write(f"{pred:01x}\n")
        
        # Write true label
        f_labels.write(f"{true_label:01x}\n")
        
        # Display progress
        match_str = "[OK]" if pred == true_label else "[MISS]"
        print(f"  {i:3d}: Label={true_label}, Pred={pred} {match_str}")
    
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
    print(f"Python Simulation Accuracy: {correct_count}/{NUM_TEST_VECTORS} ({100*correct_count/NUM_TEST_VECTORS:.1f}%)")
    print()
    print(f"Output directory: {output_path}")
    print(f"Files generated:")
    print(f"  - test_vectors_pixels.mem  ({NUM_TEST_VECTORS * 784} lines, flattened)")
    print(f"  - test_vectors_scores.mem  ({NUM_TEST_VECTORS * 10} lines, flattened)")
    print(f"  - test_vectors_meta.mem    ({NUM_TEST_VECTORS} lines, predictions)")
    print(f"  - test_vectors_labels.mem  ({NUM_TEST_VECTORS} lines, labels)")
    print()
    print("Next steps:")
    print("  1. Open Vivado project")
    print("  2. Add tb_inference.v as simulation source")
    print("  3. Set tb_inference as top module for simulation")
    print("  4. Run behavioral simulation")
    print("=" * 80)


if __name__ == "__main__":
    generate_test_vectors()






