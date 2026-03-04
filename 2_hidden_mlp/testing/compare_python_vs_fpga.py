"""
Compare Python Simulation vs FPGA Hardware Output - 2-Hidden-Layer Network

This script runs the same images through both:
1. Python simulation (with 32-bit overflow and fixed-point arithmetic)
2. Actual FPGA hardware via UART

Results can be written to a text file and/or displayed on console.

The comparison includes:
- All 10 class scores (logits) from both implementations
- Predicted digit from both implementations
- Match/mismatch status

Protocol:
  - Image Send: 0xBB 0x66 + 784 bytes + 0x66 0xBB
  - Scores Read: Send 0xCD, receive 40 bytes (10 scores × 4 bytes each)
"""

import serial
import time
import sys
import os
import struct
import numpy as np
import argparse

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
SCORES_READ_REQUEST = bytes([0xCD])

# Quantization scale (must match training)
INPUT_SCALE = 127.0

# Hardware shift amounts (must match training and FPGA implementation)
SHIFT1 = 7  # After Layer 1
SHIFT2 = 7  # After Layer 2

# Default Configuration
DEFAULT_COM_PORT = 'COM7'
DEFAULT_BAUD_RATE = 115200
DEFAULT_TEST_SAMPLES = 100
DEFAULT_OUTPUT_FILE = '../outputs/txt/comparison_output.txt'


class OutputWriter:
    """Handles output to file and/or console."""
    
    def __init__(self, output_mode='file', file_path=None):
        """
        Initialize output writer.
        
        Args:
            output_mode: 'file', 'console', or 'both'
            file_path: Path to output file (required if mode is 'file' or 'both')
        """
        self.mode = output_mode
        self.file_handle = None
        
        if output_mode in ['file', 'both']:
            if file_path is None:
                raise ValueError("file_path required for file output mode")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.file_handle = open(file_path, 'w', encoding='utf-8')
    
    def write(self, text):
        """Write text to configured output(s)."""
        if self.mode in ['console', 'both']:
            print(text, end='')
        
        if self.mode in ['file', 'both'] and self.file_handle:
            self.file_handle.write(text)
    
    def close(self):
        """Close file handle if open."""
        if self.file_handle:
            self.file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_quantized_model():
    """Load quantized weights and biases from binary files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "..", "outputs", "bin")
    
    # Check if files exist
    required_files = [
        "L1_weights.bin", "L1_biases.bin",
        "L2_weights.bin", "L2_biases.bin",
        "L3_weights.bin", "L3_biases.bin"
    ]
    
    for filename in required_files:
        filepath = os.path.join(bin_dir, filename)
        if not os.path.exists(filepath):
            print(f"ERROR: Required file not found: {filepath}")
            print("Please run the training script first to generate quantized weights.")
            sys.exit(1)
    
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
    
    mean_path = os.path.join(npy_dir, "norm_mean.npy")
    std_path = os.path.join(npy_dir, "norm_std.npy")
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("ERROR: Normalization files not found!")
        print(f"  Expected: {mean_path}")
        print(f"  Expected: {std_path}")
        print("  Run siec_2_ukryte.py first to generate them.")
        sys.exit(1)
    
    norm_mean = np.load(mean_path)
    norm_std = np.load(std_path)
    return norm_mean, norm_std


def preprocess_image(image_data, norm_mean, norm_std):
    """
    Apply the same preprocessing as during training:
    1. Normalize to [0, 1] (divide by 255)
    2. Apply PyTorch normalization: (x - mean) / std
    3. Quantize to int8: multiply by INPUT_SCALE (127)
    4. Clip to [-128, 127]
    
    Args:
        image_data: Raw pixel values (0-255), shape (784,)
        norm_mean: Normalization mean
        norm_std: Normalization std
    
    Returns:
        Preprocessed image as int8 array
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


def preprocess_for_uart(x_int8):
    """
    Convert int8 array to uint8 for UART transmission (preserves two's complement bits).
    
    Args:
        x_int8: Preprocessed image as int8 array
    
    Returns:
        Image as uint8 array for UART transmission
    """
    return x_int8.view(np.uint8)


def fpga_inference_simulation(image_int8, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B):
    """
    Simulate FPGA fixed-point inference in Python.
    
    This matches the FPGA implementation with:
    - 32-bit accumulators
    - Right-shifts after Layer 1 and Layer 2
    - ReLU activation after Layer 1 and Layer 2
    - 32-bit overflow/underflow behavior
    
    Args:
        image_int8: Preprocessed image (784,) as int8
        L1_W, L1_B: Layer 1 weights (16, 784) int8 and biases (16,) int32
        L2_W, L2_B: Layer 2 weights (16, 16) int8 and biases (16,) int32
        L3_W, L3_B: Layer 3 weights (10, 16) int8 and biases (10,) int32
    
    Returns:
        predicted_digit: Predicted class (0-9)
        class_scores: Array of 10 int32 scores
        layer_outputs: Tuple of (L1_out, L2_out, L3_out) for debugging
    """
    
    # Layer 1: 784 -> 16 with ReLU and right-shift
    L1_outputs = np.zeros(16, dtype=np.int32)
    for n in range(16):
        acc = np.int32(0)
        for i in range(784):
            product = np.int32(image_int8[i]) * np.int32(L1_W[n, i])
            acc += product
        acc += L1_B[n]
        # Right-shift by SHIFT1 (divide by 2^SHIFT1)
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
    print("Loading MNIST test dataset...")
    
    # Try torchvision first
    try:
        from torchvision import datasets, transforms
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, "..", "..", "data")
        
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        
        # Extract all images and labels
        images = []
        labels = []
        for i in range(len(mnist_test)):
            img, label = mnist_test[i]
            # Convert to numpy array (28x28), scale to 0-255
            img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
            images.append(img_np)
            labels.append(label)
        
        X_test = np.array(images)
        y_test = np.array(labels)
        
        print(f"  [OK] Loaded {len(X_test)} test images using torchvision")
        return X_test, y_test
        
    except ImportError:
        pass
    
    # Try sklearn as fallback
    try:
        from sklearn.datasets import fetch_openml
        
        print("  Torchvision not available, using sklearn...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # Use the last 10000 as test set (standard split)
        X_test = X[60000:].astype(np.uint8)
        y_test = y[60000:].astype(int)
        
        print(f"  [OK] Loaded {len(X_test)} test images using sklearn")
        return X_test, y_test
        
    except ImportError:
        print("ERROR: Neither torchvision nor sklearn is installed!")
        print("Install with: pip install torchvision  OR  pip install scikit-learn")
        sys.exit(1)


def read_scores_from_fpga(ser):
    """Read all 10 class scores from FPGA via 0xCD protocol.
    
    Returns:
        numpy array of 10 int32 scores, or None on error
    """
    ser.reset_input_buffer()
    ser.write(SCORES_READ_REQUEST)
    
    # Read 40 bytes (10 scores × 4 bytes each)
    resp = ser.read(40)
    
    if len(resp) != 40:
        print(f"Error: Expected 40 bytes, got {len(resp)}")
        return None
    
    # Unpack as 10 signed 32-bit integers (little-endian)
    scores = struct.unpack('<10i', resp)
    return np.array(scores, dtype=np.int32)


def send_image_to_fpga(ser, image_bytes):
    """
    Send preprocessed image to FPGA via UART with Safety Delays.
    """
    # 1. Send Start Marker
    ser.write(IMG_START_MARKER)
    
    # 2. Send Image Data in Chunks
    chunk_size = 64  # Small chunks to prevent buffer overflow
    total_len = len(image_bytes)
    
    for i in range(0, total_len, chunk_size):
        chunk = image_bytes[i:i+chunk_size].tobytes()
        ser.write(chunk)
        # === THE MAGIC FIX: Tiny delay to let FPGA process ===
        time.sleep(0.002) 
        # ====================================================
        
    # 3. Send End Marker
    ser.write(IMG_END_MARKER)
    ser.flush()


def run_comparison(com_port, baud_rate, test_samples, output_mode, output_file):
    """
    Main comparison function.
    
    Args:
        com_port: Serial port for FPGA connection
        baud_rate: Serial baud rate
        test_samples: Number of images to test
        output_mode: 'file', 'console', or 'both'
        output_file: Path to output file (used if mode is 'file' or 'both')
    """
    
    # Load quantized model
    print("Loading quantized model...")
    L1_W, L1_B, L2_W, L2_B, L3_W, L3_B = load_quantized_model()
    print(f"  Layer 1: W={L1_W.shape}, B={L1_B.shape}")
    print(f"  Layer 2: W={L2_W.shape}, B={L2_B.shape}")
    print(f"  Layer 3: W={L3_W.shape}, B={L3_B.shape}")
    print()
    
    # Load normalization parameters
    print("Loading normalization parameters...")
    norm_mean, norm_std = load_norm_params()
    print(f"  norm_mean shape: {norm_mean.shape}")
    print(f"  norm_std shape: {norm_std.shape}")
    print()
    
    # Load MNIST test data
    X_test, y_test = load_mnist_test_data()
    print()
    
    # Limit to requested number of samples
    if test_samples > len(X_test):
        print(f"WARNING: Requested {test_samples} samples, but only {len(X_test)} available.")
        test_samples = len(X_test)
    
    X_test = X_test[:test_samples]
    y_test = y_test[:test_samples]
    
    # Prepare output file path
    if output_mode in ['file', 'both']:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        print(f"Output will be saved to: {output_path}")
    else:
        output_path = None
        print("Output mode: Console only")
    print()
    
    # Try to open serial connection
    print(f"Connecting to FPGA on {com_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=2)
        time.sleep(1)  # Allow FPGA to reset
        ser.reset_input_buffer()
        print("✓ Connected to FPGA")
        print()
    except serial.SerialException as e:
        print(f"✗ Serial Error: {e}")
        print("Cannot proceed without FPGA connection.")
        sys.exit(1)
    
    # Start comparison
    print(f"Starting comparison on {test_samples} images...")
    print("=" * 80)
    print()
    
    with OutputWriter(output_mode, output_path) as writer:
        # Write header
        writer.write("Python Simulation vs FPGA Hardware Comparison\n")
        writer.write("2-Hidden-Layer Neural Network (16-16-10)\n")
        writer.write("=" * 80 + "\n")
        writer.write(f"Total Images: {test_samples}\n")
        writer.write(f"Hardware Shifts: Layer1={SHIFT1}, Layer2={SHIFT2}\n")
        writer.write("=" * 80 + "\n\n")
        
        match_count = 0
        mismatch_count = 0
        error_count = 0
        
        for i in range(test_samples):
            label = y_test[i]
            
            # Preprocess image once
            img_int8 = preprocess_image(X_test[i], norm_mean, norm_std)
            img_uart = preprocess_for_uart(img_int8)
            
            # 1. Run Python simulation
            python_pred, python_scores, _ = fpga_inference_simulation(
                img_int8, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B
            )
            
            # 2. Send to FPGA and get hardware scores
            send_image_to_fpga(ser, img_uart)
            
            time.sleep(0.05)  # Wait for inference
            
            fpga_scores = read_scores_from_fpga(ser)
            
            # Format and write results
            if fpga_scores is None:
                writer.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                writer.write(f"                     | FPGA Pred: ERROR | FPGA:   ERROR - Failed to read scores\n\n")
                error_count += 1
            else:
                fpga_pred = np.argmax(fpga_scores)
                writer.write(f"Image {i:3d} | Label: {label} | Python Pred: {python_pred} | Python: {python_scores.tolist()}\n")
                writer.write(f"                     | FPGA Pred: {fpga_pred}   | FPGA:   {fpga_scores.tolist()}\n")
                
                # Check if they match
                if np.array_equal(python_scores, fpga_scores):
                    writer.write(f"                     | Status: ✓ MATCH (both predictions: {python_pred})\n\n")
                    match_count += 1
                else:
                    max_diff = np.max(np.abs(python_scores - fpga_scores))
                    writer.write(f"                     | Status: ✗ MISMATCH (max diff: {max_diff}, Python pred: {python_pred}, FPGA pred: {fpga_pred})\n\n")
                    mismatch_count += 1
            
            # Progress indicator (every 10 images) - always to console
            if (i + 1) % 10 == 0 or (i + 1) == test_samples:
                progress_msg = (f"  Processed: {i+1:3d}/{test_samples} images | "
                              f"Matches: {match_count} | Mismatches: {mismatch_count} | Errors: {error_count}")
                # Only print to console if not already in console/both mode (to avoid duplication)
                if output_mode == 'file':
                    print(progress_msg)
        
        # Write summary
        writer.write("=" * 80 + "\n")
        writer.write("SUMMARY\n")
        writer.write("=" * 80 + "\n")
        writer.write(f"Total images:  {test_samples}\n")
        writer.write(f"Matches:       {match_count} ({100.0 * match_count / test_samples:.2f}%)\n")
        writer.write(f"Mismatches:    {mismatch_count} ({100.0 * mismatch_count / test_samples:.2f}%)\n")
        writer.write(f"Errors:        {error_count}\n")
        writer.write("=" * 80 + "\n")
        writer.write("END OF COMPARISON\n")
        writer.write("=" * 80 + "\n")
    
    ser.close()
    
    # Final summary - always to console
    if output_mode != 'console':
        print()
        print("=" * 80)
        print(f"Comparison complete!")
        print(f"  Total images:  {test_samples}")
        print(f"  Matches:       {match_count} ({100.0 * match_count / test_samples:.2f}%)")
        print(f"  Mismatches:    {mismatch_count} ({100.0 * mismatch_count / test_samples:.2f}%)")
        print(f"  Errors:        {error_count}")
        if output_mode in ['file', 'both']:
            print(f"  Output saved to: {output_path}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare Python simulation vs FPGA hardware inference (2-hidden-layer network)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save to default file (backward compatible)
  python compare_python_vs_fpga.py
  
  # Print to console only
  python compare_python_vs_fpga.py --output console
  
  # Save to file and print to console
  python compare_python_vs_fpga.py --output both
  
  # Custom file path
  python compare_python_vs_fpga.py --output file --file results.txt
  
  # Test more samples
  python compare_python_vs_fpga.py --samples 200 --output both
  
  # Use different COM port
  python compare_python_vs_fpga.py --port COM5
        """
    )
    
    parser.add_argument('--port', type=str, default=DEFAULT_COM_PORT,
                        help=f'Serial COM port (default: {DEFAULT_COM_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD_RATE,
                        help=f'Baud rate (default: {DEFAULT_BAUD_RATE})')
    parser.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES,
                        help=f'Number of test samples (default: {DEFAULT_TEST_SAMPLES})')
    parser.add_argument('--output', type=str, 
                        choices=['file', 'console', 'both'], 
                        default='file',
                        help='Output mode: file (save to file), console (print), or both (default: file)')
    parser.add_argument('--file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output file path (default: {DEFAULT_OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    # Run comparison with specified arguments
    run_comparison(
        com_port=args.port,
        baud_rate=args.baud,
        test_samples=args.samples,
        output_mode=args.output,
        output_file=args.file
    )

