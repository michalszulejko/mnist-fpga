"""
FPGA Accuracy Test - 2-Hidden-Layer Neural Network

This script tests the FPGA's inference accuracy on 1000 MNIST test images by:
1. Loading the first 1000 images from MNIST test dataset
2. Preprocessing each image using saved normalization parameters
3. Sending each image to the FPGA via UART
4. Reading the predicted digit from predicted_digit_ram via 0xCC protocol
5. Comparing predictions with true labels
6. Computing accuracy and confusion matrix

Protocol:
  - Image Send: 0xBB 0x66 + 784 bytes + 0x66 0xBB
  - Digit Read: Send 0xCC, receive 1 byte (predicted digit in lower 4 bits)

Usage:
  python test_fpga_accuracy.py [options]
  
  Options:
    --port PORT       Serial COM port (default: COM3)
    --baud BAUD       Baud rate (default: 115200)
    --samples N       Number of test images (default: 1000)
    --output MODE     Output mode: file, console, or both (default: both)
    --wait SECONDS    Wait time after sending image (default: 0.25)
    --file PATH       Output file path (default: ../outputs/txt/accuracy_test_results.txt)

Examples:
  python test_fpga_accuracy.py
  python test_fpga_accuracy.py --samples 500 --output console
  python test_fpga_accuracy.py --port COM5 --wait 0.3
"""

import serial
import time
import sys
import os
import struct
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
DIGIT_READ_REQUEST = bytes([0xCC])

# Quantization scale (must match training)
INPUT_SCALE = 127.0

# Default Configuration
DEFAULT_COM_PORT = 'COM3'
DEFAULT_BAUD_RATE = 115200
DEFAULT_TEST_SAMPLES = 100
DEFAULT_WAIT_TIME = 0.25
DEFAULT_OUTPUT_FILE = '../outputs/txt/accuracy_test_results.txt'


class OutputWriter:
    """Handles output to file and/or console."""
    
    def __init__(self, output_mode='both', file_path=None):
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


def load_norm_params():
    """Load PyTorch normalization parameters saved during training."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "outputs", "npy")
    mean_path = os.path.join(data_dir, "norm_mean.npy")
    std_path = os.path.join(data_dir, "norm_std.npy")
    
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
    5. Convert to uint8 for UART transmission (two's complement)
    
    Args:
        image_data: Raw pixel values (0-255), shape (784,)
        norm_mean: Normalization mean
        norm_std: Normalization std
    
    Returns:
        Preprocessed image as uint8 array for UART transmission
    """
    # Convert to float and normalize to [0, 1]
    x = image_data.astype(np.float32) / 255.0
    
    # Apply PyTorch normalization
    x_normalized = (x - norm_mean) / norm_std
    
    # Quantize to int8 range
    x_quantized = np.round(x_normalized * INPUT_SCALE)
    
    # Clip to int8 range and convert (handle signed values)
    x_int8 = np.clip(x_quantized, -128, 127).astype(np.int8)
    
    # Convert to unsigned bytes for transmission (two's complement)
    x_bytes = x_int8.view(np.uint8)
    
    return x_bytes


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


def send_image_to_fpga(ser, image_bytes):
    """
    Send preprocessed image to FPGA via UART.
    
    Args:
        ser: Serial connection
        image_bytes: Preprocessed image as uint8 array (784 bytes)
    """
    ser.write(IMG_START_MARKER)
    ser.write(image_bytes.tobytes())
    ser.write(IMG_END_MARKER)
    ser.flush()


def read_predicted_digit(ser):
    """
    Read predicted digit from FPGA via 0xCC protocol.
    
    Args:
        ser: Serial connection
    
    Returns:
        Predicted digit (0-9), or None on error
    """
    ser.reset_input_buffer()
    ser.write(DIGIT_READ_REQUEST)
    
    # Read 1 byte response
    response = ser.read(1)
    
    if len(response) != 1:
        return None
    
    # Extract lower 4 bits
    predicted_digit = response[0] & 0x0F
    
    return predicted_digit


def run_accuracy_test(com_port, baud_rate, test_samples, output_mode, output_file, wait_time):
    """
    Main test function.
    
    Args:
        com_port: Serial port for FPGA connection
        baud_rate: Serial baud rate
        test_samples: Number of images to test
        output_mode: 'file', 'console', or 'both'
        output_file: Path to output file (used if mode is 'file' or 'both')
        wait_time: Wait time after sending image (seconds)
    """
    
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
    
    # Connect to FPGA
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
    
    # Start testing
    print(f"Starting accuracy test on {test_samples} images...")
    print("=" * 80)
    print()
    
    with OutputWriter(output_mode, output_path) as writer:
        # Write header
        writer.write("FPGA Accuracy Test - 2-Hidden-Layer Neural Network\n")
        writer.write("=" * 80 + "\n")
        writer.write(f"Total Images: {test_samples}\n")
        writer.write(f"COM Port: {com_port}\n")
        writer.write(f"Baud Rate: {baud_rate}\n")
        writer.write(f"Wait Time: {wait_time}s\n")
        writer.write("=" * 80 + "\n\n")
        
        y_pred = []
        y_true = []
        error_count = 0
        
        for i in range(test_samples):
            label = y_test[i]
            
            # Preprocess image
            img_preprocessed = preprocess_image(X_test[i], norm_mean, norm_std)
            
            # Send to FPGA
            send_image_to_fpga(ser, img_preprocessed)
            
            # Wait for inference
            time.sleep(wait_time)
            
            # Read predicted digit
            predicted = read_predicted_digit(ser)
            
            if predicted is None:
                writer.write(f"Image {i:4d} | Label: {label} | Predicted: ERROR | ✗\n")
                error_count += 1
                # Don't add to accuracy calculation
            else:
                y_pred.append(predicted)
                y_true.append(label)
                
                # Check if correct
                is_correct = (predicted == label)
                status = "✓" if is_correct else "✗"
                
                writer.write(f"Image {i:4d} | Label: {label} | Predicted: {predicted} | {status}\n")
            
            # Progress indicator (every 50 images) - always to console
            if (i + 1) % 50 == 0 or (i + 1) == test_samples:
                if output_mode == 'file':
                    # Only print to console if not already printing
                    correct_so_far = sum([1 for p, t in zip(y_pred, y_true) if p == t])
                    total_so_far = len(y_pred)
                    acc_so_far = 100.0 * correct_so_far / total_so_far if total_so_far > 0 else 0.0
                    progress_msg = (f"  Progress: {i+1:4d}/{test_samples} | "
                                  f"Correct: {correct_so_far}/{total_so_far} | "
                                  f"Accuracy: {acc_so_far:.2f}% | Errors: {error_count}")
                    print(progress_msg)
        
        # Compute final accuracy and confusion matrix
        writer.write("\n")
        writer.write("=" * 80 + "\n")
        writer.write("RESULTS\n")
        writer.write("=" * 80 + "\n")
        
        if len(y_pred) == 0:
            writer.write("ERROR: No valid predictions received!\n")
        else:
            accuracy = accuracy_score(y_true, y_pred) * 100
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            correct = sum([1 for p, t in zip(y_pred, y_true) if p == t])
            total = len(y_pred)
            
            writer.write(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")
            writer.write(f"Errors: {error_count} UART/read errors\n")
            writer.write(f"Total tested: {test_samples} images\n")
            writer.write("\n")
            
            writer.write("Confusion Matrix:\n")
            writer.write("(Rows = True Labels, Columns = Predicted Labels)\n")
            writer.write("\n")
            writer.write("     ")
            for i in range(10):
                writer.write(f"{i:5d} ")
            writer.write("\n")
            writer.write("    " + "-" * 66 + "\n")
            
            for i in range(10):
                writer.write(f"{i:2d} | ")
                for j in range(10):
                    writer.write(f"{conf_matrix[i, j]:5d} ")
                writer.write("\n")
            writer.write("\n")
            
            # Show some misclassifications
            writer.write("Sample Misclassifications (first 20):\n")
            writer.write("-" * 80 + "\n")
            misclass_count = 0
            for i in range(len(y_true)):
                if y_pred[i] != y_true[i]:
                    misclass_count += 1
                    if misclass_count <= 20:
                        writer.write(f"  Image {i:5d}: True={y_true[i]}, Predicted={y_pred[i]}\n")
            
            if misclass_count == 0:
                writer.write("  No misclassifications found!\n")
            else:
                writer.write(f"\nTotal misclassifications: {misclass_count}/{len(y_true)}\n")
            
            writer.write("\n")
        
        writer.write("=" * 80 + "\n")
        writer.write("END OF TEST\n")
        writer.write("=" * 80 + "\n")
    
    ser.close()
    
    # Final summary - always to console
    if output_mode != 'console':
        print()
        print("=" * 80)
        print("Test complete!")
        if len(y_pred) > 0:
            accuracy = accuracy_score(y_true, y_pred) * 100
            correct = sum([1 for p, t in zip(y_pred, y_true) if p == t])
            total = len(y_pred)
            print(f"  Accuracy:      {accuracy:.2f}% ({correct}/{total} correct)")
            print(f"  Errors:        {error_count}")
            print(f"  Total tested:  {test_samples}")
        if output_mode in ['file', 'both']:
            print(f"  Output saved to: {output_path}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test FPGA inference accuracy on MNIST images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default test (1000 images, output to both file and console)
  python test_fpga_accuracy.py
  
  # Test 500 images, console output only
  python test_fpga_accuracy.py --samples 500 --output console
  
  # Use different COM port and wait time
  python test_fpga_accuracy.py --port COM5 --wait 0.3
  
  # Custom output file
  python test_fpga_accuracy.py --file my_results.txt
        """
    )
    
    parser.add_argument('--port', type=str, default=DEFAULT_COM_PORT,
                        help=f'Serial COM port (default: {DEFAULT_COM_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD_RATE,
                        help=f'Baud rate (default: {DEFAULT_BAUD_RATE})')
    parser.add_argument('--samples', type=int, default=DEFAULT_TEST_SAMPLES,
                        help=f'Number of test images (default: {DEFAULT_TEST_SAMPLES})')
    parser.add_argument('--output', type=str, 
                        choices=['file', 'console', 'both'], 
                        default='both',
                        help='Output mode: file, console, or both (default: both)')
    parser.add_argument('--wait', type=float, default=DEFAULT_WAIT_TIME,
                        help=f'Wait time after sending image in seconds (default: {DEFAULT_WAIT_TIME})')
    parser.add_argument('--file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output file path (default: {DEFAULT_OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    # Run test with specified arguments
    run_accuracy_test(
        com_port=args.port,
        baud_rate=args.baud,
        test_samples=args.samples,
        output_mode=args.output,
        output_file=args.file,
        wait_time=args.wait
    )

