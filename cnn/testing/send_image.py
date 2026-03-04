"""
Send a test image to the FPGA for CNN digit classification.

Protocol:
  - Start marker: 0xBB 0x66 (2 bytes)
  - Image data: 784 bytes (28x28 grayscale, preprocessed)
  - End marker: 0x66 0xBB (2 bytes)

Total transmission: 788 bytes

Preprocessing (PyTorch-style):
  - Normalize to [0, 1] by dividing by 255
  - Apply normalization: (x - mean) / std where mean=0.1307, std=0.3081
  - Quantize to int8: multiply by 127 and clip to [-128, 127]

Usage:
  python send_image.py [INDEX] [--port PORT] [--baud BAUD]
  
  INDEX: MNIST test image index (0-9999), default: 0

Examples:
  python send_image.py 0
  python send_image.py 42 --port COM7
  python send_image.py 100 --port COM3 --baud 115200
"""

import serial
import time
import sys
import os
import numpy as np

# Protocol markers
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])

# Quantization scale (must match training script)
INPUT_SCALE = 127.0

# Default configuration
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200


def load_norm_params():
    """Load PyTorch normalization parameters saved during training."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "outputs", "npy")
    mean_path = os.path.join(data_dir, "norm_mean.npy")
    std_path = os.path.join(data_dir, "norm_std.npy")
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("WARNING: Normalization files not found!")
        print(f"  Expected: {mean_path}")
        print(f"  Expected: {std_path}")
        print("  Run train_cnn.py first to generate them.")
        return None, None
    
    norm_mean = np.load(mean_path)
    norm_std = np.load(std_path)
    return norm_mean, norm_std


def apply_preprocessing(image_data, norm_mean, norm_std):
    """
    Apply the same preprocessing as during training:
    1. Normalize to 0-1 (divide by 255)
    2. Apply PyTorch normalization: (x - mean) / std
    3. Quantize to int8: multiply by INPUT_SCALE (127)
    """
    # Convert to float and normalize to 0-1
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


def load_mnist_image(index):
    """Load a single image from MNIST test set using torchvision."""
    try:
        from torchvision import datasets, transforms
        
        # Download MNIST if not present
        transform = transforms.Compose([transforms.ToTensor()])
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, "..", "..", "data")
        mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        
        if index < 0 or index >= len(mnist_test):
            print(f"ERROR: Index {index} out of range (0-{len(mnist_test)-1})")
            return None, None
        
        image, label = mnist_test[index]
        
        # Convert to numpy array (28x28), scale to 0-255
        image_np = (image.squeeze().numpy() * 255).astype(np.uint8)
        
        return image_np.flatten(), label
        
    except ImportError:
        print("ERROR: torchvision not installed. Install with: pip install torchvision")
        return None, None


def send_chunked(ser, data, chunk_size=32, delay=0.010):
    """Send data in chunks with flow control delay."""
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        ser.write(chunk)
        ser.flush()  # Ensure transmission completes before next chunk
        time.sleep(delay)


def send_image(ser, image_bytes):
    """Send image with protocol markers."""
    # Start marker
    ser.write(IMG_START_MARKER)
    # Data with chunking
    send_chunked(ser, image_bytes.tobytes())
    # End marker
    ser.write(IMG_END_MARKER)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Send MNIST image to FPGA')
    parser.add_argument('index', type=int, nargs='?', default=0, help='MNIST test image index (0-9999), default: 0')
    parser.add_argument('--port', type=str, default=DEFAULT_PORT, help=f'Serial port (default: {DEFAULT_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD, help=f'Baud rate (default: {DEFAULT_BAUD})')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading MNIST test image {args.index}...")
    image_data, label = load_mnist_image(args.index)
    
    if image_data is None:
        return 1
    
    if label is not None:
        print(f"True label: {label}")
    
    # Load normalization parameters and apply preprocessing
    print("\nApplying preprocessing (MNIST normalization + quantization)...")
    norm_mean, norm_std = load_norm_params()
    if norm_mean is None:
        return 1
    
    print(f"  Normalization: mean={norm_mean[0]:.4f}, std={norm_std[0]:.4f}")
    
    # Apply preprocessing: normalize, scale, quantize
    image_data = apply_preprocessing(image_data, norm_mean, norm_std)
    print(f"  Preprocessed data range: [{np.frombuffer(image_data, dtype=np.int8).min()}, {np.frombuffer(image_data, dtype=np.int8).max()}]")
    
    # Calculate transmission time
    total_bytes = 2 + 784 + 2  # markers + data
    estimated_time = total_bytes * 10 / args.baud
    print(f"\nEstimated transmission time: {estimated_time:.2f} seconds")
    
    try:
        # Open serial port
        print(f"\nOpening {args.port}...")
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(0.5)
        
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        start_time = time.time()
        
        # Send the image
        send_image(ser, image_data)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'=' * 50}")
        print(f"Image sent successfully!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        
        # Wait for FPGA to process the image
        print(f"\nWaiting for FPGA to process image...")
        time.sleep(0.5)
        
        # Verify by requesting the predicted digit
        print(f"Requesting predicted digit from FPGA...")
        ser.reset_input_buffer()
        ser.write(bytes([0xCC]))  # Request digit command
        ser.flush()
        
        # Read response (should be 1 byte with the digit)
        time.sleep(0.1)  # Small delay for response
        response = ser.read(1)
        
        if len(response) > 0:
            predicted_digit = response[0]
            print(f"FPGA predicted digit: {predicted_digit}")
            if label is not None:
                match_str = "✓ MATCH" if predicted_digit == label else "✗ MISMATCH"
                print(f"Expected digit: {label} {match_str}")
        else:
            print(f"WARNING: No response received from FPGA")
            print(f"  This may indicate the image was not processed correctly.")
        
        print(f"\nThe FPGA should now perform inference.")
        print(f"Check the 7-segment display for the predicted digit.")
        if label is not None:
            print(f"Expected result: {label}")
        
        ser.close()
        return 0
        
    except serial.SerialException as e:
        print(f"\nERROR: Could not open serial port: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTransfer cancelled by user.")
        if 'ser' in locals() and ser.is_open:
            ser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
