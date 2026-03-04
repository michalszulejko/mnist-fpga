"""
Send a test image to the FPGA for digit classification (2-hidden-layer network).

Protocol:
  - Start marker: 0xBB 0x66 (2 bytes)
  - Image data: 784 bytes (28x28 grayscale, 8-bit signed after preprocessing)
  - End marker: 0x66 0xBB (2 bytes)

Total transmission: 788 bytes

Image Processing (for custom images):
  - Image is resized to 20x20 pixels
  - Centered in a 28x28 black canvas (4px padding on each side)
  - Colors inverted if needed (MNIST expects white digit on black background)
  - This mimics how MNIST digits are formatted

Preprocessing (PyTorch-style):
  - Normalize to [0, 1] by dividing by 255
  - Apply normalization: (x - mean) / std where mean=0.1307, std=0.3081
  - Quantize to int8: multiply by 127 and clip to [-128, 127]

The image can come from:
  1. MNIST dataset (using torchvision or manually loading)
  2. A local image file (PNG/JPG - will be processed as above)
  3. A binary file (784 bytes raw)

Usage:
  python send_image.py [IMAGE_SOURCE] [COM_PORT] [BAUD_RATE] [--no-invert]
  
  IMAGE_SOURCE can be:
    - "mnist:INDEX" - Load from MNIST test set (e.g., "mnist:0" for first image)
    - path to .bin file (784 bytes raw)
    - path to .png/.jpg file (resized to 20x20, centered in 28x28)
  
  Options:
    --no-invert  Don't invert colors (use if image is already white-on-black)
  
Examples:
  python send_image.py                       # Default: ../../test_images/00001.png
  python send_image.py mnist:0               # Send first MNIST test image
  python send_image.py mnist:42 COM5         # Send 42nd image via COM5
  python send_image.py test.bin COM3 9600    # Send binary file
  python send_image.py digit.png             # Send PNG image (inverted)
  python send_image.py digit.png --no-invert # Send without inverting
"""

import serial
import time
import sys
import os
import numpy as np

# Protocol markers
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])

# Digit read request/response
DIGIT_READ_REQUEST = bytes([0xCC])

# Wait time before reading predicted digit (seconds) - easily changeable
READ_WAIT_TIME = 0.25

# Quantization scale (must match training script)
INPUT_SCALE = 127.0


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
        print("  Run siec_2_ukryte.py first to generate them.")
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
        print("Or use a .bin or .png file instead.")
        return None, None


def load_binary_image(filepath):
    """Load image from raw binary file (784 bytes)."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) != 784:
        print(f"WARNING: Expected 784 bytes, got {len(data)}")
    
    return np.frombuffer(data[:784], dtype=np.uint8), None


def load_image_file(filepath, target_size=20, invert=True):
    """
    Load image from PNG/JPG file and prepare for MNIST-style inference.
    
    MNIST format:
    - 28x28 grayscale image
    - White digit on black background
    - Digit is roughly 20x20 pixels, centered in the frame
    
    Args:
        filepath: Path to image file
        target_size: Size to resize the digit to (default 20, centered in 28x28)
        invert: If True, invert colors (for black digit on white background images)
    """
    try:
        from PIL import Image
        
        # Load and convert to grayscale
        img = Image.open(filepath).convert('L')
        original_size = img.size
        print(f"  Original size: {original_size[0]}x{original_size[1]}")
        
        # If image is already 28x28, assume it's MNIST format - use directly
        if original_size == (28, 28):
            print("  Image is already 28x28 MNIST format - using directly (no resize)")
            image_np = np.array(img, dtype=np.uint8)
            # Don't invert MNIST-format images by default
            if invert:
                print("  WARNING: Skipping inversion for MNIST-format image")
            return image_np.flatten(), None
        
        # For other sizes, resize to target size (e.g., 20x20) and center
        img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create 28x28 black canvas
        canvas = Image.new('L', (28, 28), color=0)  # Black background
        
        # Calculate position to center the digit
        offset = (28 - target_size) // 2  # (28-20)//2 = 4
        canvas.paste(img_resized, (offset, offset))
        
        # Convert to numpy array
        image_np = np.array(canvas, dtype=np.uint8)
        
        # Invert if needed (MNIST expects white digit on black background)
        if invert:
            image_np = 255 - image_np
            print("  Colors inverted (white digit on black background)")
        
        print(f"  Resized to: {target_size}x{target_size}, centered in 28x28")
        
        return image_np.flatten(), None
        
    except ImportError:
        print("ERROR: Pillow not installed. Install with: pip install Pillow")
        return None, None


def display_ascii_image(image_data, label=None):
    """Display image as ASCII art in terminal."""
    chars = " .:-=+*#%@"  # 10 levels of brightness
    
    print("\n" + "=" * 32)
    if label is not None:
        print(f"Label: {label}")
    print("=" * 32)
    
    for row in range(28):
        line = ""
        for col in range(28):
            pixel = image_data[row * 28 + col]
            char_idx = min(int(pixel) * len(chars) // 256, len(chars) - 1)
            line += chars[char_idx] * 2  # Double width for aspect ratio
        print(line)
    print("=" * 32)


def send_image(ser, image_data):
    """Send image data over serial."""
    # Send start marker
    print("Sending start marker (0xBB 0x66)...")
    ser.write(IMG_START_MARKER)
    time.sleep(0.05)
    
    # Send image data in chunks
    chunk_size = 64
    total = len(image_data)
    sent = 0
    
    print("Sending image: ", end="", flush=True)
    
    while sent < total:
        end = min(sent + chunk_size, total)
        chunk = bytes(image_data[sent:end])
        ser.write(chunk)
        sent = end
        
        progress = int(30 * sent / total)
        print(f"\rSending image: [{'=' * progress}{' ' * (30 - progress)}] {sent}/{total}", end="", flush=True)
        time.sleep(0.01)
    
    print()
    
    # Send end marker
    print("Sending end marker (0x66 0xBB)...")
    ser.write(IMG_END_MARKER)
    
    ser.flush()


def main():
    # Parse arguments (filter out flags)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    
    # Check for --no-invert flag
    invert_colors = '--no-invert' not in flags
    
    # Default source is 00001.png in test_images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(script_dir, "..", "..", "test_images", "00004.png")
    
    source = args[0] if len(args) > 0 else default_source
    port = args[1] if len(args) > 1 else "COM3"
    baud = int(args[2]) if len(args) > 2 else 115200
    
    print(f"Two-Hidden-Layer Network Image Upload")
    print(f"=" * 50)
    print(f"Source: {source}")
    print(f"Port:   {port}")
    print(f"Baud:   {baud}")
    print(f"Invert: {invert_colors}")
    print(f"=" * 50)
    
    # Load image based on source type
    label = None
    
    if source.startswith("mnist:"):
        try:
            index = int(source.split(":")[1])
            print(f"\nLoading MNIST test image #{index}...")
            image_data, label = load_mnist_image(index)
        except ValueError:
            print("ERROR: Invalid MNIST index format. Use 'mnist:NUMBER'")
            return 1
    elif source.endswith(".bin"):
        print(f"\nLoading binary file: {source}")
        image_data, label = load_binary_image(source)
    elif source.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        print(f"\nLoading image file: {source}")
        # Check if file exists
        if not os.path.exists(source):
            print(f"ERROR: File not found: {source}")
            return 1
        image_data, label = load_image_file(source, target_size=20, invert=invert_colors)
    else:
        print(f"ERROR: Unknown source format: {source}")
        print("Use 'mnist:INDEX', a .bin file, or an image file (.png, .jpg)")
        return 1
    
    if image_data is None:
        return 1
    
    # Display the original image
    display_ascii_image(image_data, label)
    
    # Load normalization parameters and apply preprocessing
    print("\nApplying preprocessing (PyTorch normalization + quantization)...")
    norm_mean, norm_std = load_norm_params()
    
    if norm_mean is None:
        print("ERROR: Cannot proceed without normalization parameters.")
        return 1
    
    print(f"  Mean: {norm_mean[0]:.4f}, Std: {norm_std[0]:.4f}")
    
    # Apply preprocessing: normalize, scale, quantize
    image_data = apply_preprocessing(image_data, norm_mean, norm_std)
    print(f"  Preprocessed data range: [{np.frombuffer(image_data, dtype=np.int8).min()}, {np.frombuffer(image_data, dtype=np.int8).max()}]")
    
    # Calculate transmission time
    total_bytes = 2 + 784 + 2  # markers + data
    estimated_time = total_bytes * 10 / baud
    print(f"\nEstimated transmission time: {estimated_time:.2f} seconds")
    
    try:
        # Open serial port
        print(f"\nOpening {port}...")
        ser = serial.Serial(port, baud, timeout=1)
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
        print(f"\nThe FPGA should now perform inference.")
        print(f"Waiting {READ_WAIT_TIME} seconds before reading result...")
        
        # Wait for inference to complete
        time.sleep(READ_WAIT_TIME)
        
        # Request predicted digit
        print("Requesting predicted digit...")
        ser.reset_input_buffer()  # Clear any leftover data
        ser.write(DIGIT_READ_REQUEST)
        ser.flush()
        
        # Wait for and read response
        response_timeout = 1.0  # 1 second timeout
        start_time = time.time()
        
        while ser.in_waiting == 0:
            if time.time() - start_time > response_timeout:
                print("ERROR: Timeout waiting for predicted digit response")
                ser.close()
                return 1
            time.sleep(0.01)
        
        # Read the response byte
        response = ser.read(1)
        if len(response) == 1:
            predicted_digit = response[0] & 0x0F  # Extract lower 4 bits (0-9)
            print(f"\n{'=' * 50}")
            print(f"Predicted digit: {predicted_digit}")
            print(f"{'=' * 50}")
            
            if label is not None:
                if predicted_digit == label:
                    print(f"✓ Correct! (Expected: {label})")
                else:
                    print(f"✗ Incorrect (Expected: {label}, Got: {predicted_digit})")
        else:
            print("ERROR: Failed to read predicted digit response")
            ser.close()
            return 1
        
        print(f"\nCheck the 7-segment display for the predicted digit.")
        
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

