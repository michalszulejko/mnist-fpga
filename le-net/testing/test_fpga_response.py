"""
Simple diagnostic script to test FPGA response with varying wait times.
"""
import serial
import time
import os
import numpy as np
from torchvision import datasets, transforms

DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200

# Protocol markers
IMG_START = bytes([0xBB, 0x66])
IMG_END   = bytes([0x66, 0xBB])
CMD_READ_DIGIT = bytes([0xCC])

def load_norm_params():
    """Load normalization parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(script_dir, "..", "outputs", "npy")
    mean = np.load(os.path.join(npy_path, "norm_mean.npy"))
    std = np.load(os.path.join(npy_path, "norm_std.npy"))
    return mean, std

def preprocess(image, mean, std):
    """Preprocess image."""
    x = image.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)
    return x.view(np.uint8)

def get_mnist_image(index=0):
    """Load MNIST test image."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "..", "data")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    img, label = dataset[index]
    img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
    return img_np, label

def main():
    print("="*60)
    print("FPGA Diagnostic Test - LeNet-5")
    print("="*60)

    # Load data
    print("\n1. Loading test data...")
    mean, std = load_norm_params()
    img_raw, label = get_mnist_image(0)
    img_input = preprocess(img_raw, mean, std)
    print(f"   Image 0, True Label: {label}")

    # Open serial
    print(f"\n2. Opening {DEFAULT_PORT}...")
    try:
        ser = serial.Serial(DEFAULT_PORT, DEFAULT_BAUD, timeout=5)
        time.sleep(2)
        print("   Serial port opened successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Send image
    print("\n3. Sending image to FPGA...")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    ser.write(IMG_START)

    # Send in chunks
    img_bytes = img_input.tobytes()
    for i in range(0, len(img_bytes), 32):
        ser.write(img_bytes[i:i+32])
        ser.flush()
        time.sleep(0.010)

    ser.write(IMG_END)
    print("   Image sent")

    # Try different wait times
    wait_times = [0.5, 1.0, 2.0, 5.0, 10.0]

    for wait_time in wait_times:
        print(f"\n4. Waiting {wait_time}s for inference...")
        time.sleep(wait_time)

        print(f"5. Requesting digit (0xCC)...")
        ser.reset_input_buffer()
        ser.write(CMD_READ_DIGIT)
        ser.flush()
        time.sleep(0.2)

        response = ser.read(1)

        if len(response) > 0:
            pred = response[0] & 0x0F  # Lower 4 bits
            status = "MATCH!" if pred == label else f"mismatch (expected {label})"
            print(f"   OK RESPONSE RECEIVED: Predicted = {pred} [{status}]")
            break
        else:
            print(f"   X No response (waited {wait_time}s)")

    if len(response) == 0:
        print("\n" + "="*60)
        print("FPGA DID NOT RESPOND")
        print("="*60)
        print("Possible issues:")
        print("  1. Weights not loaded correctly")
        print("  2. FPGA inference module stuck")
        print("  3. UART routing issue")
        print("  4. Image not being processed")

    ser.close()

if __name__ == "__main__":
    main()
