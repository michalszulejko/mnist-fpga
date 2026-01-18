import serial
import time
import os
import struct
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms
from tanh_lut_loader import load_tanh_lut, apply_tanh_lut

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_PORT = "COM7"  # Update this to your specific COM port
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"
NPY_DIR = "../outputs/npy"
OUTPUT_FILE = "../outputs/txt/comparison.txt"

# Protocol Markers
IMG_START = bytes([0xBB, 0x66])
IMG_END   = bytes([0x66, 0xBB])
CMD_READ_SCORES = bytes([0xCD])

# Shift parameters (Must match Training & FPGA)
# Per-layer SHIFT values - MUST MATCH inference.v and generate_lenet_vectors.py
SHIFT_CONV1 = 10  # inference.v L1_TANH: acc >>> 10
SHIFT_CONV2 = 8   # inference.v L2_TANH: acc >>> 8
SHIFT_FC1 = 9     # inference.v FC1_TANH: acc >>> 9
SHIFT_FC2 = 10    # inference.v FC2_TANH: acc >>> 10
# FC3 has no shift (output layer)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def avg_pool_2x2(input_vol):
    """
    Simulates 2x2 Average Pooling with Stride 2.
    Input: (Channels, Height, Width)
    Output: (Channels, Height/2, Width/2)
    """
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output_vol = np.zeros((c, new_h, new_w), dtype=np.int32)
    
    for ch in range(c):
        for r in range(new_h):
            for col in range(new_w):
                # Extract 2x2 window and compute average
                window = input_vol[ch, r*2 : r*2+2, col*2 : col*2+2]
                output_vol[ch, r, col] = np.sum(window) // 4
                
    return output_vol

# REMOVED: Old tanh_lut function that computed on-the-fly
# Now using the actual LUT from tanh_lut.mem via tanh_lut_loader module

def convolve_layer(input_vol, weights, biases, shift, tanh_lut, use_tanh=True):
    """
    LeNet-5 Convolution with 5x5 kernel - Bit-exact simulation of FPGA.
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 5, 5)
    Output: (Out_Channels, H-4, W-4) for no padding
           (Out_Channels, H, W) for padding=2
    """
    out_ch, in_ch, k, _ = weights.shape
    h, w = input_vol.shape[1], input_vol.shape[2]

    # Determine if padding is needed (Conv1 with padding=2)
    if k == 5 and h == 28 and w == 28:  # Conv1
        input_padded = np.pad(input_vol, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
        out_h, out_w = h, w
    else:  # Conv2 or other
        input_padded = input_vol
        out_h, out_w = h - k + 1, w - k + 1

    output_vol = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = np.int32(bias_val)
                for ch in range(in_ch):
                    window = input_padded[ch, r:r+k, c:c+k].astype(np.int32)
                    kernel = weights[f, ch].astype(np.int32)
                    acc += np.sum(window * kernel)

                acc = acc >> shift

                # Apply Tanh activation
                if use_tanh:
                    acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)

                output_vol[f, r, c] = acc

    return output_vol

def fc_layer(input_vec, weights, biases, shift, tanh_lut, use_tanh=True):
    """
    Fully Connected Layer - Bit-exact simulation of FPGA.
    Input: (N,) 1D vector
    Weights: (Out, In)
    Output: (Out,) 1D vector
    """
    out_size = weights.shape[0]
    output = np.zeros(out_size, dtype=np.int32)

    for i in range(out_size):
        acc = np.int32(biases[i])
        acc += np.sum(input_vec.astype(np.int32) * weights[i, :].astype(np.int32))
        acc = acc >> shift

        # Apply Tanh activation if specified
        if use_tanh:
            acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)

        output[i] = acc

    return output

# ==========================================
# 2. BIT-EXACT SIMULATION (LeNet-5)
# ==========================================
def simulate_lenet_inference(image, weights_dict, tanh_lut):
    """
    Simulates the LeNet-5 FPGA pipeline exactly.
    """
    # Unpack weights
    c1_w, c1_b = weights_dict['c1']
    c2_w, c2_b = weights_dict['c2']
    fc1_w, fc1_b = weights_dict['fc1']
    fc2_w, fc2_b = weights_dict['fc2']
    fc3_w, fc3_b = weights_dict['fc3']

    # 1. Input Image: (1, 28, 28)
    img_3d = image.reshape(1, 28, 28).astype(np.int32)
    
    # 2. Layer 1: Conv (5x5, padding=2) -> Tanh -> Avg Pool
    # Weights: (6 filters, 1 ch, 5, 5)
    c1_w_reshaped = c1_w.reshape(6, 1, 5, 5).astype(np.int32)
    
    # Conv Output: (6, 28, 28)
    x = convolve_layer(img_3d, c1_w_reshaped, c1_b, SHIFT_CONV1, tanh_lut, use_tanh=True)
    # Pool Output: (6, 14, 14)
    x = avg_pool_2x2(x)

    # 3. Layer 2: Conv (5x5, no padding) -> Tanh -> Avg Pool
    # Weights: (16 filters, 6 ch, 5, 5)
    c2_w_reshaped = c2_w.reshape(16, 6, 5, 5).astype(np.int32)
    
    # Conv Output: (16, 10, 10)
    x = convolve_layer(x, c2_w_reshaped, c2_b, SHIFT_CONV2, tanh_lut, use_tanh=True)
    # Pool Output: (16, 5, 5)
    x = avg_pool_2x2(x)
    
    # 4. Flatten: (16, 5, 5) -> 400
    flattened = x.flatten().astype(np.int32)
    
    # 5. FC1: 400 -> 120 with Tanh
    fc1_w_reshaped = fc1_w.reshape(120, 400).astype(np.int32)
    x = fc_layer(flattened, fc1_w_reshaped, fc1_b, SHIFT_FC1, tanh_lut, use_tanh=True)
    
    # 6. FC2: 120 -> 84 with Tanh
    fc2_w_reshaped = fc2_w.reshape(84, 120).astype(np.int32)
    x = fc_layer(x, fc2_w_reshaped, fc2_b, SHIFT_FC2, tanh_lut, use_tanh=True)
    
    # 7. FC3: 84 -> 10 (no activation, no shift)
    fc3_w_reshaped = fc3_w.reshape(10, 84).astype(np.int32)
    scores = np.zeros(10, dtype=np.int32)
    for i in range(10):
        scores[i] = np.int32(fc3_b[i]) + np.dot(x.astype(np.int32), fc3_w_reshaped[i].astype(np.int32))
    
    return scores

# ==========================================
# 3. UTILITIES
# ==========================================
def load_files():
    """Load weights and normalization parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    npy_path = os.path.join(script_dir, NPY_DIR)

    # Load Weights
    try:
        c1_w = np.fromfile(os.path.join(bin_path, "conv1_weights.bin"), dtype=np.int8)
        c1_b = np.fromfile(os.path.join(bin_path, "conv1_biases.bin"), dtype=np.int32)
        c2_w = np.fromfile(os.path.join(bin_path, "conv2_weights.bin"), dtype=np.int8)
        c2_b = np.fromfile(os.path.join(bin_path, "conv2_biases.bin"), dtype=np.int32)
        fc1_w = np.fromfile(os.path.join(bin_path, "fc1_weights.bin"), dtype=np.int8)
        fc1_b = np.fromfile(os.path.join(bin_path, "fc1_biases.bin"), dtype=np.int32)
        fc2_w = np.fromfile(os.path.join(bin_path, "fc2_weights.bin"), dtype=np.int8)
        fc2_b = np.fromfile(os.path.join(bin_path, "fc2_biases.bin"), dtype=np.int32)
        fc3_w = np.fromfile(os.path.join(bin_path, "fc3_weights.bin"), dtype=np.int8)
        fc3_b = np.fromfile(os.path.join(bin_path, "fc3_biases.bin"), dtype=np.int32)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Weight file not found! {e}")
        print("Did you run `train_lenet.py` to generate the weights?")
        exit(1)

    # Load Norm Params
    mean = np.load(os.path.join(npy_path, "norm_mean.npy"))
    std = np.load(os.path.join(npy_path, "norm_std.npy"))

    weights = {
        'c1': (c1_w, c1_b),
        'c2': (c2_w, c2_b),
        'fc1': (fc1_w, fc1_b),
        'fc2': (fc2_w, fc2_b),
        'fc3': (fc3_w, fc3_b)
    }
    return weights, (mean, std)

def get_mnist_data(num_images=5):
    """Load first N images from MNIST test set."""
    transform = transforms.Compose([transforms.ToTensor()])
    # Adjust root path if needed
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    images = []
    labels = []
    for i in range(num_images):
        img, label = dataset[i]
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
        images.append(img_np)
        labels.append(label)
    return images, labels

def preprocess(image, mean, std):
    """Normalize and Quantize Image (matches FPGA input format)."""
    x = image.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)
    return x

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("--index", type=int, default=0, help="Start index in MNIST test set")
    parser.add_argument("--count", type=int, default=3000, help="Number of images to test")
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, OUTPUT_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    output_file = open(log_path, "w", encoding='utf-8')

    print("Loading weights and data...")
    weights, (mean, std) = load_files()
    
    # Load Tanh LUT (same file FPGA uses)
    print("Loading Tanh LUT...")
    tanh_lut = load_tanh_lut()
    
    # Load enough images to cover the requested range
    required_images = args.index + args.count
    images, labels = get_mnist_data(required_images)

    # Open UART
    try:
        print(f"Opening {args.port} at {args.baud}...")
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2) # Wait for DTR/RTS
    except Exception as e:
        print(f"Error opening {args.port}: {e}")
        output_file.close()
        return

    print(f"Running comparison on {args.port}...")
    print(f"Testing {args.count} images starting from index {args.index}")
    
    output_file.write(f"Python (LeNet-5) vs FPGA Comparison\n")
    output_file.write(f"Start Index: {args.index}, Count: {args.count}\n")
    output_file.write("=" * 80 + "\n\n")

    correct_matches = 0
    fpga_accuracy_correct = 0
    python_accuracy_correct = 0

    for idx in range(args.count):
        real_idx = args.index + idx
        if real_idx >= len(images): break

        img_raw = images[real_idx]
        label = labels[real_idx]
        img_input = preprocess(img_raw, mean, std)
        
        # --- A. PYTHON SIMULATION ---
        expected_scores = simulate_lenet_inference(img_input, weights, tanh_lut)
        py_pred = np.argmax(expected_scores)

        # --- B. FPGA INFERENCE ---
        ser.reset_input_buffer()
        ser.write(IMG_START)
        
        # Chunking
        img_bytes = img_input.tobytes()
        for i in range(0, len(img_bytes), 64):
            ser.write(img_bytes[i:i+64])
            time.sleep(0.005) 
            
        ser.write(IMG_END)
        time.sleep(0.20) # Wait for inference (increased from 0.15)

        ser.write(CMD_READ_SCORES)
        time.sleep(0.01)  # Small delay before reading
        response = ser.read(40)
        
        if len(response) != 40:
            print(f"Index {real_idx}: Timeout/Error receiving scores.")
            output_file.write(f"Image {real_idx} | TIMEOUT\n")
            continue
            
        fpga_scores = np.array(struct.unpack('<10i', response), dtype=np.int32)
        fpga_pred = np.argmax(fpga_scores)

        # --- C. COMPARE ---
        is_match = np.array_equal(expected_scores, fpga_scores)
        if is_match: correct_matches += 1
        
        if fpga_pred == label: fpga_accuracy_correct += 1
        if py_pred == label: python_accuracy_correct += 1
        
        status = "MATCH" if is_match else "MISMATCH"
        
        # Console output - compact
        print(f"Img {real_idx:3d}: Label={label} Py={py_pred} FPGA={fpga_pred} [{status}]")
        
        # Log output - detailed
        py_correct = "✓" if py_pred == label else "✗"
        fpga_correct = "✓" if fpga_pred == label else "✗"
        output_file.write(f"Image {real_idx} | Label: {label} | Python Pred: {py_pred} {py_correct} | FPGA Pred: {fpga_pred} {fpga_correct} | Status: {status}\n")
        output_file.write(f"Python: {expected_scores.tolist()}\n")
        output_file.write(f"FPGA:   {fpga_scores.tolist()}\n")
        if not is_match:
             output_file.write(f"Diff:   {(fpga_scores - expected_scores).tolist()}\n")
             # Add magnitude comparison
             py_max = np.max(np.abs(expected_scores))
             fpga_max = np.max(np.abs(fpga_scores))
             output_file.write(f"Magnitude: Python max={py_max}, FPGA max={fpga_max}, Ratio={py_max/fpga_max if fpga_max > 0 else 'inf':.2f}\n")
        output_file.write("-" * 50 + "\n")

        # Small delay between images to let FPGA reset
        time.sleep(0.05)

    # Summary
    print("\n" + "="*50)
    print(" SUMMARY")
    print("="*50)
    print(f"Total Tests: {args.count}")
    print(f"Python Accuracy: {python_accuracy_correct}/{args.count} ({(python_accuracy_correct/args.count)*100:.1f}%)")
    print(f"FPGA Accuracy:   {fpga_accuracy_correct}/{args.count} ({(fpga_accuracy_correct/args.count)*100:.1f}%)")
    print(f"Bit-Exact Matches: {correct_matches}/{args.count} ({(correct_matches/args.count)*100:.1f}%)")
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("SUMMARY\n")
    output_file.write("="*80 + "\n")
    output_file.write(f"Total Tests: {args.count}\n")
    output_file.write(f"Python Accuracy: {python_accuracy_correct}/{args.count} ({(python_accuracy_correct/args.count)*100:.1f}%)\n")
    output_file.write(f"FPGA Accuracy:   {fpga_accuracy_correct}/{args.count} ({(fpga_accuracy_correct/args.count)*100:.1f}%)\n")
    output_file.write(f"Bit-Exact Matches: {correct_matches}/{args.count} ({(correct_matches/args.count)*100:.1f}%)\n")
    
    output_file.close()
    ser.close()

if __name__ == "__main__":
    main()
