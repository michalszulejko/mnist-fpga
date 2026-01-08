import serial
import time
import os
import struct
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"
NPY_DIR = "../outputs/npy"
MEM_DIR = "../outputs/mem"
OUTPUT_FILE = "../outputs/txt/lenet_comparison.txt"

# Protocol Markers
IMG_START = bytes([0xBB, 0x66])
IMG_END   = bytes([0x66, 0xBB])
CMD_READ_DIGIT = bytes([0xCC])
CMD_READ_SCORES = bytes([0xCD])

# Shift parameters (Must match Training & FPGA)
SHIFT_CONV = 8  # Adjusted from 8 to reduce quantization loss
SHIFT_FC = 8    # Adjusted from 8 to reduce quantization loss
INPUT_SCALE = 127.0

# LeNet-5 dimensions
L1_FILTERS = 6
L2_FILTERS = 16
FC1_NEURONS = 120
FC2_NEURONS = 84
FC3_NEURONS = 10

# ==========================================
# 1. TANH LUT
# ==========================================
def load_tanh_lut(mem_path):
    """Load tanh LUT from memory file."""
    lut_path = os.path.join(mem_path, "tanh_lut.mem")
    lut = np.zeros(256, dtype=np.int8)
    try:
        with open(lut_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 256:
                    break
                val = int(line.strip(), 16)
                if val > 127:
                    val -= 256
                lut[i] = val
    except FileNotFoundError:
        print("Warning: tanh_lut.mem not found, generating default...")
        for i in range(-128, 128):
            x = i / 32.0
            y = np.tanh(x)
            lut[i + 128] = int(np.clip(np.round(y * 127), -127, 127))
    return lut

def tanh_lut_apply(x, lut):
    """Apply tanh via lookup table."""
    x = np.clip(x, -128, 127)
    idx = (x + 128).astype(np.int32)
    return lut[idx].astype(np.int32)

# ==========================================
# 2. HARDWARE SIMULATION FUNCTIONS
# ==========================================
def convolve_5x5_with_padding_hw(input_vol, weights, biases, shift, tanh_lut):
    """5x5 Convolution with padding=2 (maintains spatial size)."""
    out_ch, in_ch, k, _ = weights.shape
    h, w = input_vol.shape[1], input_vol.shape[2]
    pad = 2

    # Pad input
    padded = np.pad(input_vol, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    output = np.zeros((out_ch, h, w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(h):
            for c in range(w):
                acc = np.int32(bias_val)
                for ch in range(in_ch):
                    window = padded[ch, r:r+5, c:c+5].astype(np.int32)
                    kernel = weights[f, ch].astype(np.int32)
                    acc += np.sum(window * kernel)

                acc = acc >> shift
                acc = tanh_lut_apply(np.array([acc]), tanh_lut)[0]
                output[f, r, c] = acc
    return output

def convolve_5x5_no_padding_hw(input_vol, weights, biases, shift, tanh_lut):
    """5x5 Convolution without padding."""
    out_ch, in_ch, k, _ = weights.shape
    h, w = input_vol.shape[1], input_vol.shape[2]
    out_h, out_w = h - 4, w - 4

    output = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = np.int32(bias_val)
                for ch in range(in_ch):
                    window = input_vol[ch, r:r+5, c:c+5].astype(np.int32)
                    kernel = weights[f, ch].astype(np.int32)
                    acc += np.sum(window * kernel)

                acc = acc >> shift
                acc = tanh_lut_apply(np.array([acc]), tanh_lut)[0]
                output[f, r, c] = acc
    return output

def avg_pool_hw(input_vol):
    """2x2 Average Pooling."""
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output = np.zeros((c, new_h, new_w), dtype=np.int32)

    for ch in range(c):
        for r in range(new_h):
            for c_idx in range(new_w):
                val = input_vol[ch, r*2, c_idx*2]
                val += input_vol[ch, r*2, c_idx*2+1]
                val += input_vol[ch, r*2+1, c_idx*2]
                val += input_vol[ch, r*2+1, c_idx*2+1]
                output[ch, r, c_idx] = val >> 2  # Divide by 4
    return output

def fc_with_tanh_hw(input_flat, weights, biases, shift, tanh_lut):
    """Fully connected layer with tanh activation."""
    out_features = weights.shape[0]
    output = np.zeros(out_features, dtype=np.int32)

    for i in range(out_features):
        acc = np.int32(biases[i])
        acc += np.sum(input_flat.astype(np.int32) * weights[i, :].astype(np.int32))
        acc = acc >> shift
        output[i] = tanh_lut_apply(np.array([acc]), tanh_lut)[0]
    return output

def fc_no_activation_hw(input_flat, weights, biases):
    """Final FC layer without activation."""
    out_features = weights.shape[0]
    scores = np.zeros(out_features, dtype=np.int32)

    for i in range(out_features):
        acc = np.int32(biases[i])
        acc += np.sum(input_flat.astype(np.int32) * weights[i, :].astype(np.int32))
        scores[i] = acc
    return scores

# ==========================================
# 3. BIT-EXACT SIMULATION (LeNet-5)
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

    # 2. Conv1: 1x28x28 -> 6x28x28 (with padding)
    x = convolve_5x5_with_padding_hw(img_3d, c1_w, c1_b, SHIFT_CONV, tanh_lut)
    # Pool1: 6x28x28 -> 6x14x14
    x = avg_pool_hw(x)

    # 3. Conv2: 6x14x14 -> 16x10x10 (no padding)
    x = convolve_5x5_no_padding_hw(x, c2_w, c2_b, SHIFT_CONV, tanh_lut)
    # Pool2: 16x10x10 -> 16x5x5
    x = avg_pool_hw(x)

    # 4. Flatten: 16x5x5 = 400
    flat_out = x.flatten()

    # 5. FC1: 400 -> 120 with tanh
    fc1_out = fc_with_tanh_hw(flat_out, fc1_w, fc1_b, SHIFT_FC, tanh_lut)

    # 6. FC2: 120 -> 84 with tanh
    fc2_out = fc_with_tanh_hw(fc1_out, fc2_w, fc2_b, SHIFT_FC, tanh_lut)

    # 7. FC3: 84 -> 10 (no activation)
    scores = fc_no_activation_hw(fc2_out, fc3_w, fc3_b)

    return scores

# ==========================================
# 4. UTILITIES
# ==========================================
def load_files(bin_path, npy_path, mem_path):
    """Load weights and normalization parameters."""
    print("Loading weights from binary files...")

    # Load Weights
    try:
        c1_w = np.fromfile(os.path.join(bin_path, "conv1_weights.bin"), dtype=np.int8).reshape(L1_FILTERS, 1, 5, 5)
        c1_b = np.fromfile(os.path.join(bin_path, "conv1_biases.bin"), dtype=np.int32)
        c2_w = np.fromfile(os.path.join(bin_path, "conv2_weights.bin"), dtype=np.int8).reshape(L2_FILTERS, L1_FILTERS, 5, 5)
        c2_b = np.fromfile(os.path.join(bin_path, "conv2_biases.bin"), dtype=np.int32)
        fc1_w = np.fromfile(os.path.join(bin_path, "fc1_weights.bin"), dtype=np.int8).reshape(FC1_NEURONS, 16 * 5 * 5)
        fc1_b = np.fromfile(os.path.join(bin_path, "fc1_biases.bin"), dtype=np.int32)
        fc2_w = np.fromfile(os.path.join(bin_path, "fc2_weights.bin"), dtype=np.int8).reshape(FC2_NEURONS, FC1_NEURONS)
        fc2_b = np.fromfile(os.path.join(bin_path, "fc2_biases.bin"), dtype=np.int32)
        fc3_w = np.fromfile(os.path.join(bin_path, "fc3_weights.bin"), dtype=np.int8).reshape(FC3_NEURONS, FC2_NEURONS)
        fc3_b = np.fromfile(os.path.join(bin_path, "fc3_biases.bin"), dtype=np.int32)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Weight file not found! {e}")
        print("Did you run `train_lenet.py` to generate the weights?")
        exit(1)

    # Load Norm Params
    mean = np.load(os.path.join(npy_path, "norm_mean.npy"))
    std = np.load(os.path.join(npy_path, "norm_std.npy"))

    # Load Tanh LUT
    tanh_lut = load_tanh_lut(mem_path)

    weights = {
        'c1': (c1_w, c1_b),
        'c2': (c2_w, c2_b),
        'fc1': (fc1_w, fc1_b),
        'fc2': (fc2_w, fc2_b),
        'fc3': (fc3_w, fc3_b)
    }
    return weights, (mean, std), tanh_lut

def get_mnist_data(num_images=10):
    """Load first N images from MNIST test set."""
    transform = transforms.Compose([transforms.ToTensor()])
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
    x = np.clip(np.round(x * INPUT_SCALE), -128, 127).astype(np.int8)
    return x

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("--count", type=int, default=10, help="Number of images to test")
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    npy_path = os.path.join(script_dir, NPY_DIR)
    mem_path = os.path.join(script_dir, MEM_DIR)
    log_path = os.path.join(script_dir, OUTPUT_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    output_file = open(log_path, "w", encoding='utf-8')

    print("="*60)
    print("LeNet-5 FPGA vs Python Comparison")
    print("="*60)

    weights, (mean, std), tanh_lut = load_files(bin_path, npy_path, mem_path)
    print(f"Loading {args.count} MNIST test images...")
    images, labels = get_mnist_data(args.count)

    # Open UART
    try:
        print(f"Opening {args.port} at {args.baud}...")
        ser = serial.Serial(args.port, args.baud, timeout=2)
        time.sleep(2) # Wait for DTR/RTS
    except Exception as e:
        print(f"Error opening {args.port}: {e}")
        output_file.close()
        return

    print(f"\nRunning comparison on {args.count} images...")
    print("="*60)

    output_file.write(f"LeNet-5 Python vs FPGA Comparison\n")
    output_file.write(f"Test Count: {args.count}\n")
    output_file.write("=" * 80 + "\n\n")

    correct_matches = 0
    accuracy_correct = 0

    for idx in range(args.count):
        img_raw = images[idx]
        label = labels[idx]
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
        time.sleep(0.2) # Wait for inference (LeNet is slower than simple CNN)

        ser.write(CMD_READ_SCORES)
        response = ser.read(40)

        if len(response) != 40:
            print(f"Image {idx}: Timeout/Error receiving scores (got {len(response)} bytes).")
            output_file.write(f"Image {idx} | TIMEOUT\n")
            continue

        fpga_scores = np.array(struct.unpack('<10i', response), dtype=np.int32)
        fpga_pred = np.argmax(fpga_scores)

        # --- C. COMPARE ---
        is_match = np.array_equal(expected_scores, fpga_scores)
        if is_match: correct_matches += 1

        if fpga_pred == label: accuracy_correct += 1

        status = "✓ MATCH" if is_match else "✗ MISMATCH"

        # Console output
        print(f"Image {idx:2d} | Label: {label} | Python: {py_pred} | FPGA: {fpga_pred} | {status}")

        # Log output - detailed
        output_file.write(f"Image {idx} | Label: {label} | Status: {status}\n")
        output_file.write(f"Python Pred: {py_pred}\n")
        output_file.write(f"FPGA Pred:   {fpga_pred}\n")
        output_file.write(f"Python Scores: {expected_scores.tolist()}\n")
        output_file.write(f"FPGA Scores:   {fpga_scores.tolist()}\n")
        if not is_match:
             output_file.write(f"Diff:          {(fpga_scores - expected_scores).tolist()}\n")
        output_file.write("-" * 80 + "\n\n")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Tests:          {args.count}")
    print(f"FPGA Accuracy:        {accuracy_correct}/{args.count} ({(accuracy_correct/args.count)*100:.1f}%)")
    print(f"Bit-Exact Matches:    {correct_matches}/{args.count} ({(correct_matches/args.count)*100:.1f}%)")
    print(f"Results saved to:     {log_path}")

    output_file.write("\n" + "="*80 + "\n")
    output_file.write("SUMMARY\n")
    output_file.write("="*80 + "\n")
    output_file.write(f"Total Tests: {args.count}\n")
    output_file.write(f"FPGA Accuracy: {accuracy_correct}/{args.count} ({(accuracy_correct/args.count)*100:.1f}%)\n")
    output_file.write(f"Bit-Exact Matches: {correct_matches}/{args.count} ({(correct_matches/args.count)*100:.1f}%)\n")

    output_file.close()
    ser.close()

if __name__ == "__main__":
    main()
