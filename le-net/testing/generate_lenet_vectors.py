import numpy as np
import os
import torch
from torchvision import datasets, transforms

"""
================================================================================
LeNet-5 Test Vector Generator
================================================================================
Generates memory files for Verilog simulation:
- Simulation weight files (hex format)
- Test image pixels
- Expected scores and predictions
================================================================================
"""

# ==========================================
# CONFIGURATION
# ==========================================
SHIFT_CONV = 8   # Right shift amount after Conv (adjusted from 8 to reduce quantization loss)
SHIFT_FC = 8     # Right shift amount after FC (adjusted from 8 to reduce quantization loss)
NUM_TESTS = 100   # Number of test images
INPUT_SCALE = 127.0

# LeNet-5 dimensions
L1_FILTERS = 6
L2_FILTERS = 16
FC1_NEURONS = 120
FC2_NEURONS = 84
FC3_NEURONS = 10

# Get path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")
MEM_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "mem")
OUTPUT_DIR = MEM_DIR  # Output to same mem directory

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# TANH LUT
# ==========================================
def load_tanh_lut():
    """Load tanh LUT from memory file."""
    lut_path = os.path.join(MEM_DIR, "tanh_lut.mem")
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
# DATA LOADING & PREPROCESSING
# ==========================================
def get_mnist_data(num_images=100):
    """Load first N images from MNIST test set."""
    data_root = os.path.join(SCRIPT_DIR, "..", "..", "data")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    images = []
    labels = []
    for i in range(num_images):
        img, label = dataset[i]
        images.append(img.numpy())
        labels.append(label)
    return images, labels

def preprocess_image(image_data, norm_mean, norm_std):
    """Matches FPGA Preprocessing: (x - mean)/std * 127"""
    x = (image_data - norm_mean) / norm_std
    x_quantized = np.round(x * INPUT_SCALE)
    x_int8 = np.clip(x_quantized, -128, 127).astype(np.int8)
    return x_int8

# ==========================================
# HARDWARE SIMULATION FUNCTIONS
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
# MAIN EXECUTION
# ==========================================
def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 60)
    print("LeNet-5 Test Vector Generator")
    print("=" * 60)
    print(f"Reading binaries from: {BIN_DIR}")
    print(f"Reading norm params from: {NPY_DIR}")
    print(f"Writing mem files to: {OUTPUT_DIR}")

    try:
        # Load Weights
        c1_w = np.fromfile(f"{BIN_DIR}/conv1_weights.bin", dtype=np.int8).reshape(L1_FILTERS, 1, 5, 5)
        c1_b = np.fromfile(f"{BIN_DIR}/conv1_biases.bin", dtype=np.int32)
        c2_w = np.fromfile(f"{BIN_DIR}/conv2_weights.bin", dtype=np.int8).reshape(L2_FILTERS, L1_FILTERS, 5, 5)
        c2_b = np.fromfile(f"{BIN_DIR}/conv2_biases.bin", dtype=np.int32)
        fc1_w = np.fromfile(f"{BIN_DIR}/fc1_weights.bin", dtype=np.int8).reshape(FC1_NEURONS, 16 * 5 * 5)
        fc1_b = np.fromfile(f"{BIN_DIR}/fc1_biases.bin", dtype=np.int32)
        fc2_w = np.fromfile(f"{BIN_DIR}/fc2_weights.bin", dtype=np.int8).reshape(FC2_NEURONS, FC1_NEURONS)
        fc2_b = np.fromfile(f"{BIN_DIR}/fc2_biases.bin", dtype=np.int32)
        fc3_w = np.fromfile(f"{BIN_DIR}/fc3_weights.bin", dtype=np.int8).reshape(FC3_NEURONS, FC2_NEURONS)
        fc3_b = np.fromfile(f"{BIN_DIR}/fc3_biases.bin", dtype=np.int32)

        # Load Norm Params
        norm_mean = np.load(f"{NPY_DIR}/norm_mean.npy")
        norm_std = np.load(f"{NPY_DIR}/norm_std.npy")

        # Load Tanh LUT
        tanh_lut = load_tanh_lut()

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Generate Simulation Weight Files ---
    print("\nGenerating simulation weight files...")

    # Conv weights (Conv1 + Conv2)
    with open(f"{OUTPUT_DIR}/sim_conv_weights.mem", "w") as f:
        for val in c1_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in c2_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")

    # Conv biases (Conv1 + Conv2)
    with open(f"{OUTPUT_DIR}/sim_conv_biases.mem", "w") as f:
        for val in c1_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in c2_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")

    # FC weights (FC1 + FC2 + FC3)
    with open(f"{OUTPUT_DIR}/sim_fc_weights.mem", "w") as f:
        for val in fc1_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in fc2_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in fc3_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")

    # FC biases (FC1 + FC2 + FC3)
    with open(f"{OUTPUT_DIR}/sim_fc_biases.mem", "w") as f:
        for val in fc1_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in fc2_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in fc3_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")

    print("  > sim_conv_weights.mem")
    print("  > sim_conv_biases.mem")
    print("  > sim_fc_weights.mem")
    print("  > sim_fc_biases.mem")

    # --- Generate Test Vectors ---
    print(f"\nLoading {NUM_TESTS} MNIST images...")
    images, labels = get_mnist_data(NUM_TESTS)

    print("Running LeNet-5 simulation and generating vectors...")

    correct = 0

    with open(f"{OUTPUT_DIR}/test_pixels.mem", "w") as f_pix, \
         open(f"{OUTPUT_DIR}/test_scores.mem", "w") as f_score, \
         open(f"{OUTPUT_DIR}/test_preds.mem", "w") as f_pred, \
         open(f"{OUTPUT_DIR}/test_labels.mem", "w") as f_lbl:

        for t in range(NUM_TESTS):
            # Preprocess Real Image
            img = preprocess_image(images[t], norm_mean, norm_std)
            label = labels[t]

            # LeNet-5 Hardware Simulation Pipeline
            # Conv1: 1x28x28 -> 6x28x28 (with padding)
            l1_out = convolve_5x5_with_padding_hw(img, c1_w, c1_b, SHIFT_CONV, tanh_lut)
            # Pool1: 6x28x28 -> 6x14x14
            l1_pool = avg_pool_hw(l1_out)

            # Conv2: 6x14x14 -> 16x10x10 (no padding)
            l2_out = convolve_5x5_no_padding_hw(l1_pool, c2_w, c2_b, SHIFT_CONV, tanh_lut)
            # Pool2: 16x10x10 -> 16x5x5
            l2_pool = avg_pool_hw(l2_out)

            # Flatten: 16x5x5 = 400
            flat_out = l2_pool.flatten()

            # FC1: 400 -> 120 with tanh
            fc1_out = fc_with_tanh_hw(flat_out, fc1_w, fc1_b, SHIFT_FC, tanh_lut)

            # FC2: 120 -> 84 with tanh
            fc2_out = fc_with_tanh_hw(fc1_out, fc2_w, fc2_b, SHIFT_FC, tanh_lut)

            # FC3: 84 -> 10 (no activation)
            scores = fc_no_activation_hw(fc2_out, fc3_w, fc3_b)
            pred = np.argmax(scores)

            if pred == label:
                correct += 1

            # Progress tracker
            if (t + 1) % 10 == 0:
                print(f"  Processing Image {t+1}/{NUM_TESTS} (Label: {label}, Pred: {pred})")

            # Write Files
            for p in img.flatten(): f_pix.write(f"{int(p) & 0xFF:02x}\n")
            for s in scores:        f_score.write(f"{int(s) & 0xFFFFFFFF:08x}\n")

            f_pred.write(f"{int(pred):01x}\n")
            f_lbl.write(f"{int(label):01x}\n")

    print(f"\nSimulation Accuracy: {correct}/{NUM_TESTS} ({100*correct/NUM_TESTS:.1f}%)")
    print(f"\nGenerated test vector files:")
    print(f"  > test_pixels.mem ({NUM_TESTS * 784} entries)")
    print(f"  > test_scores.mem ({NUM_TESTS * 10} entries)")
    print(f"  > test_preds.mem ({NUM_TESTS} entries)")
    print(f"  > test_labels.mem ({NUM_TESTS} entries)")
    print(f"\nSuccess! All files written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
