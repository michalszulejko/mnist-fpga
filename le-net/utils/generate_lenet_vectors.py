import numpy as np
import os
import torch
from torchvision import datasets, transforms
from tanh_lut_loader import load_tanh_lut, apply_tanh_lut

# ==========================================
# CONFIGURATION
# ==========================================
# Per-layer SHIFT values - MUST MATCH FPGA and train_lenet.py
# Based on scale propagation through the network:
SHIFT_CONV1 = 10  # Input scale = 127.0
SHIFT_CONV2 = 8   # Input scale = 31.75 (post-pool, 127/4)
SHIFT_FC1 = 9     # Input scale = 31.75 (post-pool)
SHIFT_FC2 = 10    # Input scale = 127.0 (post-tanh)
# FC3 has no shift
NUM_TESTS = 2000  # Number of test images to generate (indices 500-2499)
START_INDEX = 500 # Starting index in MNIST test set
INPUT_SCALE = 127.0 # Must match training

# Get path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "mem")

# LeNet-5 Dimensions
L1_FILTERS = 6
L2_FILTERS = 16
FC1_UNITS = 120
FC2_UNITS = 84

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def get_mnist_data(num_images=10, start_index=0):
    """Load N images from MNIST test set starting at start_index."""
    # Download MNIST if not present (to a temp or shared folder)
    data_root = os.path.join(SCRIPT_DIR, "..", "..", "data")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    images = []
    labels = []
    for i in range(start_index, start_index + num_images):
        img, label = dataset[i]
        # Keep as (1, 28, 28) for convolution
        images.append(img.numpy())
        labels.append(label)
    return images, labels

def preprocess_image(image_data, norm_mean, norm_std):
    """ Matches FPGA Preprocessing: (x - mean)/std * 127 """
    # image_data is already [0, 1] float from ToTensor()
    x = (image_data - norm_mean) / norm_std
    x_quantized = np.round(x * INPUT_SCALE)
    x_int8 = np.clip(x_quantized, -128, 127).astype(np.int8)
    return x_int8

# ==========================================
# 2. HARDWARE SIMULATION FUNCTIONS
# ==========================================
# REMOVED: Old tanh_lut function that computed on-the-fly
# Now using the actual LUT from tanh_lut.mem via tanh_lut_loader module

def convolve_hw(input_vol, weights, biases, shift, tanh_lut, use_tanh=True):
    """ Bit-exact simulation of FPGA Convolution """
    out_ch, in_ch, k, _ = weights.shape
    h, w = input_vol.shape[1], input_vol.shape[2]
    
    # Determine if padding is needed (Conv1 with padding=2)
    if k == 5 and h == 28 and w == 28:  # Conv1
        input_padded = np.pad(input_vol, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
        out_h, out_w = h, w
    else:  # Conv2 or other
        input_padded = input_vol
        out_h, out_w = h - k + 1, w - k + 1
    
    output = np.zeros((out_ch, out_h, out_w), dtype=np.int32)
    
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
                
                # Apply Tanh activation if specified
                if use_tanh:
                    acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)
                
                output[f, r, c] = acc
    return output

def avg_pool_hw(input_vol):
    """ 2x2 Average Pooling """
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output = np.zeros((c, new_h, new_w), dtype=np.int32)
    
    for ch in range(c):
        for r in range(new_h):
            for c_idx in range(new_w):
                # Average of 2x2 window
                window_sum = input_vol[ch, r*2, c_idx*2]
                window_sum += input_vol[ch, r*2, c_idx*2+1]
                window_sum += input_vol[ch, r*2+1, c_idx*2]
                window_sum += input_vol[ch, r*2+1, c_idx*2+1]
                output[ch, r, c_idx] = window_sum // 4
    return output

def flatten_hw(input_vol):
    return input_vol.flatten()

def fc_hw(input_flat, weights, biases, shift, tanh_lut, use_tanh=True):
    # Weights are (Out, In)
    out_size = weights.shape[0]
    output = np.zeros(out_size, dtype=np.int32)
    for i in range(out_size):
        acc = np.int32(biases[i])
        acc += np.sum(input_flat.astype(np.int32) * weights[i, :].astype(np.int32))
        acc = acc >> shift
        
        # Apply Tanh activation if specified
        if use_tanh:
            acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)
        
        output[i] = acc
    return output

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Reading binaries from: {BIN_DIR}")
    print(f"Reading norm params from: {NPY_DIR}")
    print(f"Writing mem files to:  {OUTPUT_DIR}")

    try:
        # Load Weights
        c1_w = np.fromfile(f"{BIN_DIR}/conv1_weights.bin", dtype=np.int8).reshape(L1_FILTERS, 1, 5, 5)
        c1_b = np.fromfile(f"{BIN_DIR}/conv1_biases.bin", dtype=np.int32)
        c2_w = np.fromfile(f"{BIN_DIR}/conv2_weights.bin", dtype=np.int8).reshape(L2_FILTERS, L1_FILTERS, 5, 5)
        c2_b = np.fromfile(f"{BIN_DIR}/conv2_biases.bin", dtype=np.int32)
        fc1_w = np.fromfile(f"{BIN_DIR}/fc1_weights.bin", dtype=np.int8).reshape(FC1_UNITS, 16 * 5 * 5)
        fc1_b = np.fromfile(f"{BIN_DIR}/fc1_biases.bin", dtype=np.int32)
        fc2_w = np.fromfile(f"{BIN_DIR}/fc2_weights.bin", dtype=np.int8).reshape(FC2_UNITS, FC1_UNITS)
        fc2_b = np.fromfile(f"{BIN_DIR}/fc2_biases.bin", dtype=np.int32)
        fc3_w = np.fromfile(f"{BIN_DIR}/fc3_weights.bin", dtype=np.int8).reshape(10, FC2_UNITS)
        fc3_b = np.fromfile(f"{BIN_DIR}/fc3_biases.bin", dtype=np.int32)
        
        # Load Norm Params
        norm_mean = np.load(f"{NPY_DIR}/norm_mean.npy")
        norm_std = np.load(f"{NPY_DIR}/norm_std.npy")
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Generate Helper MEM files for Weights ---
    print("Generating simulation weight files...")
    with open(f"{OUTPUT_DIR}/sim_conv_weights.mem", "w") as f:
        for val in c1_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in c2_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
    with open(f"{OUTPUT_DIR}/sim_conv_biases.mem", "w") as f:
        for val in c1_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in c2_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
    with open(f"{OUTPUT_DIR}/sim_fc_weights.mem", "w") as f:
        for val in fc1_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in fc2_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in fc3_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
    with open(f"{OUTPUT_DIR}/sim_fc_biases.mem", "w") as f:
        for val in fc1_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in fc2_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in fc3_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")

    # --- Load Tanh LUT ---
    print("Loading Tanh LUT...")
    tanh_lut = load_tanh_lut()
    
    # --- Generate Real MNIST Test Vectors ---
    print(f"Loading {NUM_TESTS} MNIST images (indices {START_INDEX} to {START_INDEX + NUM_TESTS - 1})...")
    images, labels = get_mnist_data(NUM_TESTS, START_INDEX)
    
    print("Running simulation and generating vectors...")
    
    with open(f"{OUTPUT_DIR}/test_pixels.mem", "w") as f_pix, \
         open(f"{OUTPUT_DIR}/test_scores.mem", "w") as f_score, \
         open(f"{OUTPUT_DIR}/test_preds.mem", "w") as f_pred, \
         open(f"{OUTPUT_DIR}/test_labels.mem", "w") as f_lbl:
        
        for t in range(NUM_TESTS):
            # Preprocess Real Image
            img = preprocess_image(images[t], norm_mean, norm_std)
            label = labels[t]
            
            # Hardware Simulation Pipeline
            # Conv1 + Tanh + Pool1
            l1_out = convolve_hw(img, c1_w, c1_b, SHIFT_CONV1, tanh_lut, use_tanh=True)
            l1_pool = avg_pool_hw(l1_out)
            
            # Conv2 + Tanh + Pool2
            l2_out = convolve_hw(l1_pool, c2_w, c2_b, SHIFT_CONV2, tanh_lut, use_tanh=True)
            l2_pool = avg_pool_hw(l2_out)
            
            # Flatten
            flat_out = flatten_hw(l2_pool)
            
            # FC1 + Tanh
            fc1_out = fc_hw(flat_out, fc1_w, fc1_b, SHIFT_FC1, tanh_lut, use_tanh=True)
            
            # FC2 + Tanh
            fc2_out = fc_hw(fc1_out, fc2_w, fc2_b, SHIFT_FC2, tanh_lut, use_tanh=True)
            
            # FC3 (no activation, no shift)
            scores = np.zeros(10, dtype=np.int32)
            for i in range(10):
                scores[i] = np.int32(fc3_b[i]) + np.dot(fc2_out.astype(np.int32), fc3_w[i].astype(np.int32))
            
            pred = np.argmax(scores)
            
            # Progress tracker
            if t % 10 == 0:
                print(f"Processing Image {t}/{NUM_TESTS} (Label: {label})")

            # Write Files
            for p in img.flatten(): f_pix.write(f"{int(p) & 0xFF:02x}\n")
            for s in scores:        f_score.write(f"{int(s) & 0xFFFFFFFF:08x}\n")
            
            f_pred.write(f"{int(pred):01x}\n")
            f_lbl.write(f"{int(label):01x}\n")

    print(f"Success! Test vectors generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
