import numpy as np
import os
import torch
from torchvision import datasets, transforms

# ==========================================
# CONFIGURATION
# ==========================================
SHIFT_CONV = 8    # Right shift amount after Conv
SHIFT_DENSE = 8   # Right shift amount after Dense
NUM_TESTS = 100   # <--- CHANGED FROM 10 TO 100
INPUT_SCALE = 127.0 # Must match training

# Get path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "mem")

# Hardware Dimensions
L1_FILTERS = 16 
L2_FILTERS = 32

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def get_mnist_data(num_images=10):
    """Load first N images from MNIST test set."""
    # Download MNIST if not present (to a temp or shared folder)
    data_root = os.path.join(SCRIPT_DIR, "..", "..", "data")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    
    images = []
    labels = []
    for i in range(num_images):
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
def convolve_hw(input_vol, weights, biases, shift):
    """ Bit-exact simulation of FPGA Convolution """
    out_ch, in_ch, k, _ = weights.shape
    h, w = input_vol.shape[1], input_vol.shape[2]
    out_h, out_w = h - 2, w - 2 
    
    output = np.zeros((out_ch, out_h, out_w), dtype=np.int32)
    
    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = np.int32(bias_val)
                for ch in range(in_ch):
                    window = input_vol[ch, r:r+3, c:c+3].astype(np.int32)
                    kernel = weights[f, ch].astype(np.int32)
                    acc += np.sum(window * kernel)
                
                acc = acc >> shift
                if acc < 0: acc = 0       # ReLU
                if acc > 127: acc = 127   # Saturation
                output[f, r, c] = acc
    return output

def max_pool_hw(input_vol):
    """ 2x2 Max Pooling """
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output = np.zeros((c, new_h, new_w), dtype=np.int32)
    
    for ch in range(c):
        for r in range(new_h):
            for c_idx in range(new_w):
                val = input_vol[ch, r*2, c_idx*2]
                val = max(val, input_vol[ch, r*2, c_idx*2+1])
                val = max(val, input_vol[ch, r*2+1, c_idx*2])
                val = max(val, input_vol[ch, r*2+1, c_idx*2+1])
                output[ch, r, c_idx] = val
    return output

def flatten_hw(input_vol):
    return input_vol.flatten()

def dense_hw(input_flat, weights, biases):
    # Weights are (10, 800)
    scores = np.zeros(10, dtype=np.int32)
    for i in range(10):
        acc = np.int32(biases[i])
        acc += np.sum(input_flat * weights[i, :])
        scores[i] = acc
    return scores

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
        c1_w = np.fromfile(f"{BIN_DIR}/conv1_weights.bin", dtype=np.int8).reshape(L1_FILTERS, 1, 3, 3)
        c1_b = np.fromfile(f"{BIN_DIR}/conv1_biases.bin", dtype=np.int32)
        c2_w = np.fromfile(f"{BIN_DIR}/conv2_weights.bin", dtype=np.int8).reshape(L2_FILTERS, L1_FILTERS, 3, 3)
        c2_b = np.fromfile(f"{BIN_DIR}/conv2_biases.bin", dtype=np.int32)
        d_w = np.fromfile(f"{BIN_DIR}/dense_weights.bin", dtype=np.int8).reshape(10, 32 * 5 * 5)
        d_b = np.fromfile(f"{BIN_DIR}/dense_biases.bin", dtype=np.int32)
        
        # Load Norm Params
        norm_mean = np.load(f"{NPY_DIR}/norm_mean.npy")
        norm_std = np.load(f"{NPY_DIR}/norm_std.npy")
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Generate Helper MEM files for Weights (Unchanged) ---
    print("Generating simulation weight files...")
    with open(f"{OUTPUT_DIR}/sim_conv_weights.mem", "w") as f:
        for val in c1_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
        for val in c2_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
    with open(f"{OUTPUT_DIR}/sim_conv_biases.mem", "w") as f:
        for val in c1_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
        for val in c2_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
    with open(f"{OUTPUT_DIR}/sim_dense_weights.mem", "w") as f:
        for val in d_w.flatten(): f.write(f"{int(val) & 0xFF:02x}\n")
    with open(f"{OUTPUT_DIR}/sim_dense_biases.mem", "w") as f:
        for val in d_b: f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")

    # --- Generate Real MNIST Test Vectors ---
    print(f"Loading {NUM_TESTS} MNIST images...")
    images, labels = get_mnist_data(NUM_TESTS)
    
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
            l1_out = convolve_hw(img, c1_w, c1_b, SHIFT_CONV)
            l1_pool = max_pool_hw(l1_out)
            
            l2_out = convolve_hw(l1_pool, c2_w, c2_b, SHIFT_CONV)
            l2_pool = max_pool_hw(l2_out)
            
            flat_out = flatten_hw(l2_pool)
            scores = dense_hw(flat_out, d_w, d_b)
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