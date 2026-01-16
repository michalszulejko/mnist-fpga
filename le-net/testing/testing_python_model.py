import os
import numpy as np
import sys
import torch
from torchvision import datasets, transforms
from tanh_lut_loader import load_tanh_lut, apply_tanh_lut

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
# Deterministic behavior
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

# Model Constants
INPUT_SCALE = 127.0

# CRITICAL FIX: Scale after average pooling
# After avg_pool (sum of 4 / 4), scale is reduced by factor of 4
POST_POOL_SCALE = 127.0 / 4.0  # = 31.75

# Per-layer SHIFT values (calibrated)
# - Conv1, FC2: Input scale = 127 (full range)
# - Conv2, FC1: Input scale = 31.75 (after pool divides by 4)
SHIFT_CONV1 = 10  # Input scale = 127.0
SHIFT_CONV2 = 8   # Input scale = 31.75 (post-pool)
SHIFT_FC1 = 9     # Input scale = 31.75 (post-pool)
SHIFT_FC2 = 10    # Input scale = 127.0 (after tanh)
# FC3 has no shift

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# REMOVED: Old tanh_lut function that computed on-the-fly
# Now using the actual LUT from tanh_lut.mem via tanh_lut_loader module

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

def convolve_layer(input_vol, weights, biases, shift, tanh_lut, use_tanh=True):
    """
    LeNet-5 Convolution with 5x5 kernel.
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 5, 5)
    Output: (Out_Channels, H-4, W-4) for no padding
           (Out_Channels, H, W) for padding=2
    """
    in_ch, h, w = input_vol.shape
    out_ch, _, k_h, k_w = weights.shape
    
    # Determine padding based on kernel size
    if k_h == 5 and k_w == 5:
        # Check if this is conv1 (with padding) or conv2 (no padding)
        if h == 28 and w == 28:  # Conv1 with padding=2
            # Add zero padding
            input_padded = np.pad(input_vol, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
            out_h, out_w = h, w  # Output same size due to padding
        else:  # Conv2 with no padding
            input_padded = input_vol
            out_h, out_w = h - 4, w - 4
    else:
        input_padded = input_vol
        out_h, out_w = h - k_h + 1, w - k_w + 1
    
    output_vol = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = 0
                # Sum over all input channels
                for ch in range(in_ch):
                    window = input_padded[ch, r:r+k_h, c:c+k_w]
                    w_kernel = weights[f, ch]
                    acc += np.sum(window * w_kernel)
                
                # Add Bias
                acc += bias_val
                
                # FPGA Pipeline Steps:
                acc = acc >> shift            # 1. Arithmetic Right Shift
                
                # Apply Tanh activation
                if use_tanh:
                    acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)
                
                output_vol[f, r, c] = acc
                
    return output_vol

def fc_layer(input_vec, weights, biases, shift, tanh_lut, use_tanh=True):
    """
    Fully Connected Layer.
    Input: (N,) 1D vector
    Weights: (Out, In)
    Output: (Out,) 1D vector
    """
    output = np.zeros(weights.shape[0], dtype=np.int32)
    
    for i in range(weights.shape[0]):
        acc = np.int32(biases[i])
        acc += np.dot(input_vec.astype(np.int32), weights[i].astype(np.int32))
        
        # Shift
        acc = acc >> shift
        
        # Apply Tanh activation if specified
        if use_tanh:
            acc = apply_tanh_lut(np.clip(acc, -128, 127), tanh_lut)
        
        output[i] = acc
    
    return output

# ==========================================
# 3. BIT-EXACT INFERENCE ENGINE
# ==========================================
def simulate_quantized_inference(image_bytes, weights_dict, tanh_lut):
    # Unpack weights
    c1_w, c1_b = weights_dict['c1']
    c2_w, c2_b = weights_dict['c2']
    fc1_w, fc1_b = weights_dict['fc1']
    fc2_w, fc2_b = weights_dict['fc2']
    fc3_w, fc3_b = weights_dict['fc3']
    
    # 0. Load Image
    img = np.frombuffer(image_bytes, dtype=np.int8)
    # Shape: (1, 28, 28) for consistent 3D processing
    img_3d = img.reshape(1, 28, 28).astype(np.int32)
    
    # --- LAYER 1: Conv1 6x5x5 with padding=2 ---
    # Reshape weights: (6 filters, 1 channel, 5, 5)
    c1_w_reshaped = c1_w.reshape(6, 1, 5, 5).astype(np.int32)
    # Conv -> Tanh
    x = convolve_layer(img_3d, c1_w_reshaped, c1_b, SHIFT_CONV1, tanh_lut, use_tanh=True)
    # Pooling: 28x28 -> 14x14
    x = avg_pool_2x2(x)

    # --- LAYER 2: Conv2 16x5x5 ---
    # Reshape weights: (16 filters, 6 channels, 5, 5)
    c2_w_reshaped = c2_w.reshape(16, 6, 5, 5).astype(np.int32)
    # Conv -> Tanh
    x = convolve_layer(x, c2_w_reshaped, c2_b, SHIFT_CONV2, tanh_lut, use_tanh=True)
    # Pooling: 10x10 -> 5x5
    x = avg_pool_2x2(x)
    
    # --- LAYER 3: FC1 400 -> 120 ---
    # Flatten: (16, 5, 5) -> 400
    flattened = x.flatten().astype(np.int32)
    
    # Reshape FC1: (120, 400)
    fc1_w_reshaped = fc1_w.reshape(120, 400).astype(np.int32)
    x = fc_layer(flattened, fc1_w_reshaped, fc1_b, SHIFT_FC1, tanh_lut, use_tanh=True)
    
    # --- LAYER 4: FC2 120 -> 84 ---
    fc2_w_reshaped = fc2_w.reshape(84, 120).astype(np.int32)
    x = fc_layer(x, fc2_w_reshaped, fc2_b, SHIFT_FC2, tanh_lut, use_tanh=True)
    
    # --- LAYER 5: FC3 84 -> 10 ---
    fc3_w_reshaped = fc3_w.reshape(10, 84).astype(np.int32)
    scores = np.zeros(10, dtype=np.int32)
    for c in range(10):
        scores[c] = np.int32(fc3_b[c]) + np.dot(x.astype(np.int32), fc3_w_reshaped[c].astype(np.int32))
        
    return np.argmax(scores)

# ==========================================
# 4. UTILITIES (LOADERS)
# ==========================================
def load_all_weights():
    try:
        # Load Conv Layers
        c1_w = np.fromfile(os.path.join(BIN_DIR, "conv1_weights.bin"), dtype=np.int8)
        c1_b = np.fromfile(os.path.join(BIN_DIR, "conv1_biases.bin"), dtype=np.int32)
        
        c2_w = np.fromfile(os.path.join(BIN_DIR, "conv2_weights.bin"), dtype=np.int8)
        c2_b = np.fromfile(os.path.join(BIN_DIR, "conv2_biases.bin"), dtype=np.int32)
        
        # Load FC Layers
        fc1_w = np.fromfile(os.path.join(BIN_DIR, "fc1_weights.bin"), dtype=np.int8)
        fc1_b = np.fromfile(os.path.join(BIN_DIR, "fc1_biases.bin"), dtype=np.int32)
        
        fc2_w = np.fromfile(os.path.join(BIN_DIR, "fc2_weights.bin"), dtype=np.int8)
        fc2_b = np.fromfile(os.path.join(BIN_DIR, "fc2_biases.bin"), dtype=np.int32)
        
        fc3_w = np.fromfile(os.path.join(BIN_DIR, "fc3_weights.bin"), dtype=np.int8)
        fc3_b = np.fromfile(os.path.join(BIN_DIR, "fc3_biases.bin"), dtype=np.int32)
        
        mean = np.load(os.path.join(NPY_DIR, "norm_mean.npy"))
        std  = np.load(os.path.join(NPY_DIR, "norm_std.npy"))
        
        return {
            'c1': (c1_w, c1_b),
            'c2': (c2_w, c2_b),
            'fc1': (fc1_w, fc1_b),
            'fc2': (fc2_w, fc2_b),
            'fc3': (fc3_w, fc3_b)
        }, (mean, std)
        
    except FileNotFoundError as e:
        sys.exit(f"Error: Missing binary file. {e}\nDid you run the training script (train_lenet.py)?")

def preprocess(image_tensor, mean, std):
    x = image_tensor.numpy().squeeze()
    x = (x - mean) / std
    x = np.clip(np.round(x * INPUT_SCALE), -128, 127).astype(np.int8)
    return x.tobytes()

def get_data():
    import logging
    logging.getLogger("torchvision").setLevel(logging.CRITICAL)
    transform = transforms.Compose([transforms.ToTensor()])
    data_root = os.path.join(SCRIPT_DIR, "..", "..", "data")
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    finally:
        sys.stdout = old_stdout
    return dataset

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    print("="*60)
    print("QUANTIZED LENET-5 INFERENCE (Fixed Scale Propagation)")
    print("="*60)

    print(f"\nSHIFT values:")
    print(f"  SHIFT_CONV1 = {SHIFT_CONV1}")
    print(f"  SHIFT_CONV2 = {SHIFT_CONV2}")
    print(f"  SHIFT_FC1 = {SHIFT_FC1}")
    print(f"  SHIFT_FC2 = {SHIFT_FC2}")

    print("\nScale chain:")
    print(f"  Input: {INPUT_SCALE}")
    print(f"  After pool: {POST_POOL_SCALE} (127/4)")

    print("\nLoading weights...")
    weights, (norm_mean, norm_std) = load_all_weights()

    # Load Tanh LUT (same file FPGA uses)
    print("Loading Tanh LUT...")
    tanh_lut = load_tanh_lut()

    dataset = get_data()

    # IMPORTANT: Skip first 500 images (used for calibration)
    # Test on remaining 9500 images for unbiased evaluation
    CALIBRATION_SAMPLES = 500
    total_images = 1000  # Test on 1000 images (indices 500-1499)
    correct = 0

    print(f"\nRunning inference on {total_images} images (indices {CALIBRATION_SAMPLES}-{CALIBRATION_SAMPLES + total_images - 1})...")
    print(f"(Skipping first {CALIBRATION_SAMPLES} images used for calibration)")

    for i in range(total_images):
        # Offset by calibration samples to avoid data leakage
        img_tensor, label = dataset[CALIBRATION_SAMPLES + i]
        img_bytes = preprocess(img_tensor, norm_mean, norm_std)

        prediction = simulate_quantized_inference(img_bytes, weights, tanh_lut)

        if prediction == label:
            correct += 1

        if (i+1) % 100 == 0:
            current_acc = (correct / (i+1)) * 100
            print(f"Processed {i+1} images... Current accuracy: {current_acc:.2f}%")

    accuracy = (correct / total_images) * 100
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {accuracy:.2f}% ({correct}/{total_images})")
    print(f"{'='*60}")

    # Compare with expected
    if accuracy < 95:
        print("\nWARNING: Accuracy below 95%. Check quantization parameters.")
    elif accuracy < 97:
        print("\nINFO: Accuracy between 95-97%. Some room for improvement.")
    else:
        print("\nSUCCESS: Accuracy >= 97%. Quantization is working well!")

if __name__ == "__main__":
    main()
