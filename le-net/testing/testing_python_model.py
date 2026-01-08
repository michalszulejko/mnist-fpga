import os
import numpy as np
import sys
import torch
from torchvision import datasets, transforms

# ==========================================
# LeNet-5 Python Inference Simulation
# ==========================================
# This simulates the quantized FPGA inference pipeline
# for LeNet-5 architecture.
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
NPY_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "npy")
MEM_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "mem")

# Model Constants
INPUT_SCALE = 127.0
SHIFT_CONV1 = 9
SHIFT_CONV2 = 9
SHIFT_FC1 = 9
SHIFT_FC2 = 9

# ==========================================
# HELPER FUNCTIONS
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
        # Generate default tanh LUT if file not found
        print("Warning: tanh_lut.mem not found, generating default...")
        for i in range(-128, 128):
            x = i / 32.0
            y = np.tanh(x)
            lut[i + 128] = int(np.clip(np.round(y * 127), -127, 127))
    return lut

def tanh_lut_apply(x, lut):
    """Apply tanh via lookup table."""
    x_clamped = np.clip(x, -128, 127)
    idx = (x_clamped + 128).astype(np.int32)
    return lut[idx].astype(np.int32)

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
                window = input_vol[ch, r*2:r*2+2, col*2:col*2+2]
                # Integer average: sum >> 2 (divide by 4)
                output_vol[ch, r, col] = np.sum(window) >> 2

    return output_vol

def convolve_5x5_with_padding(input_vol, weights, biases, shift, tanh_lut, layer_name='conv'):
    """
    5x5 Convolution with padding=2 (maintains spatial size).
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 5, 5)
    Output: (Out_Channels, H, W)
    """
    in_ch, h, w = input_vol.shape
    out_ch, _, k_h, k_w = weights.shape
    pad = 2

    padded = np.pad(input_vol, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output_vol = np.zeros((out_ch, h, w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(h):
            for c in range(w):
                acc = 0
                for ch in range(in_ch):
                    window = padded[ch, r:r+5, c:c+5]
                    w_kernel = weights[f, ch]
                    acc += np.sum(window * w_kernel)

                acc += bias_val
                acc = acc >> shift
                acc = tanh_lut_apply(np.array([acc]), tanh_lut)[0]
                output_vol[f, r, c] = acc

    return output_vol

def convolve_5x5_no_padding(input_vol, weights, biases, shift, tanh_lut, layer_name='conv'):
    """
    5x5 Convolution without padding.
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 5, 5)
    Output: (Out_Channels, H-4, W-4)
    """
    in_ch, h, w = input_vol.shape
    out_ch, _, k_h, k_w = weights.shape
    out_h, out_w = h - 4, w - 4

    output_vol = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = 0
                for ch in range(in_ch):
                    window = input_vol[ch, r:r+5, c:c+5]
                    w_kernel = weights[f, ch]
                    acc += np.sum(window * w_kernel)

                acc += bias_val
                acc = acc >> shift
                acc = tanh_lut_apply(np.array([acc]), tanh_lut)[0]
                output_vol[f, r, c] = acc

    return output_vol

def fc_layer_with_tanh(input_vec, weights, biases, shift, tanh_lut, layer_name='fc'):
    """
    Fully connected layer with tanh activation.
    Input: (in_features,)
    Weights: (out_features, in_features)
    Output: (out_features,)
    """
    out_features = weights.shape[0]
    output = np.zeros(out_features, dtype=np.int32)

    for i in range(out_features):
        acc = np.dot(input_vec.astype(np.int32), weights[i].astype(np.int32))
        acc += biases[i]
        acc = acc >> shift
        output[i] = tanh_lut_apply(np.array([acc]), tanh_lut)[0]

    return output

def fc_layer_no_activation(input_vec, weights, biases):
    """
    Final FC layer without activation.
    Input: (in_features,)
    Weights: (out_features, in_features)
    Output: (out_features,) - raw logits
    """
    out_features = weights.shape[0]
    scores = np.zeros(out_features, dtype=np.int32)

    for i in range(out_features):
        acc = np.dot(input_vec.astype(np.int32), weights[i].astype(np.int32))
        scores[i] = acc + biases[i]

    return scores

# ==========================================
# BIT-EXACT LENET-5 INFERENCE ENGINE
# ==========================================
def simulate_lenet5_inference(image_bytes, weights_dict, tanh_lut):
    """
    LeNet-5 inference simulation matching FPGA implementation.
    """
    # Unpack weights
    c1_w, c1_b = weights_dict['conv1']
    c2_w, c2_b = weights_dict['conv2']
    fc1_w, fc1_b = weights_dict['fc1']
    fc2_w, fc2_b = weights_dict['fc2']
    fc3_w, fc3_b = weights_dict['fc3']

    # Load Image: 28x28
    img = np.frombuffer(image_bytes, dtype=np.int8)
    img_3d = img.reshape(1, 28, 28).astype(np.int32)

    # --- CONV1: 6 filters, 5x5, padding=2 ---
    # Input: 1x28x28, Output: 6x28x28
    c1_w_reshaped = c1_w.reshape(6, 1, 5, 5).astype(np.int32)
    x = convolve_5x5_with_padding(img_3d, c1_w_reshaped, c1_b, SHIFT_CONV1, tanh_lut, layer_name='conv1')

    # --- POOL1: 2x2 Average Pool ---
    # Input: 6x28x28, Output: 6x14x14
    x = avg_pool_2x2(x)

    # --- CONV2: 16 filters, 5x5, no padding ---
    # Input: 6x14x14, Output: 16x10x10
    c2_w_reshaped = c2_w.reshape(16, 6, 5, 5).astype(np.int32)
    x = convolve_5x5_no_padding(x, c2_w_reshaped, c2_b, SHIFT_CONV2, tanh_lut, layer_name='conv2')

    # --- POOL2: 2x2 Average Pool ---
    # Input: 16x10x10, Output: 16x5x5
    x = avg_pool_2x2(x)

    # --- FLATTEN ---
    # 16x5x5 = 400 features
    flattened = x.flatten().astype(np.int32)

    # --- FC1: 400 -> 120 with tanh ---
    fc1_w_reshaped = fc1_w.reshape(120, 400).astype(np.int32)
    x = fc_layer_with_tanh(flattened, fc1_w_reshaped, fc1_b, SHIFT_FC1, tanh_lut, layer_name='fc1')

    # --- FC2: 120 -> 84 with tanh ---
    fc2_w_reshaped = fc2_w.reshape(84, 120).astype(np.int32)
    x = fc_layer_with_tanh(x, fc2_w_reshaped, fc2_b, SHIFT_FC2, tanh_lut, layer_name='fc2')

    # --- FC3: 84 -> 10 (raw logits) ---
    fc3_w_reshaped = fc3_w.reshape(10, 84).astype(np.int32)
    scores = fc_layer_no_activation(x, fc3_w_reshaped, fc3_b)

    return np.argmax(scores), scores

# ==========================================
# WEIGHT LOADERS
# ==========================================
def load_all_weights():
    try:
        # Conv1
        c1_w = np.fromfile(os.path.join(BIN_DIR, "conv1_weights.bin"), dtype=np.int8)
        c1_b = np.fromfile(os.path.join(BIN_DIR, "conv1_biases.bin"), dtype=np.int32)

        # Conv2
        c2_w = np.fromfile(os.path.join(BIN_DIR, "conv2_weights.bin"), dtype=np.int8)
        c2_b = np.fromfile(os.path.join(BIN_DIR, "conv2_biases.bin"), dtype=np.int32)

        # FC1
        fc1_w = np.fromfile(os.path.join(BIN_DIR, "fc1_weights.bin"), dtype=np.int8)
        fc1_b = np.fromfile(os.path.join(BIN_DIR, "fc1_biases.bin"), dtype=np.int32)

        # FC2
        fc2_w = np.fromfile(os.path.join(BIN_DIR, "fc2_weights.bin"), dtype=np.int8)
        fc2_b = np.fromfile(os.path.join(BIN_DIR, "fc2_biases.bin"), dtype=np.int32)

        # FC3
        fc3_w = np.fromfile(os.path.join(BIN_DIR, "fc3_weights.bin"), dtype=np.int8)
        fc3_b = np.fromfile(os.path.join(BIN_DIR, "fc3_biases.bin"), dtype=np.int32)

        mean = np.load(os.path.join(NPY_DIR, "norm_mean.npy"))
        std = np.load(os.path.join(NPY_DIR, "norm_std.npy"))

        return {
            'conv1': (c1_w, c1_b),
            'conv2': (c2_w, c2_b),
            'fc1': (fc1_w, fc1_b),
            'fc2': (fc2_w, fc2_b),
            'fc3': (fc3_w, fc3_b)
        }, (mean, std)

    except FileNotFoundError as e:
        sys.exit(f"Error: Missing binary file. {e}\nDid you run train_lenet.py?")

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
# MAIN LOOP
# ==========================================
def main():
    print("LeNet-5 Python Inference Simulation")
    print("=" * 40)

    weights, (norm_mean, norm_std) = load_all_weights()
    tanh_lut = load_tanh_lut()
    dataset = get_data()

    total_images = 1000
    correct = 0

    for i in range(total_images):
        img_tensor, label = dataset[i]
        img_bytes = preprocess(img_tensor, norm_mean, norm_std)
        prediction, _ = simulate_lenet5_inference(img_bytes, weights, tanh_lut)

        if prediction == label:
            correct += 1

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} images... (Accuracy: {100*correct/(i+1):.2f}%)")

    accuracy = (correct / total_images) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total_images})")

if __name__ == "__main__":
    main()
