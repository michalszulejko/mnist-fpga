"""
Debug script to simulate FPGA fixed-point inference in Python
and compare with expected behavior.
"""

import numpy as np
import os

# Load quantized weights and biases
bin_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "bin")

L1_weights = np.fromfile(os.path.join(bin_dir, "L1_weights.bin"), dtype=np.int8).reshape(16, 784)
L1_biases = np.fromfile(os.path.join(bin_dir, "L1_biases.bin"), dtype=np.int32)
L2_weights = np.fromfile(os.path.join(bin_dir, "L2_weights.bin"), dtype=np.int8).reshape(16, 16)
L2_biases = np.fromfile(os.path.join(bin_dir, "L2_biases.bin"), dtype=np.int32)
L3_weights = np.fromfile(os.path.join(bin_dir, "L3_weights.bin"), dtype=np.int8).reshape(10, 16)
L3_biases = np.fromfile(os.path.join(bin_dir, "L3_biases.bin"), dtype=np.int32)

print("Loaded quantized weights and biases:")
print(f"  L1: W={L1_weights.shape}, B={L1_biases.shape}")
print(f"  L2: W={L2_weights.shape}, B={L2_biases.shape}")
print(f"  L3: W={L3_weights.shape}, B={L3_biases.shape}")
print()

# Load a test image from MNIST
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="../../data", train=False, download=True, transform=transform)

# Test on first image (a 7)
image, label = test_dataset[0]
image_np = (image.squeeze().numpy() * 255).astype(np.uint8).flatten()

print(f"Testing on MNIST test image #0 (label: {label})")
print()

# Apply preprocessing (same as send_image.py)
norm_mean = np.load(os.path.join(bin_dir, "..", "npy", "norm_mean.npy"))
norm_std = np.load(os.path.join(bin_dir, "..", "npy", "norm_std.npy"))

x = image_np.astype(np.float32) / 255.0
x_normalized = (x - norm_mean) / norm_std
x_quantized = np.round(x_normalized * 127.0)
x_int8 = np.clip(x_quantized, -128, 127).astype(np.int8)

print(f"Preprocessed input range: [{x_int8.min()}, {x_int8.max()}]")
print(f"  Mean: {x_int8.mean():.2f}, Std: {x_int8.std():.2f}")
print()

# Simulate FPGA inference (32-bit accumulation with right-shifts)
print("=" * 70)
print("SIMULATING FPGA FIXED-POINT INFERENCE (32-bit accumulators + shifts)")
print("=" * 70)

# Hardware shift amounts (must match training script)
SHIFT1 = 7
SHIFT2 = 7
print(f"Hardware shifts: Layer 1 >> {SHIFT1}, Layer 2 >> {SHIFT2}\n")

# Layer 1: 784 -> 16 with ReLU and right-shift
print("\n--- Layer 1 ---")
L1_outputs = np.zeros(16, dtype=np.int32)
for n in range(16):
    acc = np.int32(0)
    for i in range(784):
        product = np.int32(x_int8[i]) * np.int32(L1_weights[n, i])
        acc += product
    acc += L1_biases[n]
    # Right-shift by 7 (divide by 128)
    acc = acc >> SHIFT1
    # ReLU
    L1_outputs[n] = max(0, acc)

print(f"L1 outputs (after ReLU):")
print(f"  Range: [{L1_outputs.min()}, {L1_outputs.max()}]")
print(f"  Mean: {L1_outputs.mean():.2f}")
print(f"  Nonzero: {np.count_nonzero(L1_outputs)}/16")
print(f"  Values: {L1_outputs}")

# Layer 2: 16 -> 16 with ReLU and right-shift
print("\n--- Layer 2 ---")
L2_outputs = np.zeros(16, dtype=np.int32)
for n in range(16):
    acc = np.int64(0)  # Use int64 to prevent overflow during accumulation
    for i in range(16):
        # 32bit × 8bit (L1 outputs are already shifted down)
        product = np.int64(L1_outputs[i]) * np.int64(L2_weights[n, i])
        acc += product
    acc += np.int64(L2_biases[n])
    
    # Clip to int32 range (simulating FPGA 32-bit register)
    overflow = False
    if acc > 2147483647:
        print(f"  WARNING: Neuron {n} overflow! acc={acc} > 2^31-1")
        acc = 2147483647
        overflow = True
    elif acc < -2147483648:
        print(f"  WARNING: Neuron {n} underflow! acc={acc} < -2^31")
        acc = -2147483648
        overflow = True
    
    if not overflow:
        # Right-shift by 7 (divide by 128)
        acc = np.int32(acc) >> SHIFT2
    else:
        acc = np.int32(acc)
    
    # ReLU
    L2_outputs[n] = max(0, acc)

print(f"L2 outputs (after ReLU):")
print(f"  Range: [{L2_outputs.min()}, {L2_outputs.max()}]")
print(f"  Mean: {L2_outputs.mean():.2f}")
print(f"  Nonzero: {np.count_nonzero(L2_outputs)}/16")
print(f"  Values: {L2_outputs}")

# Layer 3: 16 -> 10 (no ReLU, no shift - final outputs)
print("\n--- Layer 3 (Output) ---")
L3_outputs = np.zeros(10, dtype=np.int32)
for c in range(10):
    acc = np.int64(0)  # Use int64 to prevent overflow
    for i in range(16):
        # 32bit × 8bit (L2 outputs are already shifted down)
        product = np.int64(L2_outputs[i]) * np.int64(L3_weights[c, i])
        acc += product
    acc += np.int64(L3_biases[c])
    
    # Clip to int32 range (simulating FPGA 32-bit register)
    overflow = False
    if acc > 2147483647:
        print(f"  WARNING: Class {c} overflow! acc={acc} > 2^31-1")
        acc = 2147483647
        overflow = True
    elif acc < -2147483648:
        print(f"  WARNING: Class {c} underflow! acc={acc} < -2^31")
        acc = -2147483648
        overflow = True
    
    L3_outputs[c] = np.int32(acc)
    if overflow:
        print(f"  Class {c}: OVERFLOW - result may be incorrect")

print(f"L3 outputs (logits):")
for c in range(10):
    marker = " <-- MAX (Predicted)" if c == np.argmax(L3_outputs) else ""
    print(f"  Class {c}: {L3_outputs[c]:12d}{marker}")

predicted = np.argmax(L3_outputs)
print()
print("=" * 70)
print(f"Predicted digit: {predicted}")
print(f"True label: {label}")
print(f"Result: {'✓ CORRECT' if predicted == label else '✗ INCORRECT'}")
print("=" * 70)
print()

# Show contribution analysis
print("=" * 70)
print("CONTRIBUTION ANALYSIS (for predicted class {})".format(predicted))
print("=" * 70)
weighted_sum = sum(np.int64(L2_outputs[i]) * np.int64(L3_weights[predicted, i]) for i in range(16))
bias = L3_biases[predicted]
print(f"Weighted sum contribution: {weighted_sum}")
print(f"Bias contribution:         {bias}")
print(f"Total:                     {weighted_sum + bias}")
print(f"Weighted sum is {100.0 * weighted_sum / (weighted_sum + bias):.2f}% of total")

