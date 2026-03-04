# %%
import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

# %%
# Prepare torchvision MNIST datasets and loaders
print("Loading MNIST dataset...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),            # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transform)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

input_dim = 28 * 28
num_classes = 10

print(f"Device: {device}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# %%
# Define a neural network with 2 hidden layers
class MNISTTwoHidden(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

hidden_dim1 = 16
hidden_dim2 = 16
model = MNISTTwoHidden(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)
print("\nModel Summary:")
summary(model, input_size=(1, 28, 28))

# %%
# Train and evaluate
print("\nTraining 2-hidden-layer neural network...")
num_epochs = 10
log_interval = 100

for epoch in range(1, num_epochs + 1):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {running_loss/log_interval:.4f}")
            running_loss = 0.0

    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch} training accuracy: {train_acc:.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100.0 * correct / total
    print(f"Epoch {epoch} test accuracy: {test_acc:.2f}%")

print(f"\nFinal test accuracy: {test_acc:.2f}%")

# Save the trained model
output_model_dir = os.path.join(script_dir := os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(output_model_dir, exist_ok=True)
model_save_path = os.path.join(output_model_dir, "model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")

# %%
# Extract weights and biases from trained model
print("\nExtracting weights and biases...")
model.eval()
model_cpu = model.to("cpu")

# Extract layer parameters
# Layer structure: Flatten -> Linear(784,16) -> ReLU -> Linear(16,16) -> ReLU -> Linear(16,10)
L1_weight = model_cpu.net[1].weight.data.numpy()  # Shape: (16, 784)
L1_bias = model_cpu.net[1].bias.data.numpy()      # Shape: (16,)
L2_weight = model_cpu.net[3].weight.data.numpy()  # Shape: (16, 16)
L2_bias = model_cpu.net[3].bias.data.numpy()      # Shape: (16,)
L3_weight = model_cpu.net[5].weight.data.numpy()  # Shape: (10, 16)
L3_bias = model_cpu.net[5].bias.data.numpy()      # Shape: (10,)

print(f"L1 weights shape: {L1_weight.shape}, biases: {L1_bias.shape}")
print(f"L2 weights shape: {L2_weight.shape}, biases: {L2_bias.shape}")
print(f"L3 weights shape: {L3_weight.shape}, biases: {L3_bias.shape}")

# %%
# Quantization with dynamic scaling AND hardware right-shifts to prevent overflow
print("\nQuantizing weights and biases...")

# Input scale (inputs will be quantized to int8 range)
INPUT_SCALE = 127.0

# Hardware right-shift amounts (to prevent overflow in 32-bit accumulators)
# These shifts are applied in FPGA after each layer's accumulation
SHIFT1 = 7  # Right-shift after Layer 1 (divide by 128)
SHIFT2 = 7  # Right-shift after Layer 2 (divide by 128)

print(f"Hardware shifts: Layer 1 >> {SHIFT1}, Layer 2 >> {SHIFT2}")

# Layer 1: Quantize weights
max_abs_L1_W = np.max(np.abs(L1_weight))
L1_W_SCALE = 127.0 / max_abs_L1_W
L1_weight_int8 = np.clip(np.round(L1_weight * L1_W_SCALE), -127, 127).astype(np.int8)
# Bias scale is applied BEFORE the shift
L1_BIAS_SCALE = L1_W_SCALE * INPUT_SCALE
L1_bias_int32 = np.round(L1_bias * L1_BIAS_SCALE).astype(np.int32)

print(f"\nLayer 1:")
print(f"  Weight scale: {L1_W_SCALE:.4f}")
print(f"  Weight range: [{L1_weight.min():.4f}, {L1_weight.max():.4f}] -> [{L1_weight_int8.min()}, {L1_weight_int8.max()}]")
print(f"  Bias scale: {L1_BIAS_SCALE:.4f}")
print(f"  Bias range: [{L1_bias.min():.4f}, {L1_bias.max():.4f}] -> [{L1_bias_int32.min()}, {L1_bias_int32.max()}]")

# Output of L1 after hardware shift
# After L1: output = (input_int8 @ weight_int8.T + bias_int32) >> SHIFT1
L1_OUTPUT_SCALE = (INPUT_SCALE * L1_W_SCALE) / (2 ** SHIFT1)
print(f"  Output scale (after shift): {L1_OUTPUT_SCALE:.4f}")

# Layer 2: Input scale is L1's output scale (after shift)
max_abs_L2_W = np.max(np.abs(L2_weight))
L2_W_SCALE = 127.0 / max_abs_L2_W
L2_weight_int8 = np.clip(np.round(L2_weight * L2_W_SCALE), -127, 127).astype(np.int8)
# Bias scale accounts for L1's shifted output
L2_BIAS_SCALE = L2_W_SCALE * L1_OUTPUT_SCALE
L2_bias_int32 = np.round(L2_bias * L2_BIAS_SCALE).astype(np.int32)

print(f"\nLayer 2:")
print(f"  Weight scale: {L2_W_SCALE:.4f}")
print(f"  Weight range: [{L2_weight.min():.4f}, {L2_weight.max():.4f}] -> [{L2_weight_int8.min()}, {L2_weight_int8.max()}]")
print(f"  Bias scale: {L2_BIAS_SCALE:.4f}")
print(f"  Bias range: [{L2_bias.min():.4f}, {L2_bias.max():.4f}] -> [{L2_bias_int32.min()}, {L2_bias_int32.max()}]")

# Output of L2 after hardware shift
L2_OUTPUT_SCALE = (L1_OUTPUT_SCALE * L2_W_SCALE) / (2 ** SHIFT2)
print(f"  Output scale (after shift): {L2_OUTPUT_SCALE:.4f}")

# Layer 3: Input scale is L2's output scale (after shift)
max_abs_L3_W = np.max(np.abs(L3_weight))
L3_W_SCALE = 127.0 / max_abs_L3_W
L3_weight_int8 = np.clip(np.round(L3_weight * L3_W_SCALE), -127, 127).astype(np.int8)
# Bias scale accounts for L2's shifted output
L3_BIAS_SCALE = L3_W_SCALE * L2_OUTPUT_SCALE
L3_bias_int32 = np.round(L3_bias * L3_BIAS_SCALE).astype(np.int32)

print(f"\nLayer 3:")
print(f"  Weight scale: {L3_W_SCALE:.4f}")
print(f"  Weight range: [{L3_weight.min():.4f}, {L3_weight.max():.4f}] -> [{L3_weight_int8.min()}, {L3_weight_int8.max()}]")
print(f"  Bias scale: {L3_BIAS_SCALE:.4f}")
print(f"  Bias range: [{L3_bias.min():.4f}, {L3_bias.max():.4f}] -> [{L3_bias_int32.min()}, {L3_bias_int32.max()}]")
print(f"  Final output scale: {L3_W_SCALE * L2_OUTPUT_SCALE:.4f}")

# %%
# Save to .MEM files (hex format, two's complement)
print("\nSaving weights and biases to .mem files...")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_mem_dir = os.path.join(script_dir, "..", "outputs", "mem")
output_bin_dir = os.path.join(script_dir, "..", "outputs", "bin")
output_npy_dir = os.path.join(script_dir, "..", "outputs", "npy")
os.makedirs(output_mem_dir, exist_ok=True)
os.makedirs(output_bin_dir, exist_ok=True)
os.makedirs(output_npy_dir, exist_ok=True)

def save_mem_weights(filename, weights_int8):
    """Save INT8 weights to .mem file (2 hex digits, two's complement)"""
    with open(filename, "w") as f:
        for val in weights_int8.flatten():
            val = int(val)
            if val < 0:
                val = val + 256
            f.write(f"{val:02x}\n")

def save_mem_biases(filename, biases_int32):
    """Save INT32 biases to .mem file (8 hex digits, two's complement)"""
    with open(filename, "w") as f:
        for val in biases_int32:
            val = int(val)
            if val < 0:
                val = val + (1 << 32)
            f.write(f"{val:08x}\n")

def save_bin_weights(filename, weights_int8):
    """Save INT8 weights to .bin file"""
    # Convert to int16 first to avoid overflow when adding 256
    weights_int16 = weights_int8.astype(np.int16)
    weights_uint8 = np.where(weights_int16 < 0, weights_int16 + 256, weights_int16).astype(np.uint8)
    weights_uint8.flatten().tofile(filename)

def save_bin_biases(filename, biases_int32):
    """Save INT32 biases to .bin file (little-endian)"""
    # Convert to int64 first to avoid overflow when adding 2^32
    biases_int64 = biases_int32.astype(np.int64)
    biases_uint32 = np.where(biases_int64 < 0, biases_int64 + (1 << 32), biases_int64).astype(np.uint32)
    biases_uint32.tofile(filename)

# Save Layer 1
save_mem_weights(os.path.join(output_mem_dir, "L1_weights.mem"), L1_weight_int8)
save_mem_biases(os.path.join(output_mem_dir, "L1_biases.mem"), L1_bias_int32)
save_bin_weights(os.path.join(output_bin_dir, "L1_weights.bin"), L1_weight_int8)
save_bin_biases(os.path.join(output_bin_dir, "L1_biases.bin"), L1_bias_int32)

# Save Layer 2
save_mem_weights(os.path.join(output_mem_dir, "L2_weights.mem"), L2_weight_int8)
save_mem_biases(os.path.join(output_mem_dir, "L2_biases.mem"), L2_bias_int32)
save_bin_weights(os.path.join(output_bin_dir, "L2_weights.bin"), L2_weight_int8)
save_bin_biases(os.path.join(output_bin_dir, "L2_biases.bin"), L2_bias_int32)

# Save Layer 3
save_mem_weights(os.path.join(output_mem_dir, "L3_weights.mem"), L3_weight_int8)
save_mem_biases(os.path.join(output_mem_dir, "L3_biases.mem"), L3_bias_int32)
save_bin_weights(os.path.join(output_bin_dir, "L3_weights.bin"), L3_weight_int8)
save_bin_biases(os.path.join(output_bin_dir, "L3_biases.bin"), L3_bias_int32)

print(f"Generated: {output_mem_dir}/L1_weights.mem, L1_biases.mem")
print(f"Generated: {output_mem_dir}/L2_weights.mem, L2_biases.mem")
print(f"Generated: {output_mem_dir}/L3_weights.mem, L3_biases.mem")
print(f"Generated: {output_bin_dir}/L1_weights.bin, L1_biases.bin")
print(f"Generated: {output_bin_dir}/L2_weights.bin, L2_biases.bin")
print(f"Generated: {output_bin_dir}/L3_weights.bin, L3_biases.bin")

# %%
# Save scale information and normalization parameters
print("\nSaving scale information...")

scale_info_path = os.path.join(output_npy_dir, "scale_info.txt")
with open(scale_info_path, "w") as f:
    f.write("=== Two-Hidden-Layer Network Quantization ===\n\n")
    f.write(f"Input scale factor: {INPUT_SCALE}\n")
    f.write(f"Hardware shifts: Layer 1 >> {SHIFT1}, Layer 2 >> {SHIFT2}\n\n")
    f.write(f"Layer 1:\n")
    f.write(f"  Weight scale: {L1_W_SCALE}\n")
    f.write(f"  Bias scale: {L1_BIAS_SCALE}\n")
    f.write(f"  Output scale (after shift): {L1_OUTPUT_SCALE}\n\n")
    f.write(f"Layer 2:\n")
    f.write(f"  Weight scale: {L2_W_SCALE}\n")
    f.write(f"  Bias scale: {L2_BIAS_SCALE}\n")
    f.write(f"  Output scale (after shift): {L2_OUTPUT_SCALE}\n\n")
    f.write(f"Layer 3:\n")
    f.write(f"  Weight scale: {L3_W_SCALE}\n")
    f.write(f"  Bias scale: {L3_BIAS_SCALE}\n")
    f.write(f"  Final output scale: {L3_W_SCALE * L2_OUTPUT_SCALE}\n\n")
    f.write(f"Final test accuracy: {test_acc:.2f}%\n")

# Save normalization parameters (mean=0.1307, std=0.3081 for MNIST)
norm_mean = np.array([0.1307])
norm_std = np.array([0.3081])
np.save(os.path.join(output_npy_dir, "norm_mean.npy"), norm_mean)
np.save(os.path.join(output_npy_dir, "norm_std.npy"), norm_std)

print(f"Generated: {scale_info_path}")
print(f"Generated: {output_npy_dir}/norm_mean.npy, norm_std.npy")

print("\nDone!")
