import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import random

# --- Configuration ---
# Deterministic behavior
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
EPOCHS = 10          # More epochs for LeNet-5
LR = 0.001
INPUT_SCALE = 127.0  # Scaling factor; inputs clipped to [-128, 127] (full int8 range)

# CRITICAL FIX: Scale after average pooling (sum of 4 values // 4)
# After tanh: scale = 127.0
# After avg_pool: scale = 127.0 / 4 = 31.75
POST_POOL_SCALE = 127.0 / 4.0  # = 31.75

# Per-layer SHIFT values (calibrated)
SHIFT_CONV1 = 10  # Input scale = 127.0
SHIFT_CONV2 = 8   # Input scale = 31.75 (post-pool)
SHIFT_FC1 = 9     # Input scale = 31.75 (post-pool)
SHIFT_FC2 = 10    # Input scale = 127.0 (post-tanh)

# --- 1. Define LeNet-5 Model ---
class LeNet5(nn.Module):
    """
    LeNet-5 Architecture for MNIST:
    - Conv1: 6 filters, 5x5 kernel, padding=2 -> Output: 6x28x28
    - Pool1: 2x2 Average Pool -> Output: 6x14x14
    - Conv2: 16 filters, 5x5 kernel -> Output: 16x10x10
    - Pool2: 2x2 Average Pool -> Output: 16x5x5
    - FC1: 400 -> 120 with Tanh
    - FC2: 120 -> 84 with Tanh
    - FC3: 84 -> 10 (raw logits)
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28 -> 28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28 -> 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14 -> 10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10 -> 5

        # Classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 -> 120
        self.fc2 = nn.Linear(120, 84)          # 120 -> 84
        self.fc3 = nn.Linear(84, num_classes)  # 84 -> 10

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)

        # Flatten: 16x5x5 = 400
        x = x.view(x.size(0), -1)

        # FC Layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer (raw logits)

        return x

# --- 2. Train ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard MNIST Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # Use generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training LeNet-5 on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_acc = 100 * correct / total

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        test_acc = 100 * test_correct / test_total

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return model

# --- 3. Helpers for .MEM export (Two's Complement Hex) ---
def save_mem_weights(filename, weights_int8):
    with open(filename, "w") as f:
        for val in weights_int8.flatten():
            val = int(val)
            if val < 0: val += 256
            f.write(f"{val:02x}\n")

def save_mem_biases(filename, biases_int32):
    with open(filename, "w") as f:
        for val in biases_int32:
            val = int(val)
            if val < 0: val += (1 << 32)
            f.write(f"{val:08x}\n")

# --- 4. Generate Tanh LUT ---
def generate_tanh_lut(output_dir):
    """
    Generate 256-entry tanh lookup table for FPGA.
    Input range: -128 to 127 (int8)
    Output range: -127 to 127 (int8, representing -1.0 to 1.0)
    """
    lut = []
    for i in range(-128, 128):
        # Scale input to reasonable tanh range
        # Typical activations after conv are in range ~[-4, 4]
        x = i / 32.0
        y = np.tanh(x)
        y_int = int(np.clip(np.round(y * 127), -127, 127))
        lut.append(y_int)

    # Save as hex file for Verilog $readmemh
    with open(os.path.join(output_dir, "tanh_lut.mem"), "w") as f:
        for val in lut:
            if val < 0:
                val += 256
            f.write(f"{val:02x}\n")

    print(f"Generated tanh LUT: {len(lut)} entries")
    return np.array(lut, dtype=np.int8)

# --- 5. Export Quantized Weights ---
def export_weights(model):
    print("\n--- Starting LeNet-5 Weight Export (Fixed Scale Propagation) ---")

    # Setup Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "..", "outputs")

    dirs = {
        "bin": os.path.join(outputs_dir, "bin"),
        "mem": os.path.join(outputs_dir, "mem"),
        "npy": os.path.join(outputs_dir, "npy")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"Using SHIFT values: {SHIFT_CONV1}/{SHIFT_CONV2}/{SHIFT_FC1}/{SHIFT_FC2}")

    # Save Normalization Params
    np.save(os.path.join(dirs["npy"], "norm_mean.npy"), np.array([0.1307]))
    np.save(os.path.join(dirs["npy"], "norm_std.npy"), np.array([0.3081]))

    # Generate Tanh LUT
    generate_tanh_lut(dirs["mem"])

    # --- Extract Layers ---
    c1_w = model.conv1.weight.data.cpu().numpy()  # (6, 1, 5, 5)
    c1_b = model.conv1.bias.data.cpu().numpy()    # (6,)

    c2_w = model.conv2.weight.data.cpu().numpy()  # (16, 6, 5, 5)
    c2_b = model.conv2.bias.data.cpu().numpy()    # (16,)

    fc1_w = model.fc1.weight.data.cpu().numpy()   # (120, 400)
    fc1_b = model.fc1.bias.data.cpu().numpy()     # (120,)

    fc2_w = model.fc2.weight.data.cpu().numpy()   # (84, 120)
    fc2_b = model.fc2.bias.data.cpu().numpy()     # (84,)

    fc3_w = model.fc3.weight.data.cpu().numpy()   # (10, 84)
    fc3_b = model.fc3.bias.data.cpu().numpy()     # (10,)

    # =========================================================================
    # QUANTIZATION WITH CORRECT SCALE PROPAGATION
    # =========================================================================
    # Key insight: After avg_pool (sum/4), the scale is reduced by 4!
    #
    # Scale chain:
    # Input: 127.0
    # After Conv1+Tanh: 127.0 (tanh LUT outputs [-127,127])
    # After Pool1: 127.0/4 = 31.75 (avg divides by 4!)
    # After Conv2+Tanh: 127.0
    # After Pool2: 127.0/4 = 31.75
    # After FC1+Tanh: 127.0
    # After FC2+Tanh: 127.0
    # =========================================================================

    print("\n=== Scale Propagation Chain ===")
    print(f"Input scale: {INPUT_SCALE}")
    print(f"Post-pool scale: {POST_POOL_SCALE} (127/4)")

    # --- CONV1 ---
    # Input scale: 127.0 (from normalized image)
    print("\nQuantizing Conv1...")
    w_scale_c1 = 127.0 / np.max(np.abs(c1_w))
    c1_w_int8 = np.clip(np.round(c1_w * w_scale_c1), -128, 127).astype(np.int8)

    # Bias scale = input_scale * weight_scale
    b_scale_c1 = INPUT_SCALE * w_scale_c1
    c1_b_int32 = np.round(c1_b * b_scale_c1).astype(np.int32)

    print(f"  Input scale: {INPUT_SCALE}")
    print(f"  Weight scale: {w_scale_c1:.2f}")
    print(f"  Bias scale: {b_scale_c1:.2f}")

    # After tanh: output scale = 127.0
    # After pool1: output scale = 127.0 / 4 = 31.75

    # --- CONV2 ---
    # Input scale: 31.75 (after pool1 divides by 4!)
    print("\nQuantizing Conv2...")
    w_scale_c2 = 127.0 / np.max(np.abs(c2_w))
    c2_w_int8 = np.clip(np.round(c2_w * w_scale_c2), -128, 127).astype(np.int8)

    # CRITICAL FIX: Use POST_POOL_SCALE (31.75) not 127!
    b_scale_c2 = POST_POOL_SCALE * w_scale_c2
    c2_b_int32 = np.round(c2_b * b_scale_c2).astype(np.int32)

    print(f"  Input scale: {POST_POOL_SCALE} (after pool1)")
    print(f"  Weight scale: {w_scale_c2:.2f}")
    print(f"  Bias scale: {b_scale_c2:.2f}")

    # After tanh: output scale = 127.0
    # After pool2: output scale = 127.0 / 4 = 31.75

    # --- FC1 ---
    # Input scale: 31.75 (after pool2 divides by 4!)
    print("\nQuantizing FC1...")
    w_scale_fc1 = 127.0 / np.max(np.abs(fc1_w))
    fc1_w_int8 = np.clip(np.round(fc1_w * w_scale_fc1), -128, 127).astype(np.int8)

    # CRITICAL FIX: Use POST_POOL_SCALE (31.75) not 127!
    b_scale_fc1 = POST_POOL_SCALE * w_scale_fc1
    fc1_b_int32 = np.round(fc1_b * b_scale_fc1).astype(np.int32)

    print(f"  Input scale: {POST_POOL_SCALE} (after pool2)")
    print(f"  Weight scale: {w_scale_fc1:.2f}")
    print(f"  Bias scale: {b_scale_fc1:.2f}")

    # After tanh: output scale = 127.0

    # --- FC2 ---
    # Input scale: 127.0 (after tanh, no pooling)
    print("\nQuantizing FC2...")
    w_scale_fc2 = 127.0 / np.max(np.abs(fc2_w))
    fc2_w_int8 = np.clip(np.round(fc2_w * w_scale_fc2), -128, 127).astype(np.int8)

    # Input is directly from tanh, scale = 127.0
    b_scale_fc2 = 127.0 * w_scale_fc2
    fc2_b_int32 = np.round(fc2_b * b_scale_fc2).astype(np.int32)

    print(f"  Input scale: 127.0 (after tanh)")
    print(f"  Weight scale: {w_scale_fc2:.2f}")
    print(f"  Bias scale: {b_scale_fc2:.2f}")

    # After tanh: output scale = 127.0

    # --- FC3 ---
    # Input scale: 127.0 (after tanh, no pooling)
    print("\nQuantizing FC3 (output layer)...")
    w_scale_fc3 = 127.0 / np.max(np.abs(fc3_w))
    fc3_w_int8 = np.clip(np.round(fc3_w * w_scale_fc3), -128, 127).astype(np.int8)

    b_scale_fc3 = 127.0 * w_scale_fc3
    fc3_b_int32 = np.round(fc3_b * b_scale_fc3).astype(np.int32)

    print(f"  Input scale: 127.0 (after tanh)")
    print(f"  Weight scale: {w_scale_fc3:.2f}")
    print(f"  Bias scale: {b_scale_fc3:.2f}")

    # --- SAVE FILES ---
    layers = [
        ("conv1", c1_w_int8, c1_b_int32),
        ("conv2", c2_w_int8, c2_b_int32),
        ("fc1", fc1_w_int8, fc1_b_int32),
        ("fc2", fc2_w_int8, fc2_b_int32),
        ("fc3", fc3_w_int8, fc3_b_int32)
    ]

    total_weights = 0
    total_biases = 0

    for name, w, b in layers:
        # Binaries
        w.tofile(os.path.join(dirs["bin"], f"{name}_weights.bin"))
        b.tofile(os.path.join(dirs["bin"], f"{name}_biases.bin"))

        # Memory Files (Hex)
        save_mem_weights(os.path.join(dirs["mem"], f"{name}_weights.mem"), w)
        save_mem_biases(os.path.join(dirs["mem"], f"{name}_biases.mem"), b)

        print(f"Saved {name}: Weights {w.shape} ({w.size} bytes), Biases {b.shape} ({b.size * 4} bytes)")
        total_weights += w.size
        total_biases += b.size * 4

    print(f"\nTotal: {total_weights} weight bytes + {total_biases} bias bytes = {total_weights + total_biases} bytes")
    print(f"Outputs saved to: {outputs_dir}")

if __name__ == "__main__":
    model = train()
    export_weights(model)
