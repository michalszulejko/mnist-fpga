# LeNet-5 Implementation Changes

This document details all changes made to convert the original 2-layer CNN implementation to the classic LeNet-5 architecture.

## Table of Contents

1. [Architecture Comparison](#architecture-comparison)
2. [Python Training Changes](#python-training-changes)
3. [FPGA Inference Module Changes](#fpga-inference-module-changes)
4. [Memory Module Changes](#memory-module-changes)
5. [Weight Loader Changes](#weight-loader-changes)
6. [Testing Script Changes](#testing-script-changes)
7. [File Summary](#file-summary)

---

## Architecture Comparison

### Original CNN vs LeNet-5

| Aspect | Original CNN | LeNet-5 |
|--------|--------------|---------|
| Conv1 | 16 filters, 3x3, no padding | 6 filters, 5x5, padding=2 |
| Conv1 Output | 16x26x26 | 6x28x28 |
| Pool1 | 2x2 Max Pool | 2x2 Average Pool |
| Pool1 Output | 16x13x13 | 6x14x14 |
| Conv2 | 32 filters, 3x3 | 16 filters, 5x5 |
| Conv2 Output | 32x11x11 | 16x10x10 |
| Pool2 | 2x2 Max Pool | 2x2 Average Pool |
| Pool2 Output | 32x5x5 = 800 | 16x5x5 = 400 |
| FC Layers | 1 (800 -> 10) | 3 (400 -> 120 -> 84 -> 10) |
| Activation | ReLU | Tanh |
| Total Parameters | ~12,868 | ~61,706 |

### LeNet-5 Layer Dimensions

```
INPUT: 1x28x28 (784 bytes)
    |
CONV1: 6 filters, 5x5, padding=2
    -> Output: 6x28x28 (4,704 values)
    -> Tanh activation
    |
POOL1: 2x2 Average Pool
    -> Output: 6x14x14 (1,176 values)
    |
CONV2: 16 filters, 5x5, no padding
    -> Output: 16x10x10 (1,600 values)
    -> Tanh activation
    |
POOL2: 2x2 Average Pool
    -> Output: 16x5x5 (400 values)
    |
FLATTEN: 400 features
    |
FC1: 400 -> 120
    -> Tanh activation
    |
FC2: 120 -> 84
    -> Tanh activation
    |
FC3: 84 -> 10
    -> Raw logits (no activation)
    |
OUTPUT: argmax(scores) = predicted digit
```

---

## Python Training Changes

### File: `training/train_lenet.py`

**Replaced:** `training/train_cnn.py`

#### Training Hyperparameters

```python
BATCH_SIZE = 64
EPOCHS = 10          # More epochs for LeNet-5
LR = 0.001           # Adam optimizer
```

#### Quantization Parameters

```python
INPUT_SCALE = 127.0  # Scaling factor; inputs clipped to [-128, 127] (full int8 range)

# Per-layer SHIFT values (actual values used)
SHIFT_CONV1 = 9  # Conv1: profiled optimum is SHIFT=10
SHIFT_CONV2 = 9  # Conv2: profiled optimum is SHIFT=12
SHIFT_FC1 = 9    # FC1: profiled optimum is SHIFT=12
SHIFT_FC2 = 9    # FC2: profiled optimum is SHIFT=12

# Legacy constants (not used)
SHIFT_CONV = 8
SHIFT_FC = 8
```

**Note:** All layers currently use SHIFT=9 for consistency, though profiling suggests higher values for some layers. Input data is quantized by multiplying by `INPUT_SCALE` and clipping to the full int8 range [-128, 127].

#### Model Architecture Changes

```python
# OLD: ImprovedCNN
class ImprovedCNN(nn.Module):
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(32 * 5 * 5, 10)

# NEW: LeNet5
class LeNet5(nn.Module):
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28 -> 28
    self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28 -> 14
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14 -> 10
    self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10 -> 5
    self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 -> 120
    self.fc2 = nn.Linear(120, 84)          # 120 -> 84
    self.fc3 = nn.Linear(84, 10)           # 84 -> 10
```

#### Activation Change

```python
# OLD: ReLU
x = self.relu(self.conv1(x))

# NEW: Tanh
x = torch.tanh(self.conv1(x))
```

#### Pooling Change

```python
# OLD: MaxPool
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# NEW: AvgPool
self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
```

#### Weight Export Changes

- Added export for 5 layers instead of 3
- Added tanh LUT generation (`tanh_lut.mem`)
- Updated bias scale propagation for tanh normalization

#### Tanh LUT Generation

The training script generates a 256-entry lookup table for tanh activation:

```python
def generate_tanh_lut(output_dir):
    lut = []
    for i in range(-128, 128):
        x = i / 32.0  # Scale to approximately [-4, 4] range
        y = np.tanh(x)
        y_int = int(np.clip(np.round(y * 127), -127, 127))
        lut.append(y_int)
```

**Key points:**
- Input range: -128 to 127 (int8) → scaled to approximately [-4, 4] using `x = i / 32.0`
- Output range: **[-127, 127]** (intentionally avoiding -128 for symmetry around 0)
- This differs from input quantization which uses full int8 range [-128, 127]
- Saved as hex file for Verilog `$readmemh`

#### New Output Files

| File | Size | Description |
|------|------|-------------|
| conv1_weights.bin | 150 bytes | 6x1x5x5 int8 |
| conv1_biases.bin | 24 bytes | 6 int32 |
| conv2_weights.bin | 2,400 bytes | 16x6x5x5 int8 |
| conv2_biases.bin | 64 bytes | 16 int32 |
| fc1_weights.bin | 48,000 bytes | 120x400 int8 |
| fc1_biases.bin | 480 bytes | 120 int32 |
| fc2_weights.bin | 10,080 bytes | 84x120 int8 |
| fc2_biases.bin | 336 bytes | 84 int32 |
| fc3_weights.bin | 840 bytes | 10x84 int8 |
| fc3_biases.bin | 40 bytes | 10 int32 |
| tanh_lut.mem | 256 bytes | Tanh lookup table |

---

## FPGA Inference Module Changes

### File: `inference/rtl/inference.v`

#### FSM State Changes

**Old States (19 states):**
```verilog
IDLE, L1_LOAD_BIAS, L1_LOAD_BIAS_WAIT, L1_PREFETCH, L1_CONV, L1_SAVE,
L1_POOL, L2_LOAD_BIAS, L2_LOAD_BIAS_WAIT, L2_PREFETCH, L2_CONV, L2_SAVE,
L2_POOL, DENSE_LOAD_BIAS, DENSE_LOAD_BIAS_WAIT, DENSE_PREFETCH,
DENSE_MULT, DENSE_NEXT, DONE_STATE
```

**New States (35 states):**
```verilog
IDLE,
L1_LOAD_BIAS, L1_LOAD_BIAS_WAIT, L1_PREFETCH, L1_CONV, L1_TANH, L1_SAVE,
L1_POOL, L1_POOL_CALC,
L2_LOAD_BIAS, L2_LOAD_BIAS_WAIT, L2_PREFETCH, L2_CONV, L2_TANH, L2_SAVE,
L2_POOL, L2_POOL_CALC,
FC1_LOAD_BIAS, FC1_LOAD_BIAS_WAIT, FC1_PREFETCH, FC1_MULT, FC1_TANH, FC1_SAVE,
FC2_LOAD_BIAS, FC2_LOAD_BIAS_WAIT, FC2_PREFETCH, FC2_MULT, FC2_TANH, FC2_SAVE,
FC3_LOAD_BIAS, FC3_LOAD_BIAS_WAIT, FC3_PREFETCH, FC3_MULT, FC3_NEXT,
DONE_STATE
```

#### Kernel Size Change (3x3 -> 5x5)

```verilog
// OLD: 3x3 kernel
if (kr == 2 && kc == 2) begin
    state <= L1_SAVE;
end

// NEW: 5x5 kernel
if (kr == 4 && kc == 4) begin
    state <= L1_TANH;
end
```

#### Pooling Change (Max -> Average)

```verilog
// OLD: Max Pooling
if (buf_a_rd_data > max_val) max_val <= buf_a_rd_data;
buf_b_wr_data <= max_val;

// NEW: Average Pooling
pool_sum <= pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
temp_val <= pool_sum >>> 2;  // Divide by 4
buf_b_wr_data <= temp_val[7:0];
```

#### Activation Change (ReLU -> Tanh LUT)

```verilog
// OLD: ReLU
temp_val = acc >>> 8;
if (temp_val < 0) temp_val = 0;
if (temp_val > 127) temp_val = 127;

// NEW: Tanh via LUT
temp_val <= acc >>> 8;
// Then in next state:
if (temp_val < -128)
    tanh_addr <= 8'd0;
else if (temp_val > 127)
    tanh_addr <= 8'd255;
else
    tanh_addr <= temp_val[7:0] + 8'd128;
buf_a_wr_data <= tanh_data;  // LUT output
```

#### New Port Additions

```verilog
// Tanh LUT interface
output reg [7:0] tanh_addr,
input wire signed [7:0] tanh_data,

// FC Weights RAM (much larger)
output reg [15:0] fc_w_addr,  // Was [12:0]
input wire signed [7:0] fc_w_data,

// FC Biases RAM (214 entries)
output reg [7:0] fc_b_addr,   // Was [3:0]
input wire signed [31:0] fc_b_data,

// Buffer C (new buffer for FC intermediates)
output reg [8:0] buf_c_addr,
output reg [7:0] buf_c_wr_data,
output reg buf_c_wr_en,
input wire signed [7:0] buf_c_rd_data,
```

---

## Memory Module Changes

### File: `inference/rtl/ram_cnn.v`

#### Buffer Size Changes

```verilog
// OLD
(* ram_style = "distributed" *) reg [7:0] buffer_a [0:10815];  // 16*26*26
(* ram_style = "distributed" *) reg [7:0] buffer_b [0:2703];   // 16*13*13
(* ram_style = "distributed" *) reg [7:0] dense_ram [0:7999];  // 800*10

// NEW
(* ram_style = "distributed" *) reg [7:0] buffer_a [0:4703];   // 6*28*28
(* ram_style = "distributed" *) reg [7:0] buffer_b [0:1175];   // 6*14*14
(* ram_style = "distributed" *) reg [7:0] buffer_c [0:399];    // 16*5*5
```

### File: `inference/rtl/conv_ram.v`

#### Conv Weights RAM Size

```verilog
// OLD: 4752 bytes (144 + 4608)
(* ram_style = "distributed" *) reg [7:0] ram [0:8191];

// NEW: 2550 bytes (150 + 2400)
(* ram_style = "distributed" *) reg [7:0] ram [0:2549];
```

#### Conv Biases RAM Size

```verilog
// OLD: 48 biases (16 + 32)
(* ram_style = "distributed" *) reg [31:0] ram [0:47];

// NEW: 22 biases (6 + 16)
(* ram_style = "distributed" *) reg [31:0] ram [0:21];
```

### New File: `inference/rtl/fc_ram.v`

**Purpose:** Dedicated RAM modules for FC layer weights and biases.

```verilog
// FC Weights: 58,920 bytes (uses BRAM due to size)
(* ram_style = "block" *) reg [7:0] ram [0:58919];

// FC Biases: 214 entries (distributed RAM)
(* ram_style = "distributed" *) reg [31:0] ram [0:213];
```

### New File: `inference/rtl/tanh_lut.v`

**Purpose:** 256-entry lookup table for tanh activation.

```verilog
(* ram_style = "distributed" *) reg [7:0] lut [0:255];

initial begin
    $readmemh("tanh_lut.mem", lut);
end

assign data = lut[addr];  // Combinational read
```

---

## Weight Loader Changes

### File: `inference/rtl/load_weights.v`

#### Packet Structure Change

**Old Packet (12,984 bytes):**
```
Offset    Size      Data
0         144       Conv1 weights (16x1x3x3)
144       64        Conv1 biases (16 x int32)
208       4,608     Conv2 weights (32x16x3x3)
4,816     128       Conv2 biases (32 x int32)
4,944     8,000     Dense weights (10x800)
12,944    40        Dense biases (10 x int32)
```

**New Packet (62,670 bytes):**
```
Offset    Size      Data
0         150       Conv1 weights (6x1x5x5)
150       24        Conv1 biases (6 x int32)
174       2,400     Conv2 weights (16x6x5x5)
2,574     64        Conv2 biases (16 x int32)
2,638     48,000    FC1 weights (120x400)
50,638    480       FC1 biases (120 x int32)
51,118    10,080    FC2 weights (84x120)
61,198    336       FC2 biases (84 x int32)
61,534    840       FC3 weights (10x84)
62,374    40        FC3 biases (10 x int32)
62,414    256       Tanh LUT (256 entries)
```

#### New Output Ports

```verilog
// FC Weights
output reg [15:0] fc_w_addr,
output reg [7:0] fc_w_data,
output reg fc_w_en,

// FC Biases
output reg [7:0] fc_b_addr,
output reg [31:0] fc_b_data,
output reg fc_b_en,

// Tanh LUT
output reg [7:0] tanh_addr,
output reg [7:0] tanh_data,
output reg tanh_en
```

---

## Testing Script Changes

### File: `testing/testing_python_model.py`

#### Model Constants

The testing script uses the same quantization parameters as training:

```python
INPUT_SCALE = 127.0
SHIFT_CONV1 = 9
SHIFT_CONV2 = 9
SHIFT_FC1 = 9
SHIFT_FC2 = 9
```

#### Tanh LUT Loading

The script loads the tanh LUT from `tanh_lut.mem` with fallback generation:

```python
def load_tanh_lut():
    try:
        # Load from file
        lut_path = os.path.join(MEM_DIR, "tanh_lut.mem")
        # Parse hex values...
    except FileNotFoundError:
        # Fallback: generate default LUT
        for i in range(-128, 128):
            x = i / 32.0  # Same scaling as training
            y = np.tanh(x)
            lut[i + 128] = int(np.clip(np.round(y * 127), -127, 127))
```

#### Pooling Function Change

```python
# OLD: Max Pooling
def max_pool_2x2(input_vol):
    output_vol[ch, r, col] = np.max(window)

# NEW: Average Pooling
def avg_pool_2x2(input_vol):
    output_vol[ch, r, col] = np.sum(window) >> 2  # Integer divide by 4
```

#### Activation Change

```python
# OLD: ReLU
if acc < 0: acc = 0
if acc > 127: acc = 127

# NEW: Tanh LUT
def tanh_lut_apply(x, lut):
    x = np.clip(x, -128, 127)
    idx = (x + 128).astype(np.int32)
    return lut[idx].astype(np.int32)
```

#### Convolution Functions

```python
# NEW: 5x5 convolution with padding
def convolve_5x5_with_padding(input_vol, weights, biases, shift, tanh_lut):
    pad = 2
    padded = np.pad(input_vol, ((0, 0), (pad, pad), (pad, pad)), ...)
    # 5x5 kernel operations
    acc = tanh_lut_apply(np.array([acc]), tanh_lut)[0]

# NEW: 5x5 convolution without padding
def convolve_5x5_no_padding(input_vol, weights, biases, shift, tanh_lut):
    out_h, out_w = h - 4, w - 4
    # 5x5 kernel operations
```

#### FC Layer Functions

```python
# NEW: FC layer with tanh
def fc_layer_with_tanh(input_vec, weights, biases, shift, tanh_lut):
    acc = np.dot(input_vec, weights[i]) + biases[i]
    acc = acc >> shift
    output[i] = tanh_lut_apply(np.array([acc]), tanh_lut)[0]

# NEW: Final FC layer without activation
def fc_layer_no_activation(input_vec, weights, biases):
    scores[i] = np.dot(input_vec, weights[i]) + biases[i]
```

#### Weight Loading

```python
# OLD: 3 layers
weights = {
    'c1': (c1_w, c1_b),
    'c2': (c2_w, c2_b),
    'dense': (dw, db)
}

# NEW: 5 layers
weights = {
    'conv1': (c1_w, c1_b),
    'conv2': (c2_w, c2_b),
    'fc1': (fc1_w, fc1_b),
    'fc2': (fc2_w, fc2_b),
    'fc3': (fc3_w, fc3_b)
}
```

---

## Utilities

### File: `utils/send_weights.py`

**Purpose:** Send LeNet-5 weights and tanh LUT to FPGA via UART.

#### Features

- Loads 10 weight files + tanh_lut.mem (11 total files)
- Expected payload: 62,670 bytes
- Total packet: 62,674 bytes (including 4-byte markers)
- Transmission mode: 16-byte chunks with 20ms delay (slow & safe)

#### Protocol Markers

```python
START_MARKER = np.array([0xAA, 0x55], dtype=np.uint8)  # Weight start
END_MARKER   = np.array([0x55, 0xAA], dtype=np.uint8)  # Weight end
```

**Note:** Image transfers use different markers (`0xBB 0x66` / `0x66 0xBB`).

#### Protocol Design Rationale

- **2-byte markers** reduce false positives from random data that might match a single byte
- **Palindromic pairs** (`0xAA55` ↔ `0x55AA`) provide easy synchronization
- **Separate markers** for weights vs images allow the UART router to multiplex different data types

#### Weight Packet Structure (62,674 bytes total)

```
Offset    Size      Data
----------------------------------------------
0         2         START_MARKER (0xAA 0x55)
2         150       Conv1 weights (6x1x5x5)
152       24        Conv1 biases (6 x int32)
176       2,400     Conv2 weights (16x6x5x5)
2,576     64        Conv2 biases (16 x int32)
2,640     48,000    FC1 weights (120x400)
50,640    480       FC1 biases (120 x int32)
51,120    10,080    FC2 weights (84x120)
61,200    336       FC2 biases (84 x int32)
61,536    840       FC3 weights (10x84)
62,376    40        FC3 biases (10 x int32)
62,416    256       Tanh LUT (256 entries)
62,672    2         END_MARKER (0x55 0xAA)
```

---

## File Summary

### Modified Files

| File | Changes |
|------|---------|
| `training/train_lenet.py` | New file replacing train_cnn.py with LeNet-5 model |
| `inference/rtl/inference.v` | Complete rewrite for LeNet-5 FSM |
| `inference/rtl/ram_cnn.v` | Updated buffer sizes, added Buffer C |
| `inference/rtl/conv_ram.v` | Updated for 5x5 kernels and fewer filters |
| `inference/rtl/load_weights.v` | Updated for new packet structure |
| `testing/testing_python_model.py` | LeNet-5 inference simulation |
| `utils/send_weights.py` | Updated for LeNet-5 weight transmission (62,670 bytes payload vs original 12,984 bytes) |

### New Files

| File | Purpose |
|------|---------|
| `inference/rtl/fc_ram.v` | FC weights and biases RAM |
| `inference/rtl/tanh_lut.v` | Tanh activation lookup table |
| `LENET_CHANGES.md` | This documentation file |

### Deleted Files

| File | Reason |
|------|--------|
| `training/train_cnn.py` | Replaced by train_lenet.py |

---

## Resource Estimation

| Resource | Original CNN | LeNet-5 |
|----------|--------------|---------|
| Weight RAM | ~13 KB | ~62 KB |
| Working RAM | ~14 KB | ~7 KB |
| Total Memory | ~27 KB | ~69 KB |
| FSM States | 19 | 35 |
| Tanh LUT | N/A | 256 bytes |

**Note:** LeNet-5's 62KB weight storage uses BRAM for FC weights instead of distributed RAM due to Artix-7 XC7A35T LUT constraints.

---

## Document Status

**Version:** 1.0  
**Status:** Production/Stable  
**Last Verified:** January 2026  
**Compatible FPGA Modules:**
- `inference/rtl/inference.v`
- `inference/rtl/load_weights.v`
- `inference/rtl/fc_ram.v`
- `inference/rtl/tanh_lut.v`
- `inference/rtl/uart_router.v`

---

*Document created: January 2025*  
*Last updated: January 2026*  
*LeNet-5 implementation for MNIST-FPGA project*
