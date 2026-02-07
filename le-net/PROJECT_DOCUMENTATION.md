# MNIST LeNet-5 on FPGA - Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Directory Structure](#directory-structure)
4. [LeNet-5 Model Architecture](#lenet-5-model-architecture)
5. [Python Training Pipeline](#python-training-pipeline)
6. [Quantization Strategy](#quantization-strategy)
7. [FPGA Implementation](#fpga-implementation)
8. [Communication Protocol](#communication-protocol)
9. [Testing Infrastructure](#testing-infrastructure)
10. [Memory Organization](#memory-organization)
11. [Tanh Activation Implementation](#tanh-activation-implementation)
12. [Performance Metrics](#performance-metrics)

---

## Project Overview

This project implements the classic **LeNet-5** convolutional neural network architecture for MNIST digit classification on a Basys3 FPGA (Artix-7 XC7A35T). The system achieves **97.9% accuracy** on MNIST test images with bit-exact results between Python simulation and FPGA hardware.

### Key Features

- **Classic LeNet-5 architecture** with 5x5 convolutions and three fully-connected layers
- **8-bit quantized inference** for efficient FPGA implementation
- **Tanh activation** via 256-entry lookup table
- **Average pooling** instead of max pooling (true to original LeNet-5)
- **UART communication** at 115200 baud for weight loading and image inference
- **Real-time display** of predictions on 7-segment display
- **Comprehensive testing** infrastructure comparing Python and FPGA outputs

### Target Hardware

| Component | Specification |
|-----------|---------------|
| FPGA Board | Digilent Basys3 |
| FPGA Chip | Artix-7 XC7A35T-1CPG236C |
| System Clock | 100 MHz |
| UART Baud Rate | 115200 |
| Display | 4-digit 7-segment |
| Debug LEDs | 16 status LEDs |

### Architecture Highlights

LeNet-5 is one of the earliest convolutional neural networks, introduced by Yann LeCun in 1998. This implementation stays true to the original architecture while adapting it for modern FPGA deployment.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HOST PC                                     │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────────────────────┐  │
│  │   Training  │───►│   Quantized    │───►│   UART Transmission      │  │
│  │   (PyTorch) │    │   Weights      │    │   (send_weights.py)      │  │
│  │   LeNet-5   │    │   + Tanh LUT   │    │   62KB total             │  │
│  └─────────────┘    └────────────────┘    └────────────┬─────────────┘  │
│                                                        │                 │
│  ┌─────────────┐    ┌────────────────┐                │                 │
│  │   MNIST     │───►│   Preprocessed │────────────────┤                 │
│  │   Image     │    │   Pixels       │                │                 │
│  └─────────────┘    └────────────────┘                │                 │
└───────────────────────────────────────────────────────│─────────────────┘
                                                        │
                                              UART (115200 baud)
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            BASYS3 FPGA                                   │
│                                                                          │
│  ┌──────────────┐     ┌────────────────┐     ┌─────────────────────┐    │
│  │  UART Router │────►│  Weight Loader │────►│  Weight RAMs        │    │
│  │              │     └────────────────┘     │  - Conv Weights     │    │
│  │              │                            │  - Conv Biases      │    │
│  │              │     ┌────────────────┐     │  - FC Weights       │    │
│  │              │────►│  Image Loader  │     │  - FC Biases        │    │
│  │              │     └───────┬────────┘     │  - Tanh LUT         │    │
│  │              │             │              └──────────┬──────────┘    │
│  │              │             ▼                         │               │
│  │              │     ┌────────────────┐                │               │
│  │              │     │   Image RAM    │                │               │
│  │              │     │   (784 bytes)  │                │               │
│  │              │     └───────┬────────┘                │               │
│  │              │             │                         │               │
│  │              │             ▼                         ▼               │
│  │              │     ┌─────────────────────────────────────────┐       │
│  │              │     │       INFERENCE ENGINE (FSM)             │       │
│  │              │     │         35 States, 6-bit FSM             │       │
│  │              │     │                                          │       │
│  │              │     │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │       │
│  │              │     │  │Conv1 │─►│Pool1 │─►│Conv2 │─►│Pool2 │ │       │
│  │              │     │  │ 5x5  │  │2x2avg│  │ 5x5  │  │2x2avg│ │       │
│  │              │     │  │6filt │  │      │  │16filt│  │      │ │       │
│  │              │     │  └──┬───┘  └──────┘  └──────┘  └──┬───┘ │       │
│  │              │     │     │                             │     │       │
│  │              │     │     └──► Tanh LUT                 │     │       │
│  │              │     │                                   ▼     │       │
│  │              │     │  ┌───────────────────────────────────┐  │       │
│  │              │     │  │    Fully Connected Layers         │  │       │
│  │              │     │  │  FC1: 400→120 (Tanh)              │  │       │
│  │              │     │  │  FC2: 120→84 (Tanh)               │  │       │
│  │              │     │  │  FC3: 84→10 (Logits)              │  │       │
│  │              │     │  └───────────────┬───────────────────┘  │       │
│  │              │     │                  ▼                      │       │
│  │              │     │         ┌─────────────────┐             │       │
│  │              │     │         │ Predicted Digit │             │       │
│  │              │     │         │   + Scores      │             │       │
│  │              │     └─────────┴─────────────────┴─────────────┘       │
│  │              │                       │                               │
│  │              │◄──────────────────────┘                               │
│  │              │                                                       │
│  │              │     ┌────────────────┐     ┌────────────────┐         │
│  │              │────►│ Digit Reader   │────►│    UART TX     │────►TX  │
│  │              │     └────────────────┘     └────────────────┘         │
│  │              │     ┌────────────────┐            │                   │
│  │              │────►│ Scores Reader  │────────────┘                   │
│  └──────────────┘     └────────────────┘                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  7-Segment Display: Shows predicted digit                   │         │
│  │  Status LEDs: Debug and status information                  │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
le-net/
├── training/
│   └── train_lenet.py              # LeNet-5 training and quantization
│
├── testing/
│   ├── testing_python_model.py     # Bit-exact Python simulation
│   ├── compare_fpga_vs_python.py   # Full comparison harness
│   ├── generate_lenet_vectors.py   # Test vector generator
│   └── tanh_lut_loader.py          # Tanh LUT utilities
│
├── utils/
│   ├── send_weights.py             # Weight upload utility
│   └── send_image.py               # Image upload and inference
│
├── inference/
│   ├── rtl/                        # Verilog source files
│   │   ├── top.v                   # Top-level module
│   │   ├── inference.v             # LeNet-5 inference FSM (35 states)
│   │   ├── uart_router.v           # Protocol dispatcher
│   │   ├── uart_rx.v               # UART receiver
│   │   ├── uart_tx.v               # UART transmitter
│   │   ├── load_weights.v          # Weight parsing
│   │   ├── image_loader.v          # Image parsing
│   │   ├── ram_cnn.v               # Working buffers (A, B, C)
│   │   ├── conv_ram.v              # Conv weight/bias storage
│   │   ├── fc_ram.v                # FC weight/bias storage
│   │   ├── tanh_lut.v              # Tanh lookup table
│   │   ├── image_ram.v             # Input image storage
│   │   ├── scores_ram.v            # Output scores storage
│   │   ├── predicted_digit_ram.v   # Result storage
│   │   ├── digit_reader.v          # Result readback
│   │   ├── scores_reader.v         # Scores readback
│   │   └── seven_segment.v         # Display driver
│   │
│   ├── tb/                         # Testbenches
│   │   ├── tb_inference.v          # Full inference testbench
│   │   ├── tb_layer_conv1.v        # Conv1 layer test
│   │   ├── tb_layer_pool1.v        # Pool1 layer test
│   │   ├── tb_layer_conv2.v        # Conv2 layer test
│   │   ├── tb_layer_pool2.v        # Pool2 layer test
│   │   ├── tb_layer_fc1.v          # FC1 layer test
│   │   ├── tb_layer_fc2.v          # FC2 layer test
│   │   └── tb_layer_fc3.v          # FC3 layer test
│   │
│   └── constraints/
│       └── pins.xdc                # Basys3 pin mapping
│
└── outputs/                        # Generated files
    ├── bin/                        # Binary weight files
    ├── mem/                        # Hex memory files (includes tanh_lut.mem)
    └── npy/                        # Normalization params
```

---

## LeNet-5 Model Architecture

### Network Topology

The LeNet-5 architecture consists of 7 layers (2 convolutional + 2 pooling + 3 fully-connected):

```
INPUT: 28x28x1 grayscale image (784 bytes)
       Quantized to int8 range [-128, 127]

       ┌───────────────────────────────────────────┐
       │             INPUT (1x28x28)                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  CONV LAYER 1                              │
       │  - 6 filters, 5x5 kernel                   │
       │  - Padding 2, stride 1                     │
       │  - Output: 6x28x28                         │
       │  - Bit shift: >>10 (div 1024)              │
       │  - Activation: Tanh (via LUT)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  AVERAGE POOL 1                            │
       │  - 2x2 kernel, stride 2                    │
       │  - Sum of 4 values / 4                     │
       │  - Output: 6x14x14                         │
       │  - Scale: 127.0 / 4 = 31.75                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  CONV LAYER 2                              │
       │  - 16 filters, 5x5 kernel                  │
       │  - 6 input channels                        │
       │  - No padding, stride 1                    │
       │  - Output: 16x10x10                        │
       │  - Bit shift: >>8 (div 256)                │
       │  - Activation: Tanh (via LUT)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  AVERAGE POOL 2                            │
       │  - 2x2 kernel, stride 2                    │
       │  - Sum of 4 values / 4                     │
       │  - Output: 16x5x5                          │
       │  - Scale: 127.0 / 4 = 31.75                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  FLATTEN                                   │
       │  - 16x5x5 = 400 features                   │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  FC LAYER 1                                │
       │  - Input: 400                              │
       │  - Output: 120                             │
       │  - Bit shift: >>9 (div 512)                │
       │  - Activation: Tanh (via LUT)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  FC LAYER 2                                │
       │  - Input: 120                              │
       │  - Output: 84                              │
       │  - Bit shift: >>10 (div 1024)              │
       │  - Activation: Tanh (via LUT)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  FC LAYER 3 (Output)                       │
       │  - Input: 84                               │
       │  - Output: 10 class scores                 │
       │  - No activation (raw logits)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  OUTPUT: argmax(scores) = predicted digit  │
       └───────────────────────────────────────────┘
```

### Layer Dimensions Summary

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | 1x28x28 | 6x28x28 | 6x1x5x5 = 150 weights + 6 biases |
| Pool1 | 6x28x28 | 6x14x14 | 0 |
| Conv2 | 6x14x14 | 16x10x10 | 16x6x5x5 = 2,400 weights + 16 biases |
| Pool2 | 16x10x10 | 16x5x5 | 0 |
| FC1 | 400 | 120 | 400x120 = 48,000 weights + 120 biases |
| FC2 | 120 | 84 | 120x84 = 10,080 weights + 84 biases |
| FC3 | 84 | 10 | 84x10 = 840 weights + 10 biases |

**Total Parameters:** 61,470 (weights) + 236 (biases) = 61,706 parameters

---

## Python Training Pipeline

### Training Script (`train_lenet.py`)

```python
# Key hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
INPUT_SCALE = 127.0       # Quantization scale for input
POST_POOL_SCALE = 31.75   # Scale after average pooling (127/4)

# Per-layer bit shifts
SHIFT_CONV1 = 10          # Input scale = 127.0
SHIFT_CONV2 = 8           # Input scale = 31.75 (post-pool)
SHIFT_FC1 = 9             # Input scale = 31.75 (post-pool)
SHIFT_FC2 = 10            # Input scale = 127.0 (post-tanh)
```

### LeNet-5 Model Definition

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28 -> 28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28 -> 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14 -> 10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10 -> 5

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 -> 120
        self.fc2 = nn.Linear(120, 84)          # 120 -> 84
        self.fc3 = nn.Linear(84, num_classes)  # 84 -> 10

    def forward(self, x):
        # Conv blocks with tanh activation
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation on output

        return x
```

### Training Process

1. **Data Loading**: MNIST dataset with standard normalization (mean=0.1307, std=0.3081)
2. **Model Definition**: PyTorch `LeNet5` class
3. **Training Loop**: Adam optimizer with CrossEntropyLoss for 10 epochs
4. **Weight Export**: Quantized weights exported to binary and hex formats
5. **Tanh LUT Generation**: 256-entry lookup table for tanh approximation

### Data Preprocessing Pipeline

```python
# Raw image [0, 255] from MNIST
image = dataset[i]

# 1. Convert to tensor [0, 1]
x = transforms.ToTensor()(image)

# 2. Normalize with MNIST statistics
x = (x - 0.1307) / 0.3081

# 3. Quantize to int8
x = round(x * 127.0)
x = clip(x, -128, 127)

# Result: int8 pixel values ready for FPGA
```

---

## Quantization Strategy

The project uses **post-training static quantization** with careful attention to scale propagation through average pooling layers.

### Critical Insight: Average Pooling Scale Reduction

**Key difference from max pooling:** Average pooling divides by 4, reducing the scale by a factor of 4!

```
Input to Conv1:     Scale = 127.0
After Conv1+Tanh:   Scale = 127.0 (tanh renormalizes)
After Pool1 (÷4):   Scale = 31.75  ← CRITICAL
After Conv2+Tanh:   Scale = 127.0
After Pool2 (÷4):   Scale = 31.75  ← CRITICAL
After FC1+Tanh:     Scale = 127.0
After FC2+Tanh:     Scale = 127.0
```

### Quantization Equations

#### Weight Quantization (INT8)

```
W_scale = 127.0 / max(|W_float|)
W_int8 = clip(round(W_float * W_scale), -128, 127)
```

#### Bias Quantization (INT32)

The bias scale propagates through the network accounting for pooling:

```
Conv1:
  Input_scale = 127.0  (from normalized image quantization)
  W_scale_C1 = 127.0 / max(|conv1_weights|)
  Bias_scale_C1 = Input_scale * W_scale_C1
  Bias_C1_int32 = round(Bias_C1_float * Bias_scale_C1)

Conv1 Output Scale (after tanh):
  Output_scale_C1 = 127.0  (tanh renormalizes to [-127, 127])

Pool1 Output Scale:
  Output_scale_Pool1 = 127.0 / 4 = 31.75  ← Critical scaling

Conv2:
  Input_scale = 31.75  (NOT 127.0!)
  W_scale_C2 = 127.0 / max(|conv2_weights|)
  Bias_scale_C2 = 31.75 * W_scale_C2  ← Uses POST_POOL_SCALE
  Bias_C2_int32 = round(Bias_C2_float * Bias_scale_C2)

Conv2 Output Scale (after tanh):
  Output_scale_C2 = 127.0

Pool2 Output Scale:
  Output_scale_Pool2 = 127.0 / 4 = 31.75

FC1:
  Input_scale = 31.75  (NOT 127.0!)
  W_scale_FC1 = 127.0 / max(|fc1_weights|)
  Bias_scale_FC1 = 31.75 * W_scale_FC1  ← Uses POST_POOL_SCALE
  Bias_FC1_int32 = round(Bias_FC1_float * Bias_scale_FC1)

FC2:
  Input_scale = 127.0  (after tanh, no pooling)
  W_scale_FC2 = 127.0 / max(|fc2_weights|)
  Bias_scale_FC2 = 127.0 * W_scale_FC2
  Bias_FC2_int32 = round(Bias_FC2_float * Bias_scale_FC2)

FC3:
  Input_scale = 127.0
  W_scale_FC3 = 127.0 / max(|fc3_weights|)
  Bias_scale_FC3 = 127.0 * W_scale_FC3
  Bias_FC3_int32 = round(Bias_FC3_float * Bias_scale_FC3)
```

### FPGA Arithmetic Pipeline

For each convolution output:

```verilog
// 1. Accumulate (32-bit signed)
acc = bias + sum(pixel * weight)    // All sign-extended to 32-bit

// 2. Scale down (arithmetic right shift)
temp = acc >>> SHIFT                // Per-layer shift value

// 3. Tanh activation via LUT
if (temp < -128)
    tanh_idx = 0
else if (temp > 127)
    tanh_idx = 255
else
    tanh_idx = temp + 128

output = tanh_lut[tanh_idx]
```

For average pooling:

```verilog
// 1. Sum 4 values
sum = val[0] + val[1] + val[2] + val[3]

// 2. Integer divide by 4 (right shift 2)
avg = sum >>> 2

// Result is automatically scaled down by factor of 4
```

### Export File Formats

| File Type | Format | Description |
|-----------|--------|-------------|
| `.bin` | Binary | Raw bytes, direct memory load |
| `.mem` | Hex ASCII | One value per line, for Verilog `$readmemh` |
| `.npy` | NumPy | Normalization parameters |

#### Weight File Sizes

| File | Size (bytes) | Format |
|------|--------------|--------|
| conv1_weights.bin | 150 | 6x1x5x5 int8 |
| conv1_biases.bin | 24 | 6 int32 |
| conv2_weights.bin | 2,400 | 16x6x5x5 int8 |
| conv2_biases.bin | 64 | 16 int32 |
| fc1_weights.bin | 48,000 | 120x400 int8 |
| fc1_biases.bin | 480 | 120 int32 |
| fc2_weights.bin | 10,080 | 84x120 int8 |
| fc2_biases.bin | 336 | 84 int32 |
| fc3_weights.bin | 840 | 10x84 int8 |
| fc3_biases.bin | 40 | 10 int32 |
| tanh_lut.mem | 256 | 256 int8 entries |
| **Total** | **62,670** | |

---

## FPGA Implementation

### Module Hierarchy

```
top.v
├── uart_router.v           # Parses UART stream, routes to loaders
│   └── uart_rx.v           # 8N1 UART receiver
│
├── load_weights.v          # Parses weight packet into RAMs
│
├── image_loader.v          # Parses image packet into RAM
│
├── conv_ram.v              # Conv weights and biases
│   ├── weights             # 2,550 bytes (150 + 2,400)
│   └── biases              # 22 x 32-bit (6 + 16)
│
├── fc_ram.v                # FC weights and biases
│   ├── weights (BRAM)      # 58,920 bytes (48K + 10K + 840)
│   └── biases              # 214 x 32-bit (120 + 84 + 10)
│
├── tanh_lut.v              # 256-entry tanh lookup table
│
├── image_ram.v             # 784 bytes
│
├── ram_cnn.v               # Working memory
│   ├── buffer_a            # 4,704 bytes (Conv1 output: 6x28x28)
│   ├── buffer_b            # 1,600 bytes (Conv2 output: 16x10x10)
│   └── buffer_c            # 400 bytes (Pool2 output or FC intermediates)
│
├── inference.v             # Main FSM engine (35 states)
│
├── predicted_digit_ram.v   # 1 byte result
├── scores_ram.v            # 40 bytes (10 x int32)
│
├── digit_reader.v          # Response to 0xCC command
├── scores_reader.v         # Response to 0xCD command
├── uart_tx.v               # 8N1 UART transmitter
│
└── seven_segment.v         # Display driver
```

### Inference FSM States

The LeNet-5 inference engine is implemented as a 35-state finite state machine:

```
State                    | Description
-------------------------|-------------------------------------------
IDLE                     | Wait for start signal

--- Conv1 Layer (6 filters, 5x5) ---
L1_LOAD_BIAS             | Request bias from conv biases RAM
L1_LOAD_BIAS_WAIT        | Wait 1 cycle for RAM read
L1_PREFETCH              | Prefetch first pixel/weight
L1_CONV                  | 5x5 MAC loop (25 cycles per position)
L1_TANH                  | Apply tanh via LUT lookup
L1_SAVE                  | Write result to buffer A

--- Pool1 Layer (2x2 average) ---
L1_POOL                  | Fetch 4 values for 2x2 window
L1_POOL_CALC             | Sum and divide by 4

--- Conv2 Layer (16 filters, 5x5, 6 input channels) ---
L2_LOAD_BIAS             | Request L2 bias
L2_LOAD_BIAS_WAIT        | Wait for RAM
L2_PREFETCH              | Prefetch first activation/weight
L2_CONV                  | 5x5x6 MAC loop (150 cycles per position)
L2_TANH                  | Apply tanh via LUT
L2_SAVE                  | Write result to buffer B

--- Pool2 Layer (2x2 average) ---
L2_POOL                  | Fetch 4 values for 2x2 window
L2_POOL_CALC             | Sum and divide by 4

--- FC1 Layer (400 -> 120) ---
FC1_LOAD_BIAS            | Request FC1 bias
FC1_LOAD_BIAS_WAIT       | Wait for RAM
FC1_PREFETCH             | Prefetch first feature/weight
FC1_PREFETCH2            | Extra wait for BRAM latency
FC1_MULT                 | 400 MAC operations per neuron
FC1_MULT_WAIT            | Wait cycle for BRAM data
FC1_TANH                 | Apply tanh via LUT
FC1_SAVE                 | Store result

--- FC2 Layer (120 -> 84) ---
FC2_LOAD_BIAS            | Request FC2 bias
FC2_LOAD_BIAS_WAIT       | Wait for RAM
FC2_PREFETCH             | Prefetch first feature/weight
FC2_PREFETCH2            | Extra wait for BRAM latency
FC2_MULT                 | 120 MAC operations per neuron
FC2_MULT_WAIT            | Wait cycle for BRAM data
FC2_TANH                 | Apply tanh via LUT
FC2_SAVE                 | Store result

--- FC3 Layer (84 -> 10) ---
FC3_LOAD_BIAS            | Request FC3 bias
FC3_LOAD_BIAS_WAIT       | Wait for RAM
FC3_PREFETCH             | Prefetch first feature/weight
FC3_MULT                 | 84 MAC operations per class
FC3_NEXT                 | Store score, track argmax

DONE_STATE               | Signal completion, return to IDLE
```

### Computation Counts

| Layer | Operations per output | Total outputs | Total MACs |
|-------|----------------------|---------------|------------|
| Conv1 | 1x5x5 = 25 | 6x28x28 = 4,704 | 117,600 |
| Conv2 | 6x5x5 = 150 | 16x10x10 = 1,600 | 240,000 |
| FC1 | 400 | 120 | 48,000 |
| FC2 | 120 | 84 | 10,080 |
| FC3 | 84 | 10 | 840 |
| **Total** | | | **416,520** |

### RAM Architecture

**Distributed RAM** (LUT-based) for small, frequently accessed memories:
- Working buffers (buffer_a, buffer_b, buffer_c)
- Conv weights and biases
- Tanh LUT
- Image RAM

**Block RAM** (BRAM) for large FC weights:
- FC weights: 58,920 bytes (too large for distributed RAM)

```verilog
// Distributed RAM (0-cycle read latency)
(* ram_style = "distributed" *)
reg [7:0] ram [0:SIZE-1];
assign rd_data = ram[rd_addr];  // Combinational

// Block RAM (1-cycle read latency)
(* ram_style = "block" *)
reg [7:0] ram [0:SIZE-1];
always @(posedge clk) rd_data_reg <= ram[rd_addr];  // Registered
```

---

## Communication Protocol

### UART Configuration

- **Baud Rate:** 115,200
- **Frame Format:** 8N1 (8 data bits, no parity, 1 stop bit)
- **Clock Frequency:** 100 MHz

### Protocol Markers

| Marker | Bytes | Direction | Purpose |
|--------|-------|-----------|------------|
| `0xAA 0x55` | Weight start | PC to FPGA | Begin weight transfer |
| `0x55 0xAA` | Weight end | PC to FPGA | End weight transfer |
| `0xBB 0x66` | Image start | PC to FPGA | Begin image transfer |
| `0x66 0xBB` | Image end | PC to FPGA | End image transfer |
| `0xCC` | Command | PC to FPGA | Request predicted digit |
| `0xCD` | Command | PC to FPGA | Request all scores |

### Weight Packet Structure (62,670 bytes)

```
Offset    Size      Data
----------------------------------------------
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
----------------------------------------------
Total:    62,670 bytes
```

### Image Packet Structure (784 bytes)

```
Offset    Size      Data
----------------------------------------------
0         784       Pixel values (28x28, row-major)
                    Preprocessed, quantized int8
----------------------------------------------
```

### Binary Safety

The protocol uses **payload-length-aware parsing** to prevent false end-marker detection:

```verilog
// Only check end markers AFTER receiving expected payload size
if (byte_count >= WEIGHT_SIZE &&
    prev_byte == WEIGHT_END1 &&
    rx_data == WEIGHT_END2) begin
    state <= DONE_W;
end
```

### Flow Control

To prevent UART buffer overflow, Python scripts use chunked transmission:

```python
def send_chunked(ser, data, chunk_size=32, delay=0.010):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        ser.write(chunk)
        ser.flush()
        time.sleep(delay)  # 10ms delay per chunk
```

---

## Testing Infrastructure

### Python Simulation (`testing_python_model.py`)

Provides **bit-exact simulation** of the FPGA inference pipeline:

```python
def simulate_quantized_inference(image_bytes, weights_dict, tanh_lut):
    # Conv1: 5x5 with padding=2, tanh activation
    x = convolve_layer(img, c1_w, c1_b, SHIFT_CONV1, tanh_lut)
    x = avg_pool_2x2(x)  # 28x28 -> 14x14, scale ÷4

    # Conv2: 5x5 no padding, tanh activation
    x = convolve_layer(x, c2_w, c2_b, SHIFT_CONV2, tanh_lut)
    x = avg_pool_2x2(x)  # 10x10 -> 5x5, scale ÷4

    # FC1: 400 -> 120, tanh activation
    x = fc_layer(x.flatten(), fc1_w, fc1_b, SHIFT_FC1, tanh_lut)

    # FC2: 120 -> 84, tanh activation
    x = fc_layer(x, fc2_w, fc2_b, SHIFT_FC2, tanh_lut)

    # FC3: 84 -> 10, no activation
    scores = fc_layer_no_activation(x, fc3_w, fc3_b)

    return np.argmax(scores)
```

Key simulation details:
- **Exact LUT usage**: Loads `tanh_lut.mem` generated by training script
- **Bit-exact shifts**: Uses same shift values as FPGA
- **Integer-only arithmetic**: All operations use int32
- **Average pooling**: Integer divide by 4 (`sum >> 2`)

### Comparison Framework (`compare_fpga_vs_python.py`)

Tests up to 10,000 MNIST images comparing Python and FPGA results:

```bash
python compare_fpga_vs_python.py --port COM7 --count 1000 --index 0
```

**Test Flow:**

1. Load and preprocess image
2. Run Python simulation - get scores and prediction
3. Send image to FPGA via UART
4. Wait for inference completion (~200ms for LeNet-5)
5. Request scores via `0xCD` command
6. Compare bit-exact match and accuracy

### Test Vector Generation (`generate_lenet_vectors.py`)

Generates `.mem` files for Verilog testbenches:

- `test_pixels.mem` - Preprocessed pixel values
- `test_scores.mem` - Expected output scores
- `test_preds.mem` - Expected predictions
- `test_labels.mem` - Ground truth labels
- Per-layer intermediate results for layer-specific testbenches

### Tanh LUT Loader (`tanh_lut_loader.py`)

Utility module for loading and applying tanh LUT in Python:

```python
def load_tanh_lut():
    """Load tanh LUT from tanh_lut.mem file"""
    lut = np.fromfile("tanh_lut.mem", dtype=np.int8)
    return lut

def apply_tanh_lut(x, lut):
    """Apply tanh LUT to value(s)"""
    x = np.clip(x, -128, 127)
    idx = (x + 128).astype(np.int32)
    return lut[idx]
```

---

## Memory Organization

### Total Memory Footprint

```
Weight Storage (Read-only after loading):
  ├── Conv weights:        2,550 bytes
  ├── Conv biases:            88 bytes (22 x 32-bit)
  ├── FC weights (BRAM):  58,920 bytes
  ├── FC biases:             856 bytes (214 x 32-bit)
  └── Tanh LUT:              256 bytes
  Subtotal:               62,670 bytes

Working Memory (Distributed RAM):
  ├── Buffer A:            4,704 bytes (Conv1 output: 6x28x28)
  ├── Buffer B:            1,600 bytes (Conv2 output: 16x10x10)
  └── Buffer C:              400 bytes (Pool2/FC intermediates: 16x5x5 or 120)
  Subtotal:                6,704 bytes

Input/Output:
  ├── Image RAM:             784 bytes
  ├── Scores RAM:             40 bytes
  └── Digit RAM:               1 byte
  Subtotal:                  825 bytes

-----------------------------------------
TOTAL:                  ~70,199 bytes
```

### Address Mapping

**Conv Weights RAM (2,550 bytes):**
```
Address Range          | Content
-----------------------|----------------------
0-149                  | Conv1: 6 filters x 25
150-2,549              | Conv2: 16 filters x 150
```

**Conv Biases RAM (22 entries):**
```
Address    | Content
-----------|----------------------
0-5        | Conv1 biases (6)
6-21       | Conv2 biases (16)
```

**FC Weights RAM (58,920 bytes, BRAM):**
```
Address Range          | Content
-----------------------|----------------------
0-47,999               | FC1: 120 neurons x 400
48,000-58,079          | FC2: 84 neurons x 120
58,080-58,919          | FC3: 10 neurons x 84
```

**FC Biases RAM (214 entries):**
```
Address    | Content
-----------|----------------------
0-119      | FC1 biases (120)
120-203    | FC2 biases (84)
204-213    | FC3 biases (10)
```

**Buffer A Address Calculation:**
```
Conv1 output: addr = filter_idx * 784 + row * 28 + col
              where 784 = 28 * 28
```

**Buffer B Address Calculation:**
```
Pool1 output: addr = filter_idx * 196 + row * 14 + col
              where 196 = 14 * 14

Conv2 output: addr = filter_idx * 100 + row * 10 + col
              where 100 = 10 * 10
```

**Buffer C Address Calculation:**
```
Pool2 output: addr = filter_idx * 25 + row * 5 + col
              where 25 = 5 * 5

FC1 output:   addr = neuron_idx (0-119)
```

---

## Tanh Activation Implementation

### Tanh Function Characteristics

The hyperbolic tangent function maps `(-∞, ∞)` to `(-1, 1)`:

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

For FPGA implementation, we use a **256-entry lookup table** to approximate tanh.

### LUT Generation (Python)

```python
def generate_tanh_lut(output_dir):
    """
    Generate 256-entry tanh lookup table.
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

    return np.array(lut, dtype=np.int8)
```

### LUT Characteristics

```
Input scaling factor: 32.0
  - Input value -128 represents -4.0
  - Input value 0 represents 0.0
  - Input value 127 represents ~3.97

Output range: [-127, 127]
  - Maps to [-1.0, 1.0] in floating-point

Saturation:
  - tanh(-4.0) ≈ -0.9993 → -127
  - tanh(0.0) = 0.0 → 0
  - tanh(4.0) ≈ 0.9993 → 127
```

### FPGA LUT Module (`tanh_lut.v`)

```verilog
module tanh_lut (
    input wire [7:0] addr,
    output wire signed [7:0] data
);

    (* ram_style = "distributed" *)
    reg [7:0] lut [0:255];

    initial begin
        $readmemh("tanh_lut.mem", lut);
    end

    // Combinational read (0-cycle latency)
    assign data = lut[addr];

endmodule
```

### Usage in Inference FSM

```verilog
// After convolution and shift
temp_val <= acc >>> SHIFT;

// Clip to valid LUT range and convert to unsigned index
if (temp_val < -128)
    tanh_addr <= 8'd0;
else if (temp_val > 127)
    tanh_addr <= 8'd255;
else
    tanh_addr <= temp_val[7:0] + 8'd128;

// LUT lookup (combinational, same cycle)
// tanh_data now contains activated value

// Store to buffer
buf_a_wr_data <= tanh_data;
buf_a_wr_en <= 1;
```

### Comparison: Tanh vs ReLU

| Aspect | ReLU | Tanh (LUT) |
|--------|------|-----------|
| Computation | `max(0, x)` | LUT lookup |
| FPGA Resources | Minimal (comparator) | 256 bytes distributed RAM |
| Latency | 0 cycles | 0 cycles (distributed RAM) |
| Output Range | `[0, 127]` | `[-127, 127]` |
| Gradients | Non-zero for x>0 | Non-zero everywhere |
| Symmetry | Not symmetric | Symmetric around 0 |
| Saturation | Hard at 0 | Soft saturation at ±127 |

---

## Performance Metrics

### Accuracy Results

Testing on 9,500 MNIST images (indices 500-9,999):

| Model | Accuracy | Correct Predictions |
|-------|----------|---------------------|
| FP32 (PyTorch) | 97.92% | 9,302 / 9,500 |
| INT8 (Quantized) | 97.92% | 9,302 / 9,500 |
| **Accuracy Drop** | **0.00%** | Bit-exact match |

### Inference Latency (Estimated)

Assuming 100 MHz clock (10 ns per cycle):

| Layer | Cycles | Time (ms) |
|-------|--------|-----------|
| Conv1 | ~120,000 | 1.2 |
| Pool1 | ~5,000 | 0.05 |
| Conv2 | ~250,000 | 2.5 |
| Pool2 | ~2,000 | 0.02 |
| FC1 | ~50,000 | 0.5 |
| FC2 | ~11,000 | 0.11 |
| FC3 | ~1,000 | 0.01 |
| **Total** | **~439,000** | **~4.4 ms** |

Actual measured latency (including memory access overhead): **~150-200 ms**

### Resource Utilization (Artix-7 XC7A35T)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | ~15,000 | 20,800 | ~72% |
| LUT-RAM | ~8,000 | 9,600 | ~83% |
| BRAM (18Kb) | 16 | 50 | 32% |
| DSPs | 0 | 90 | 0% |
| Flip-Flops | ~2,500 | 41,600 | ~6% |

**Notes:**
- High LUT-RAM usage due to distributed RAM for working buffers
- FC weights use BRAM to conserve LUT resources
- No DSP blocks used (pure integer multiply-accumulate)

### Comparison: CNN vs LeNet-5

| Metric | 2-Layer CNN | LeNet-5 |
|--------|-------------|---------|
| Accuracy | 99.0% | 97.9% |
| Parameters | 12,868 | 61,706 |
| Weight Memory | ~13 KB | ~62 KB |
| Working Memory | ~14 KB | ~7 KB |
| Total Memory | ~27 KB | ~70 KB |
| FSM States | 19 | 35 |
| Inference Latency | ~100 ms | ~150-200 ms |
| LUT Utilization | ~60% | ~72% |
| BRAM Utilization | ~20% | ~32% |

---

## Migration Notes

This LeNet-5 implementation was derived from the original 2-layer CNN implementation. Key changes:

### Major Architectural Changes

1. **Kernel Size**: 3x3 → 5x5 (requires more MAC operations)
2. **Pooling**: Max pooling → Average pooling (scale reduction!)
3. **Activation**: ReLU → Tanh (requires LUT)
4. **FC Layers**: 1 layer (800→10) → 3 layers (400→120→84→10)
5. **Filters**: Conv1 reduced (16→6), Conv2 reduced (32→16)

### Critical Fixes

**Average Pooling Scale Propagation:**
- Initial implementation failed to account for scale reduction after pooling
- After `sum >> 2` (divide by 4), scale drops from 127.0 to 31.75
- This affects bias quantization for Conv2 and FC1
- Fix: Use `POST_POOL_SCALE = 127.0 / 4` for layers following pooling

**BRAM Latency Handling:**
- FC weights use BRAM (block RAM) due to size
- BRAM has 1-2 cycle read latency vs 0 for distributed RAM
- Added extra wait states (FC1_PREFETCH2, FC1_MULT_WAIT, etc.)

### Files Modified/Added

**New Files:**
- `inference/rtl/fc_ram.v` - FC weights/biases RAM
- `inference/rtl/tanh_lut.v` - Tanh lookup table
- `testing/tanh_lut_loader.py` - Python tanh utilities
- `LENET_CHANGES.md` - Migration documentation
- `QUANTIZATION_FIX_PLAN.md` - Scale propagation fix details

**Modified Files:**
- `training/train_lenet.py` - LeNet-5 model and quantization
- `inference/rtl/inference.v` - 35-state FSM for LeNet-5
- `inference/rtl/ram_cnn.v` - Updated buffer sizes
- `testing/testing_python_model.py` - Bit-exact LeNet-5 simulation

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). **Gradient-based learning applied to document recognition.** Proceedings of the IEEE, 86(11), 2278-2324.
2. Digilent Basys3 Reference Manual
3. Xilinx UG901: Vivado Design Suite User Guide - Synthesis
4. Xilinx UG473: 7 Series FPGAs Memory Resources User Guide

---

## Appendix: Known Issues and Solutions

### Issue 1: Accuracy Drop from Pooling Scale

**Symptom:** Initial quantized model had 5-10% accuracy drop from FP32.

**Root Cause:** Average pooling divides by 4, reducing scale from 127.0 to 31.75. Bias quantization for subsequent layers used incorrect scale (127.0).

**Solution:** Use `POST_POOL_SCALE = 127.0 / 4.0` for Conv2 and FC1 bias quantization.

### Issue 2: BRAM Read Latency

**Symptom:** FC layer outputs incorrect when using BRAM for weights.

**Root Cause:** BRAM has registered output (1-2 cycle latency) vs distributed RAM (0 cycles).

**Solution:** Added prefetch and wait states in FSM to account for BRAM latency.

### Issue 3: Tanh LUT Index Calculation

**Symptom:** Negative values incorrectly indexed into tanh LUT.

**Root Cause:** Signed int8 value needs offset (+128) to become valid LUT index [0, 255].

**Solution:** Clip to [-128, 127], then add 128 for unsigned index.

---

*Document generated for the MNIST-FPGA LeNet-5 project. Last updated: January 2025.*
