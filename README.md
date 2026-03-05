# Implementation of Convolutional Neural Network on FPGA for MNIST Digit Recognition

Engineering thesis project implementing neural network inference on an FPGA for handwritten digit classification. The project traces a progression from simple logistic regression through multi-layer perceptrons to a LeNet-5 CNN, with the final model deployed on an Artix-7 FPGA using INT8 quantization and UART communication.

## Project Goal

- Train multiple model architectures in PyTorch (logistic regression, MLP, CNN, LeNet-5)
- Quantize trained models to INT8 weights / INT32 biases
- Implement inference in synthesizable Verilog for Artix-7 FPGA
- Communicate with FPGA via UART (weight loading + image inference)
- Display predicted digit on the Basys3 7-segment display

## Model Architectures

| Directory | Model | Architecture | Activation | Accuracy |
|-----------|-------|-------------|------------|----------|
| `regresja/` | Softmax regression | 784 → 10 | — | 92.13% |
| `2_ukryte/` | 2-hidden-layer MLP | 784 → 16 → 16 → 10 | ReLU | 95.7% |
| `cnn/` | Simple CNN | 1 conv + 1 dense | ReLU | 99% |
| `le-net/` | **LeNet-5** | 2 conv + 3 FC | tanh | **97.6%** |

The LeNet-5 implementation in `le-net/` is the primary and most complete version, with full Verilog RTL, testbenches, and UART integration.

## Folder Structure

| Folder | Description |
|--------|-------------|
| `le-net/` | LeNet-5 CNN — training, quantization, Verilog RTL, testbenches, UART utilities |
| `regresja/` | Softmax (logistic) regression model |
| `2_ukryte/` | Two-hidden-layer MLP (784 → 16 → 16 → 10) |
| `cnn/` | Simple CNN (1 conv layer + 1 dense layer) |
| `test_images/` | PNG test images for FPGA inference testing |
| `text/` | LaTeX thesis and presentation source files |

## Workflow

1. **Train and quantize** — `uv run python le-net/training/train_lenet.py`
2. **Outputs** — `.bin`, `.mem`, `.npy` files generated in `le-net/outputs/`
3. **Synthesize** — open Vivado project, run synthesis and implementation for Artix-7 XC7A35T
4. **Upload weights** — `uv run python le-net/utils/send_weights.py`
5. **Send image** — `uv run python le-net/testing/send_image.py`
6. **Read result** — predicted digit shown on the 7-segment display

## Hardware

- **Board:** Digilent Basys3 (Artix-7 XC7A35T-1CPG236C)
- **Clock:** 100 MHz
- **Interface:** USB-UART at 115200 baud

## Dependencies

Install Python dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Main packages: `torch`, `torchvision`, `scikit-learn`, `numpy`, `pillow`, `pyserial`, `pandas`, `torchinfo`, `pypdf`

Additional tools:
- **Xilinx Vivado** — FPGA synthesis and implementation

## Testing

```bash
# Bit-exact quantized model accuracy (no hardware needed)
uv run python le-net/testing/testing_python_model.py

# Compare FPGA output vs Python model (requires FPGA connected via UART)
uv run python le-net/testing/compare_fpga_vs_python.py --port COMX --count 100
```

## Thesis

The full thesis document is included as `text/thesis/text.pdf`.
