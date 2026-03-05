# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engineering thesis project: FPGA-based LeNet-5 inference for MNIST digit classification. Three components: Python training/quantization pipeline, Verilog RTL for Artix-7 FPGA (Basys3), and a LaTeX thesis document.

## Build Commands

### Python (package manager: uv)
```bash
uv sync                                    # install dependencies
uv run python le-net/training/train_lenet.py  # train LeNet-5, export quantized weights
```

### LaTeX thesis
```bash
cd text/thesis
latexmk -pdf text.tex          # full build (pdflatex + biber)
# or manually:
pdflatex text.tex && biber text && pdflatex text.tex && pdflatex text.tex
```

### FPGA (Xilinx Vivado)
Synthesis and implementation are done in Vivado GUI. Constraints file: `le-net/inference/constraints/pins.xdc`. Target: Artix-7 XC7A35T-1CPG236C (Basys3), 100 MHz clock.

## Architecture

### Training → FPGA pipeline
1. `le-net/training/train_lenet.py` trains LeNet-5 in PyTorch, quantizes to INT8 weights / INT32 biases, generates tanh LUT
2. Outputs go to `le-net/outputs/` in three formats: `.bin` (binary), `.mem` (hex for Verilog `$readmemh`), `.npy`
3. `le-net/utils/send_weights.py` sends 62,670-byte weight packet to FPGA over UART (115200 baud)
4. `le-net/utils/send_image.py` sends a 784-byte image and reads back the predicted digit

### Verilog RTL (`le-net/inference/rtl/`)
- `top.v` — top-level module wiring everything together
- `inference.v` — 44-state FSM implementing Conv1→Tanh→Pool1→Conv2→Tanh→Pool2→FC1→FC2→FC3
- `uart_router.v` — parses UART byte stream, routes to weight/image loaders
- `ram_cnn.v`, `fc_ram.v`, `tanh_lut.v`, `image_ram.v` — memory subsystem (distributed RAM + BRAM)
- Testbenches in `le-net/inference/tb/` — per-layer unit tests and full-system tests

### Quantization constants (fixed in train_lenet.py)
```
INPUT_SCALE=127.0  SHIFT_CONV1=10  SHIFT_CONV2=8  SHIFT_FC1=9  SHIFT_FC2=10
```
These must stay synchronized between the Python training script and the Verilog RTL shift amounts.

### LaTeX thesis (`text/thesis/text.tex`)
Single monolithic file (~1500 lines). Bibliography via biblatex/biber from `references.bib`. Heavy use of TikZ/pgfplots for diagrams. The thesis is in English with Polish babel support for the title page.

## Testing

No automated test runner (no pytest). Testing is script-based:
```bash
# Bit-exact integer simulation (no hardware needed)
uv run python le-net/testing/testing_python_model.py

# Generate .mem test vectors for Verilog testbenches
uv run python le-net/testing/generate_lenet_vectors.py

# Compare FPGA output vs Python (requires FPGA connected via UART)
uv run python le-net/testing/compare_fpga_vs_python.py --port COM7 --count 100
```

## Key Documentation

- `le-net/PROJECT_DOCUMENTATION.md` — detailed technical reference (memory maps, UART protocol, FSM states, resource utilization)
- `le-net/LENET_CHANGES.md` — migration notes from generic CNN to LeNet-5

## Legacy Directories

`regresja/`, `2_ukryte/`, `cnn/` contain older/simpler model implementations. The active implementation is in `le-net/`. `le-net-szulejko/` is a streamlined copy of `le-net/` with fewer testbenches.
