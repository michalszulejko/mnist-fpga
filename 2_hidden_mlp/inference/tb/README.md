# Isolated Testbench for 2-Hidden-Layer Inference Module

This testbench tests `inference.v` in **complete isolation** from UART, weight loaders, and other modules. It loads test data directly from `.mem` files and compares FPGA results with Python simulation results.

## Purpose

The testbench helps diagnose whether computation errors are in:

- `inference.v` itself (pipeline timing, accumulator, sign extension, etc.)

- External modules (UART, weight_loader, image_loader, etc.)

If the testbench passes but hardware fails, the problem is in data loading/communication.

If the testbench fails, the problem is in `inference.v` computation.

## Prerequisites

1. Trained model with quantized weights in `2_ukryte/outputs/mem/`:

   - `L1_weights.mem`, `L1_biases.mem`

   - `L2_weights.mem`, `L2_biases.mem`

   - `L3_weights.mem`, `L3_biases.mem`

2. Python with numpy and torchvision (or sklearn):

   ```bash

   pip install numpy torch torchvision

   ```

## Usage

### Step 1: Generate Test Vectors

Run the Python script to generate test vectors:

```bash

cd 2_ukryte/testing

python generate_test_vectors.py

```

This creates in `2_ukryte/inference/tb/`:

- `test_vectors_pixels.mem` - 78,400 lines (100 images × 784 pixels)

- `test_vectors_scores.mem` - 1,000 lines (100 images × 10 scores)

- `test_vectors_meta.mem` - 100 lines (expected predictions)

- `test_vectors_labels.mem` - 100 lines (true labels)

### Step 2: Run Simulation in Vivado

1. Open your Vivado project

2. Add `tb_inference.v` as a simulation source

3. Set `tb_inference` as the top module for simulation

4. Run Behavioral Simulation

Alternatively, for command-line simulation:

```bash

cd 2_ukryte/inference/tb

xvlog ../rtl/inference.v tb_inference.v

xelab tb_inference -s sim

xsim sim -runall

```

### Step 3: Analyze Results

The testbench will output:

- Per-test-case comparison of all 10 class scores

- Whether scores match exactly

- Whether predictions match

- Summary statistics

## Expected Output

If `inference.v` is working correctly:

```

================================================================================

FINAL RESULTS

================================================================================

Total Tests:             100

Exact Score Matches:     100

Score Mismatches:        0

Prediction Matches:      100 (100.0%)

================================================================================



ALL TESTS PASSED - EXACT SCORE MATCH

inference.v is computing correctly!

```

If there are computation errors:

```

================================================================================

FINAL RESULTS

================================================================================

Total Tests:             100

Exact Score Matches:     0

Score Mismatches:        100

Prediction Matches:      30 (30.0%)

================================================================================



TESTS FAILED

inference.v has computation errors!

```

## Troubleshooting

### "Cannot open file" errors

- Make sure you run simulation from the `tb/` directory

- Verify weight files exist in `../../outputs/mem/`

- Run `generate_test_vectors.py` first

### Simulation timeout

- The 3-layer network takes ~13,000 cycles per inference

- Total for 100 tests: ~1.3M cycles = ~13ms at 100MHz

- Timeout is set to 500ms which should be sufficient

### Score mismatches

- Check the detailed per-class comparison in the output

- Look for patterns (e.g., all class 7 scores wrong)

- Compare max error values

## Files

```
2_ukryte/inference/tb/
├── README.md                    # This file
├── tb_inference.v               # Testbench (Verilog)
├── test_vectors_pixels.mem      # Generated: preprocessed images
├── test_vectors_scores.mem      # Generated: expected scores
├── test_vectors_meta.mem        # Generated: expected predictions
└── test_vectors_labels.mem      # Generated: true labels
```
