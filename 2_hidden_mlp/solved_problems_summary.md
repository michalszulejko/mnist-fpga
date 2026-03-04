# Debugging Summary: FPGA Neural Network Implementation

## Executive Summary

This document summarizes the debugging process for the FPGA implementation of a 2-Hidden-Layer MLP (Neural Network) for MNIST digit recognition. The project encountered two major issues that prevented correct hardware operation despite passing all isolated testbench simulations.

**Final Status**: ✅ **100% Bit-Perfect Hardware Acceleration Achieved**
- All 100 test images produce identical scores between Python simulation and FPGA
- Zero mismatches in predictions or raw score values
- Complete hardware-software verification

---

## Session 1: Binary Safety & Protocol Issues

### Initial Problem

The `inference.v` module passed all isolated testbench simulations with 100% accuracy, but the actual hardware implementation failed or produced garbage results when loaded via UART.

**Symptom**: Correct predictions (e.g., "It's a 7") but significantly different raw scores compared to Python simulation.

### Initial Diagnosis

**Lack of "Binary Safety" in the UART Protocol**

The `uart_router` was interpreting raw data bytes (weights or pixels) as protocol "End Markers" (`0x55 0xAA` or `0x66 0xBB`) if those specific values appeared randomly in the data stream. This caused the FPGA to terminate data loading prematurely, leaving the rest of the neural network memory empty (zeros), leading to incorrect predictions.

### Bugs Identified & Fixed

#### 1. `uart_router.v` - Binary Safety Fix

**Issue**: Terminating transfer early on random data bytes that matched protocol markers.

**Root Cause**: The router was continuously checking for end markers throughout the entire data stream, not just at the expected end.

**Fix**: Implemented a byte counter. The router now explicitly ignores the UART input stream for marker detection until the expected number of bytes has officially arrived:
- 13,128 bytes for weights
- 784 bytes for images

**Result**: Prevents false-positive marker detection during data transmission.

#### 2. `uart_tx.v` - Timing/Race Condition Fix

**Issue**: The `busy` signal was a register, causing a 1-clock-cycle delay before going high. The `scores_reader` was checking `busy` too early and overwriting the transmit buffer before the previous byte was sent.

**Root Cause**: Register-based `busy` signal created a race condition where new data could be written while previous transmission was still in progress.

**Fix**: Changed `busy` to a combinatorial wire logic so it asserts instantly when a send command is issued.

**Result**: Eliminates race conditions in UART transmission.

#### 3. `image_loader.v` - Alignment Fix

**Issue**: The `uart_router` consumes the start sequence (`0xBB 0x66`) but passes the final byte (`0x66`) to the loader. The loader treated this marker byte as Pixel 0.

**Result**: The entire image in FPGA RAM was shifted right by 1 byte/pixel, causing all subsequent pixels to be misaligned.

**Fix**: Added a `first_byte_flag` to drop the very first byte received from the router (which is the leftover marker byte).

**Result**: Correct pixel alignment in image memory.

### Tests Performed (Session 1)

#### `generate_test_vectors.py`
- **Purpose**: Created "Gold Standard" inputs and outputs to verify the logic of `inference.v` in isolation.
- **Result**: Generated test vectors for 100 images with expected scores.

#### `tb_inference.v` (Vivado Simulation)
- **Purpose**: Verified that the Verilog math (MAC operations, ReLU, bit-shifting) exactly matches the Python simulation.
- **Result**: ✅ **PASSED 100%** - Proved the computation logic is correct.

#### `compare_python_vs_fpga.py`
- **Purpose**: End-to-end integration test. Sends an image via UART, reads back the raw scores (logits), and compares them to Python.
- **Result**: Predictions matched (e.g., "It's a 7") but raw scores differed significantly.
- **Significance**: This specific pattern (Correct Prediction / Incorrect Score) led to the discovery of the 1-pixel image shift. The network still recognized the digit despite the shift, but the math was slightly off.

### Key Insight from Session 1

The pattern of **correct predictions with incorrect scores** indicated that:
- The inference computation logic was correct (predictions matched)
- The data was being loaded, but with a systematic offset (1-byte shift)
- This pointed to alignment issues in the data loading pipeline

### Status After Session 1

After applying fixes to `uart_router.v`, `uart_tx.v`, and `image_loader.v`, a critical observation was made:

**The "Phantom" Run**: After applying the fix to `image_loader.v` and running the test again, the results were bit-for-bit identical to the previous broken run.

**Conclusion**: The FPGA was not updated. Changing the Verilog code does not automatically update the hardware.

**Action Required**: Must run Synthesis, Implementation, and Generate Bitstream in Vivado, then Program the Device with the new `.bit` file before fixes take effect.

---

## Session 2: Throughput & Flow Control Issues

### Initial Problem

After fixing the protocol issues, score mismatches persisted. The FPGA scores did not match the Python simulation, even though predictions were often correct.

### Initial Diagnosis

**Suspected**: UART Receiver (`uart_rx.v`) was timing out or mistiming the "Stop Bit," causing it to miss the start of the next byte during rapid data transmission.

**Initial Fix Attempt**: Implemented an "Early Release" mechanism in `uart_rx.v` to make it ready for new data faster. While this improved stability, it did not solve the score mismatch.

### Diagnostic Tests Performed

A systematic series of tests was designed to isolate exactly where data corruption was occurring.

#### Test A: The "Black Image" Test (`send_black.py`)

**Action**: Sent an image consisting entirely of zeros.

**Purpose**: Since Input×Weight=0, the output depends only on the Biases.

**Result**: ✅ **PASSED**

**Conclusion**: This proved that:
- Biases are correct
- The Inference Logic (Math) is perfect
- The error was limited to Weights or Input Image data

#### Test B: Single Pixel at Index 0 (`send_1_pixel.py`)

**Action**: Sent an image where only the very first pixel (index 0) had a value.

**Purpose**: To check if the start of the transmission was aligned correctly.

**Result**: ✅ **PASSED**

**Conclusion**: The UART Router and Image Loader are correctly handling the start markers; there is no offset at the beginning of the stream.

#### Test C: Single Pixel at Index 100

**Action**: Sent an image where only the 100th pixel had a value.

**Purpose**: To check if data gets corrupted during the stream (mid-transmission).

**Result**: ❌ **FAILED**

**Conclusion**: This was the breakthrough. It proved that somewhere in the middle of the stream, a byte was being dropped, causing all subsequent data to shift index (Pixel 100 was being multiplied by Weight 101).

### The "Smoking Gun" & Final Diagnosis

**Observation**: Even though scores were wrong, the FPGA often predicted the correct digit (e.g., both Python and FPGA said "7"). This meant the network was working, but using "wrong" or "shifted" data.

**Root Cause**: **Throughput Density / Lack of Flow Control**

The PC was sending data (13,000 bytes for weights, 784 for images) back-to-back at full speed. The FPGA takes a few clock cycles to write a received byte into BRAM. During that tiny write window, the FPGA was "deaf" to the UART line, causing it to miss the Start Bit of the very next byte sent by the PC.

**Result**: A byte was dropped, and every subsequent weight/pixel shifted to the left, corrupting the matrix multiplication.

### The Solution

**Software-Side Flow Control**: Modified the Python scripts (`send_weights.py` and `compare_python_vs_fpga.py`).

**The Fix**: 
- Split the data transmission into small chunks (64 bytes)
- Added a tiny sleep command (`time.sleep(0.002)`) between chunks

**Outcome**: This gave the FPGA "breathing room" to finish writing to memory before the next chunk arrived.

### Final Result (Session 2)

- **Score Mismatch**: 0
- **Accuracy**: 100% Match
- **Status**: ✅ **Bit-Perfect Hardware Acceleration Achieved**

The FPGA output now matches the Python reference simulation exactly.

---

## Test Summary

| Test Script | Purpose | Result | Key Finding |
|------------|---------|--------|-------------|
| `generate_test_vectors.py` | Generate gold standard test vectors | ✅ Pass | Created reference data for verification |
| `tb_inference.v` | Verify inference logic in isolation | ✅ Pass (100%) | Computation logic is correct |
| `compare_python_vs_fpga.py` | End-to-end integration test | ⚠️ Partial | Correct predictions, wrong scores → alignment issue |
| `send_black.py` | Test with zero input (bias-only) | ✅ Pass | Biases and math logic correct |
| `send_1_pixel.py` (index 0) | Test start alignment | ✅ Pass | No offset at stream start |
| `send_1_pixel.py` (index 100) | Test mid-stream corruption | ❌ Fail | Byte dropping detected |
| `compare_python_vs_fpga.py` (after fixes) | Final verification | ✅ Pass (100%) | Bit-perfect match achieved |

---

## Code Changes Summary

### Modules Modified

#### 1. `uart_router.v`
- **Change**: Added byte counter to prevent false marker detection
- **Logic**: Ignore marker detection until expected byte count reached
- **Impact**: Prevents premature termination of data transfers

#### 2. `uart_tx.v`
- **Change**: Converted `busy` signal from register to combinatorial logic
- **Logic**: Instant assertion when send command issued
- **Impact**: Eliminates race conditions in transmission

#### 3. `image_loader.v`
- **Change**: Added `first_byte_flag` to drop first received byte
- **Logic**: Skip the leftover marker byte from router
- **Impact**: Corrects 1-byte pixel alignment shift

#### 4. `send_weights.py` (Python)
- **Change**: Implemented chunked transmission with delays
- **Logic**: Send 64-byte chunks with 2ms sleep between chunks
- **Impact**: Provides flow control for BRAM write operations

#### 5. `compare_python_vs_fpga.py` (Python)
- **Change**: Implemented chunked transmission with delays
- **Logic**: Send 64-byte chunks with 2ms sleep between chunks
- **Impact**: Prevents byte dropping during image transmission

### Files Affected

**Verilog Modules:**
- `2_ukryte/inference/rtl/uart_router.v`
- `2_ukryte/inference/rtl/uart_tx.v`
- `2_ukryte/inference/rtl/image_loader.v`

**Python Scripts:**
- `2_ukryte/utils/send_weights.py`
- `2_ukryte/testing/compare_python_vs_fpga.py`

---

## Lessons Learned

### 1. Binary Safety in Protocol Design
- **Issue**: Protocol markers can appear randomly in data streams
- **Solution**: Use byte counters or length-based termination instead of pattern matching
- **Best Practice**: Always validate that protocol markers cannot appear in legitimate data, or use length-based protocols

### 2. Timing and Race Conditions
- **Issue**: Register-based status signals create timing windows for race conditions
- **Solution**: Use combinatorial logic for critical control signals when instant response is needed
- **Best Practice**: Consider timing implications of register vs. wire for status signals

### 3. Data Alignment in Multi-Stage Pipelines
- **Issue**: Protocol markers consumed by one stage can leak into data for next stage
- **Solution**: Explicitly track and filter boundary bytes between pipeline stages
- **Best Practice**: Document byte consumption at each pipeline stage

### 4. Throughput vs. Processing Speed
- **Issue**: High-speed data transmission can exceed processing capability
- **Solution**: Implement flow control (chunking + delays) on the sender side
- **Best Practice**: Always consider processing time when designing communication protocols

### 5. Hardware Update Requirements
- **Issue**: Verilog code changes don't automatically update FPGA hardware
- **Solution**: Must run full synthesis, implementation, and bitstream generation
- **Best Practice**: Always verify bitstream is updated after code changes

### 6. Systematic Debugging Approach
- **Key Insight**: Isolated tests (black image, single pixels) were crucial for pinpointing the exact failure point
- **Best Practice**: Design targeted tests that isolate specific components or behaviors

### 7. Correct Predictions ≠ Correct Implementation
- **Key Insight**: The network could predict correctly even with shifted data, but scores revealed the problem
- **Best Practice**: Always verify raw outputs, not just final predictions

---

## Final Status

✅ **All Issues Resolved**

- **Inference Logic**: Verified 100% correct via testbench
- **Protocol Safety**: Binary-safe UART routing implemented
- **Timing Issues**: Race conditions eliminated
- **Data Alignment**: Pixel alignment corrected
- **Flow Control**: Chunked transmission prevents byte dropping
- **Hardware Verification**: 100% bit-perfect match with Python simulation

**Test Results**: 100/100 images match exactly (100% accuracy, 0 mismatches)

The FPGA neural network implementation is now fully functional and verified.

