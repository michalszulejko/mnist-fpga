import serial
import time
import sys
import os
import argparse
import numpy as np

# Configuration
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"
MEM_DIR = "../outputs/mem"

# Protocol Markers (Must match uart_router.v)
START_MARKER = np.array([0xAA, 0x55], dtype=np.uint8)
END_MARKER   = np.array([0x55, 0xAA], dtype=np.uint8)

def load_tanh_lut(mem_path):
    """Load tanh LUT from .mem file and convert to binary."""
    lut_path = os.path.join(mem_path, "tanh_lut.mem")
    lut = np.zeros(256, dtype=np.uint8)

    try:
        with open(lut_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 256:
                    break
                val = int(line.strip(), 16) & 0xFF
                lut[i] = val
        print(f"  Loaded tanh_lut.mem ({lut.size} bytes)")
    except FileNotFoundError:
        print("Error: tanh_lut.mem not found!")
        raise

    return lut

def load_files(bin_path, mem_path):
    """Load weights from binary files into numpy arrays."""
    files = [
        ("conv1_weights.bin", "Conv1 Weights"),
        ("conv1_biases.bin",  "Conv1 Biases"),
        ("conv2_weights.bin", "Conv2 Weights"),
        ("conv2_biases.bin",  "Conv2 Biases"),
        ("fc1_weights.bin",   "FC1 Weights"),
        ("fc1_biases.bin",    "FC1 Biases"),
        ("fc2_weights.bin",   "FC2 Weights"),
        ("fc2_biases.bin",    "FC2 Biases"),
        ("fc3_weights.bin",   "FC3 Weights"),
        ("fc3_biases.bin",    "FC3 Biases"),
    ]

    data_chunks = []

    print(f"Loading weight files from {bin_path}...")

    for filename, label in files:
        path = os.path.join(bin_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}\nDid you run train_lenet.py?")

        chunk = np.fromfile(path, dtype=np.uint8)
        data_chunks.append(chunk)
        print(f"  {label:<20}: {chunk.size:>6} bytes")

    # Load Tanh LUT
    tanh_lut = load_tanh_lut(mem_path)
    data_chunks.append(tanh_lut)
    print(f"  {'Tanh LUT':<20}: {tanh_lut.size:>6} bytes")

    # Calculate total payload size (excluding markers)
    payload_size = sum(c.size for c in data_chunks)
    print(f"  {'Total Payload':<20}: {payload_size:>6} bytes")

    # Validation
    EXPECTED_SIZE = 62670  # LeNet-5 total weight package size
    if payload_size != EXPECTED_SIZE:
        print(f"\nWARNING: Payload size ({payload_size}) does not match FPGA expected size ({EXPECTED_SIZE})!")
        print("Check weight_loader.v 'TOTAL' parameter.")

    return data_chunks, payload_size

def send_weights(port, baud):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    mem_path = os.path.join(script_dir, MEM_DIR)

    try:
        chunks, payload_size = load_files(bin_path, mem_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Concatenate everything
    all_data = np.concatenate([START_MARKER] + chunks + [END_MARKER])
    total_packet_size = all_data.size

    print(f"\nTotal Transmission packet: {total_packet_size} bytes")

    # Connect to Serial
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Error opening port {port}: {e}")
        return 1

    print(f"Sending to {port} (SLOW & SAFE MODE)...")
    start_time = time.time()

    # --- SLOW MODE SETTINGS ---
    # Reduced chunk size and increased delay to prevent buffer overflow
    CHUNK_SIZE = 16
    DELAY = 0.020        # 20ms delay
    # --------------------------

    bytes_sent = 0
    raw_bytes = all_data.tobytes()

    for i in range(0, len(raw_bytes), CHUNK_SIZE):
        chunk = raw_bytes[i : i + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        bytes_sent += len(chunk)

        # Progress Bar
        progress = (bytes_sent / total_packet_size) * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}% ({bytes_sent}/{total_packet_size})")
        sys.stdout.flush()

        time.sleep(DELAY)

    print("\nDone.")
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s")

    ser.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    sys.exit(send_weights(args.port, args.baud))
