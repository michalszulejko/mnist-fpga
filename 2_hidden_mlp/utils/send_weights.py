"""
Send neural network weights to FPGA via UART.

Model: Two-Hidden-Layer Neural Network for MNIST
  - Layer 1: 784 -> 16 (12,544 weights + 64 biases)
  - Layer 2: 16 -> 16 (256 weights + 64 biases)
  - Layer 3: 16 -> 10 (160 weights + 40 biases)
  - Total: 13,128 bytes (12,960 weights + 168 biases)

Protocol:
  - Start marker: 0xAA 0x55
  - Data bytes: all weight/bias files concatenated
  - End marker: 0x55 0xAA

Configuration:
  - Port: COM3 (hardcoded)
  - Baud rate: 115200 (hardcoded)
  - Binary files: ../outputs/bin/

Usage:
  python send_weights.py
"""

import serial
import os
import sys
import time

# Configuration
PORT = "COM7"           # Serial port
BAUD_RATE = 115200      # Baud rate
BIN_DIR = "../outputs/bin"  # Directory with binary files

# Protocol markers
START_MARKER = bytes([0xAA, 0x55])
END_MARKER = bytes([0x55, 0xAA])

# Order of files to send (must match FPGA memory layout expectations)
FILES_TO_SEND = [
    "L1_weights.bin",
    "L1_biases.bin",
    "L2_weights.bin",
    "L2_biases.bin",
    "L3_weights.bin",
    "L3_biases.bin",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, BIN_DIR)
    
    # Collect all data
    all_data = bytearray()
    
    print("Two-Hidden-Layer Neural Network - Weight Upload")
    print("=" * 50)
    print("Loading binary files...")
    print("-" * 50)
    
    for filename in FILES_TO_SEND:
        filepath = os.path.join(bin_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"ERROR: {filename} not found!")
            print(f"Looking in: {bin_dir}")
            print("\nMake sure you've run the training script first:")
            print("  python training/siec_2_ukryte.py")
            sys.exit(1)
        
        with open(filepath, "rb") as f:
            file_data = f.read()
        
        all_data.extend(file_data)
        print(f"  {filename:20} : {len(file_data):6} bytes")
    
    print("-" * 50)
    print(f"  Total data          : {len(all_data):6} bytes")
    
    # Verify expected size
    expected_size = 13128  # 12,960 weights + 168 biases
    if len(all_data) != expected_size:
        print(f"\n⚠ WARNING: Expected {expected_size} bytes, got {len(all_data)} bytes!")
        print("  This may indicate incorrect model parameters.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    print()
    
    # Check for accidental marker bytes in data
    # (This would cause early termination - very unlikely but let's check)
    marker_check = bytes(all_data)
    if END_MARKER in marker_check:
        print("WARNING: End marker sequence (0x55 0xAA) found in data!")
        print("         This may cause early termination.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Open serial port
    print(f"Serial port:        {PORT}")
    print(f"Baud rate:          {BAUD_RATE}")
    print(f"Binary directory:   {bin_dir}")
    print("=" * 50)
    print()
    print(f"Opening {PORT} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
    except serial.SerialException as e:
        print(f"\nERROR: Could not open serial port: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if the correct COM port is specified")
        print("  2. Make sure no other program is using the port")
        print("  3. Verify the FPGA is connected and powered on")
        print("  4. On Windows, check Device Manager for COM port number")
        sys.exit(1)
    
    # Wait a moment for FPGA to be ready
    print("Waiting for FPGA to be ready...")
    time.sleep(0.5)
    
    # Calculate expected transmission time
    total_bytes = len(START_MARKER) + len(all_data) + len(END_MARKER)
    expected_time = (total_bytes * 10) / BAUD_RATE  # 10 bits per byte (8 data + start + stop)
    
    print()
    print(f"Sending {total_bytes} bytes...")
    print(f"Expected time: {expected_time:.2f} seconds")
    print()
    
    start_time = time.time()
    
    # Send start marker
    print("Sending start marker (0xAA 0x55)...")
    ser.write(START_MARKER)
    ser.flush()
    
# Send data with progress indicator
    print("Sending data", end="", flush=True)
    chunk_size = 64  # Reduced chunk size for safety
    bytes_sent = 0
    
    for i in range(0, len(all_data), chunk_size):
        chunk = all_data[i:i + chunk_size]
        ser.write(chunk)
        bytes_sent += len(chunk)
        
        # Progress indicator
        progress = bytes_sent / len(all_data) * 100
        print(f"\rSending data: {bytes_sent:6}/{len(all_data)} bytes ({progress:5.1f}%)", end="", flush=True)
        
        # === CRITICAL FIX: Give FPGA time to write to BRAM ===
        time.sleep(0.005) # 5ms delay between chunks
        # ====================================================
        
    ser.flush()
    print()
    
    # Send end marker
    print("Sending end marker (0x55 0xAA)...")
    ser.write(END_MARKER)
    ser.flush()
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 50)
    print("TRANSFER COMPLETE!")
    print(f"  Bytes sent: {total_bytes}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Actual rate: {total_bytes * 8 / elapsed_time:.0f} bits/sec")
    print("=" * 50)
    print()
    print("Check FPGA LEDs:")
    print("  - led[3] ON = Transfer successful")
    print("  - led[4] ON = Error (overflow)")
    print()
    
    ser.close()


if __name__ == "__main__":
    main()


