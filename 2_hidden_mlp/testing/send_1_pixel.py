import serial
import time
import struct
import numpy as np
import os
import sys

# Constants
COM_PORT = 'COM7'  # Update if needed
BAUD_RATE = 115200
SHIFT1 = 7
SHIFT2 = 7

def load_weights():
    bin_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "bin")
    L1_W = np.fromfile(os.path.join(bin_dir, "L1_weights.bin"), dtype=np.int8).reshape(16, 784)
    L1_B = np.fromfile(os.path.join(bin_dir, "L1_biases.bin"), dtype=np.int32)
    L2_W = np.fromfile(os.path.join(bin_dir, "L2_weights.bin"), dtype=np.int8).reshape(16, 16)
    L2_B = np.fromfile(os.path.join(bin_dir, "L2_biases.bin"), dtype=np.int32)
    L3_W = np.fromfile(os.path.join(bin_dir, "L3_weights.bin"), dtype=np.int8).reshape(10, 16)
    L3_B = np.fromfile(os.path.join(bin_dir, "L3_biases.bin"), dtype=np.int32)
    return L1_W, L1_B, L2_W, L2_B, L3_W, L3_B

def run_simulation(pixel_val, pixel_index, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B):
    # Construct input: All zeros except one pixel
    img = np.zeros(784, dtype=np.int8)
    if pixel_index >= 0:
        img[pixel_index] = pixel_val

    # Layer 1
    L1_out = np.zeros(16, dtype=np.int32)
    for n in range(16):
        # Accumulate: Bias + (Weight * Pixel)
        acc = int(L1_B[n]) + int(img[pixel_index]) * int(L1_W[n, pixel_index])
        L1_out[n] = max(0, acc >> SHIFT1)

    # Layer 2
    L2_out = np.zeros(16, dtype=np.int32)
    for n in range(16):
        acc = np.int64(L2_B[n])
        for i in range(16):
            acc += np.int64(L1_out[i]) * np.int64(L2_W[n, i])
        L2_out[n] = max(0, np.int32(acc) >> SHIFT2)

    # Layer 3
    L3_out = np.zeros(10, dtype=np.int32)
    for c in range(10):
        acc = np.int64(L3_B[c])
        for i in range(16):
            acc += np.int64(L2_out[i]) * np.int64(L3_W[c, i])
        L3_out[c] = np.int32(acc)
        
    return L3_out

def main():
    L1_W, L1_B, L2_W, L2_B, L3_W, L3_B = load_weights()
    
    # Value to set the pixel to
    TEST_VAL = 100
    
    print(f"Testing Single Pixel at Index 0 with value {TEST_VAL}...")
    
    # 1. Calculate Expected Scores if Pixel 0 is handled correctly
    expected_0 = run_simulation(TEST_VAL, 100, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B)
    
    # 2. Calculate Expected Scores if Pixel 0 is shifted to Index 1
    expected_1 = run_simulation(TEST_VAL, 1, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B)
    
    # 3. Calculate Expected Scores if Pixel 0 is ignored (Black Image)
    expected_black = run_simulation(0, 0, L1_W, L1_B, L2_W, L2_B, L3_W, L3_B)

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1)
        ser.reset_input_buffer()

        # Send Image: Pixel 0 = 100, others = 0
        img_data = bytearray(784)
        img_data[0] = TEST_VAL 
        
        ser.write(bytes([0xBB, 0x66]))
        ser.write(img_data)
        ser.write(bytes([0x66, 0xBB]))
        
        time.sleep(0.1)

        ser.write(bytes([0xCD]))
        resp = ser.read(40)
        
        if len(resp) == 40:
            fpga_scores = np.array(struct.unpack('<10i', resp), dtype=np.int32)
            print("\nFPGA RESULT:")
            print(fpga_scores)
            
            print("\n--- DIAGNOSIS ---")
            if np.array_equal(fpga_scores, expected_0):
                print("SUCCESS! FPGA read the pixel at Index 0 correctly.")
            elif np.array_equal(fpga_scores, expected_1):
                print("FAILURE: FPGA shifted the pixel to Index 1.")
                print("FIX: You are dropping one byte too many in image_loader.v")
            elif np.array_equal(fpga_scores, expected_black):
                print("FAILURE: FPGA ignored the pixel (read as 0).")
                print("FIX: You might be overwriting or dropping the first byte completely.")
            else:
                print("FAILURE: FPGA result matches none of the standard error cases.")
                print(f"Diff from Expected 0: {fpga_scores - expected_0}")
                
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()