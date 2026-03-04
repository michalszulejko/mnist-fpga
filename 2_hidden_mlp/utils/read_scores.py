"""
Read Class Scores (Logits) from FPGA - 2-Hidden-Layer Neural Network

This script connects to the FPGA via UART and reads all 10 class scores
(logits) using the 0xCD protocol. These are the raw output values before
the argmax operation that determines the predicted digit.

Protocol:
  - Send: 0xCD (1 byte request)
  - Receive: 40 bytes (10 scores × 4 bytes each, little-endian int32)

Memory Layout:
  Bytes 0-3:   Class 0 score (int32, little-endian)
  Bytes 4-7:   Class 1 score
  Bytes 8-11:  Class 2 score
  Bytes 12-15: Class 3 score
  Bytes 16-19: Class 4 score
  Bytes 20-23: Class 5 score
  Bytes 24-27: Class 6 score
  Bytes 28-31: Class 7 score
  Bytes 32-35: Class 8 score
  Bytes 36-39: Class 9 score

Usage:
  python read_scores.py [options]
  
  Options:
    --port PORT       Serial COM port (default: COM3)
    --baud BAUD       Baud rate (default: 115200)
    --continuous      Continuously read scores (press Ctrl+C to stop)
    --softmax         Also compute and display softmax probabilities
    --wait SECONDS    Wait time between continuous reads (default: 1.0)

Examples:
  python read_scores.py
  python read_scores.py --port COM5
  python read_scores.py --continuous --wait 0.5
  python read_scores.py --softmax
"""

import serial
import time
import sys
import struct
import argparse
import numpy as np


# Protocol Constants
SCORES_READ_REQUEST = bytes([0xCD])
NUM_SCORES = 10
BYTES_PER_SCORE = 4
TOTAL_BYTES = NUM_SCORES * BYTES_PER_SCORE

# Default Configuration
DEFAULT_COM_PORT = 'COM3'
DEFAULT_BAUD_RATE = 115200
DEFAULT_WAIT_TIME = 1.0


def read_scores_from_fpga(ser):
    """
    Read all 10 class scores from FPGA via 0xCD protocol.
    
    Args:
        ser: Serial connection
    
    Returns:
        List of 10 signed int32 scores, or None on error
    """
    # Clear input buffer
    ser.reset_input_buffer()
    
    # Send request
    ser.write(SCORES_READ_REQUEST)
    
    # Read 40 bytes response
    response = ser.read(TOTAL_BYTES)
    
    if len(response) != TOTAL_BYTES:
        print(f"ERROR: Expected {TOTAL_BYTES} bytes, received {len(response)}")
        return None
    
    # Parse 40 bytes into 10 signed int32 values (little-endian)
    scores = []
    for i in range(NUM_SCORES):
        offset = i * BYTES_PER_SCORE
        score_bytes = response[offset:offset + BYTES_PER_SCORE]
        # '<i' means little-endian signed int32
        score = struct.unpack('<i', score_bytes)[0]
        scores.append(score)
    
    return scores


def compute_softmax(scores):
    """
    Compute softmax probabilities from raw scores.
    
    Args:
        scores: List of 10 raw scores (int32)
    
    Returns:
        NumPy array of 10 probabilities (float, sum to 1.0)
    """
    scores_np = np.array(scores, dtype=np.float64)
    
    # Subtract max for numerical stability
    scores_shifted = scores_np - np.max(scores_np)
    
    # Compute exp
    exp_scores = np.exp(scores_shifted)
    
    # Normalize
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities


def display_scores(scores, show_softmax=False):
    """
    Display scores in a readable format.
    
    Args:
        scores: List of 10 raw scores
        show_softmax: Whether to compute and show softmax probabilities
    """
    # Find max score and predicted digit
    max_score = max(scores)
    predicted_digit = scores.index(max_score)
    
    print("=" * 70)
    print("Class Scores (Logits) from FPGA")
    print("=" * 70)
    
    # Compute softmax if requested
    if show_softmax:
        probabilities = compute_softmax(scores)
    
    # Display each class
    for i in range(NUM_SCORES):
        score = scores[i]
        is_max = (i == predicted_digit)
        marker = " <-- MAX (Predicted)" if is_max else ""
        
        if show_softmax:
            prob = probabilities[i] * 100
            print(f"Class {i}: {score:12d} (raw score) | {prob:6.2f}% (softmax){marker}")
        else:
            print(f"Class {i}: {score:12d} (raw score){marker}")
    
    print("=" * 70)
    print(f"Predicted Digit: {predicted_digit}")
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Read class scores (logits) from FPGA via UART',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single read (default)
  python read_scores.py
  
  # Use different COM port
  python read_scores.py --port COM5
  
  # Continuous reading with 0.5s interval
  python read_scores.py --continuous --wait 0.5
  
  # Display softmax probabilities
  python read_scores.py --softmax
        """
    )
    
    parser.add_argument('--port', type=str, default=DEFAULT_COM_PORT,
                        help=f'Serial COM port (default: {DEFAULT_COM_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD_RATE,
                        help=f'Baud rate (default: {DEFAULT_BAUD_RATE})')
    parser.add_argument('--continuous', action='store_true',
                        help='Continuously read scores (Ctrl+C to stop)')
    parser.add_argument('--softmax', action='store_true',
                        help='Also compute and display softmax probabilities')
    parser.add_argument('--wait', type=float, default=DEFAULT_WAIT_TIME,
                        help=f'Wait time between continuous reads in seconds (default: {DEFAULT_WAIT_TIME})')
    
    args = parser.parse_args()
    
    # Connect to FPGA
    print(f"Connecting to FPGA on {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        time.sleep(0.5)  # Allow serial connection to stabilize
        print("✓ Connected to FPGA")
        print()
    except serial.SerialException as e:
        print(f"✗ Serial Error: {e}")
        print("Cannot proceed without FPGA connection.")
        sys.exit(1)
    
    try:
        if args.continuous:
            print("Continuous mode: Press Ctrl+C to stop")
            print()
            read_count = 0
            while True:
                read_count += 1
                print(f"\n--- Read #{read_count} ---")
                
                scores = read_scores_from_fpga(ser)
                
                if scores is None:
                    print("Failed to read scores from FPGA")
                else:
                    display_scores(scores, show_softmax=args.softmax)
                
                time.sleep(args.wait)
        else:
            # Single read
            scores = read_scores_from_fpga(ser)
            
            if scores is None:
                print("Failed to read scores from FPGA")
                sys.exit(1)
            else:
                display_scores(scores, show_softmax=args.softmax)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    finally:
        ser.close()
        print("\nSerial connection closed.")


if __name__ == "__main__":
    main()






