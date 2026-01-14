"""
LeNet-5 Weight Sender - Wrapper for send_weights_lenet.py

This script redirects to send_weights_lenet.py which handles
the full LeNet-5 architecture:
  - Conv1: 6 filters, 5x5 (150 bytes)
  - Conv2: 16 filters, 5x5 (2400 bytes)
  - FC1: 400 -> 120 (48000 bytes)
  - FC2: 120 -> 84 (10080 bytes)
  - FC3: 84 -> 10 (840 bytes)
  - Tanh LUT: 256 bytes

Usage:
  python send_weights.py [--port PORT] [--baud BAUD]
"""

# Redirect to the proper LeNet implementation
from send_weights_lenet import send_weights
import argparse
import sys

DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send LeNet-5 weights to FPGA')
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    sys.exit(send_weights(args.port, args.baud))
