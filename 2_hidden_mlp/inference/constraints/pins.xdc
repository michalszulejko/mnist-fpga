/*
================================================================================
Basys3 Pin Constraints for Two-Hidden-Layer Neural Network FPGA Implementation
================================================================================

Copy the content below (starting from "## Clock") into your Vivado XDC file.
These constraints are for the Digilent Basys3 board (Artix-7 XC7A35T-1CPG236C).

Model: Two-Hidden-Layer Neural Network for MNIST
  - Layer 1: 784 -> 16 neurons (ReLU)
  - Layer 2: 16 -> 16 neurons (ReLU)
  - Layer 3: 16 -> 10 outputs (softmax)
  - Total Parameters: 13,128 bytes (12,960 weights + 168 biases)

================================================================================
*/

## Clock - 100 MHz oscillator
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} -add [get_ports clk]

## Reset - Center Button (active high)
set_property PACKAGE_PIN U18 [get_ports rst]
set_property IOSTANDARD LVCMOS33 [get_ports rst]

## UART (USB-UART bridge)
set_property PACKAGE_PIN B18 [get_ports rx]
set_property IOSTANDARD LVCMOS33 [get_ports rx]

set_property PACKAGE_PIN A18 [get_ports tx]
set_property IOSTANDARD LVCMOS33 [get_ports tx]

## LEDs
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property PACKAGE_PIN W18 [get_ports {led[4]}]
set_property PACKAGE_PIN U15 [get_ports {led[5]}]
set_property PACKAGE_PIN U14 [get_ports {led[6]}]
set_property PACKAGE_PIN V14 [get_ports {led[7]}]
set_property PACKAGE_PIN V13 [get_ports {led[8]}]
set_property PACKAGE_PIN V3 [get_ports {led[9]}]
set_property PACKAGE_PIN W3 [get_ports {led[10]}]
set_property PACKAGE_PIN U3 [get_ports {led[11]}]
set_property PACKAGE_PIN P3 [get_ports {led[12]}]
set_property PACKAGE_PIN N3 [get_ports {led[13]}]
set_property PACKAGE_PIN P1 [get_ports {led[14]}]
set_property PACKAGE_PIN L1 [get_ports {led[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

## 7-Segment Display - Segments (active low)
set_property PACKAGE_PIN W7 [get_ports {seg[0]}]
set_property PACKAGE_PIN W6 [get_ports {seg[1]}]
set_property PACKAGE_PIN U8 [get_ports {seg[2]}]
set_property PACKAGE_PIN V8 [get_ports {seg[3]}]
set_property PACKAGE_PIN U5 [get_ports {seg[4]}]
set_property PACKAGE_PIN V5 [get_ports {seg[5]}]
set_property PACKAGE_PIN U7 [get_ports {seg[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg[*]}]

## 7-Segment Display - Anodes (active low)
set_property PACKAGE_PIN U2 [get_ports {an[0]}]
set_property PACKAGE_PIN U4 [get_ports {an[1]}]
set_property PACKAGE_PIN V4 [get_ports {an[2]}]
set_property PACKAGE_PIN W4 [get_ports {an[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[*]}]

## Switches (optional - for debug/control)
set_property PACKAGE_PIN V17 [get_ports {sw[0]}]
set_property PACKAGE_PIN V16 [get_ports {sw[1]}]
set_property PACKAGE_PIN W16 [get_ports {sw[2]}]
set_property PACKAGE_PIN W17 [get_ports {sw[3]}]
set_property PACKAGE_PIN W15 [get_ports {sw[4]}]
set_property PACKAGE_PIN V15 [get_ports {sw[5]}]
set_property PACKAGE_PIN W14 [get_ports {sw[6]}]
set_property PACKAGE_PIN W13 [get_ports {sw[7]}]
set_property PACKAGE_PIN V2 [get_ports {sw[8]}]
set_property PACKAGE_PIN T3 [get_ports {sw[9]}]
set_property PACKAGE_PIN T2 [get_ports {sw[10]}]
set_property PACKAGE_PIN R3 [get_ports {sw[11]}]
set_property PACKAGE_PIN W2 [get_ports {sw[12]}]
set_property PACKAGE_PIN U1 [get_ports {sw[13]}]
set_property PACKAGE_PIN T1 [get_ports {sw[14]}]
set_property PACKAGE_PIN R2 [get_ports {sw[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[*]}]

## Buttons
set_property PACKAGE_PIN T18 [get_ports btnU]
set_property PACKAGE_PIN W19 [get_ports btnL]
set_property PACKAGE_PIN T17 [get_ports btnR]
set_property PACKAGE_PIN U17 [get_ports btnD]
set_property IOSTANDARD LVCMOS33 [get_ports btnU]
set_property IOSTANDARD LVCMOS33 [get_ports btnL]
set_property IOSTANDARD LVCMOS33 [get_ports btnR]
set_property IOSTANDARD LVCMOS33 [get_ports btnD]

## Configuration options
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]

/*
================================================================================
LED Mapping for Two-Hidden-Layer Neural Network:
================================================================================

During Weight Loading:
  led[0]     - Blinks when receiving bytes
  led[1]     - Waiting for start marker
  led[2]     - Receiving data
  led[3]     - Transfer complete (success)
  led[4]     - Error (overflow)
  led[15:8]  - Progress indicator (lower 8 bits of address)

During Inference:
  led[3:0]   - Predicted digit (0-9 in binary)
  led[4]     - Inference busy
  led[5]     - Inference done
  led[6]     - Image loaded
  led[7]     - Weights loaded
  led[15:8]  - Progress from weight loader

7-Segment Display:
  Shows predicted digit (0-9)

Buttons:
  btnC (Center) - Reset (directly wired) or Start Inference
  btnU (Up)     - Reserved
  btnD (Down)   - Reserved
  btnL (Left)   - Reserved
  btnR (Right)  - Reserved

================================================================================
Model Architecture:
================================================================================

Layer 1: Input -> Hidden1
  - Input: 784 pixels
  - Output: 16 neurons
  - Weights: 12,544 bytes (8-bit signed)
  - Biases: 64 bytes (16 x 32-bit signed)
  - Activation: ReLU

Layer 2: Hidden1 -> Hidden2
  - Input: 16 neurons
  - Output: 16 neurons
  - Weights: 256 bytes (8-bit signed)
  - Biases: 64 bytes (16 x 32-bit signed)
  - Activation: ReLU

Layer 3: Hidden2 -> Output
  - Input: 16 neurons
  - Output: 10 classes
  - Weights: 160 bytes (8-bit signed)
  - Biases: 40 bytes (10 x 32-bit signed)
  - Activation: Softmax (argmax for prediction)

Total Parameters: 13,128 bytes

================================================================================
*/






