/*
================================================================================
Tanh Lookup Table Module for LeNet-5
================================================================================
256-entry LUT for tanh approximation
Input: 8-bit index (0-255, representing -128 to 127 after offset)
Output: 8-bit signed value (-127 to 127, representing tanh output)
================================================================================
*/

module tanh_lut (
    input wire clk,

    // Read interface (for inference)
    input wire [7:0] addr,
    output wire signed [7:0] data,

    // Debug read interface (for UART readback)
    input wire [7:0] dbg_addr,
    output wire [7:0] dbg_data,

    // Write interface (for initialization via UART)
    input wire [7:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en
);

    // 256-entry LUT stored in distributed RAM
    (* ram_style = "distributed" *) reg [7:0] lut [0:255];

    // Initialize with pre-computed tanh values
    // This will be overwritten by $readmemh or UART loading
    initial begin
        // Default initialization (linear approximation)
        // Proper values should be loaded from tanh_lut.mem
        $readmemh("tanh_lut.mem", lut);
    end

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) lut[wr_addr] <= wr_data;
    end

    // Asynchronous read for 0-cycle latency
    assign data = lut[addr];

    // Debug read port (asynchronous)
    assign dbg_data = lut[dbg_addr];

endmodule
