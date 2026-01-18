/*
================================================================================
LeNet-5 Convolution Weights and Biases RAM
================================================================================
Conv1: 6*1*5*5 = 150 weights + 6 biases
Conv2: 16*6*5*5 = 2400 weights + 16 biases
Total: 2550 weights + 22 biases
================================================================================
*/

// Combined Conv Weights RAM (L1 + L2)
// L1: 0 - 149 (150 bytes)
// L2: 150 - 2549 (2400 bytes)
module conv_weights_ram (
    input wire clk,
    input wire [11:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [11:0] rd_addr,
    output wire [7:0] rd_data,
    // Debug read port
    input wire [11:0] dbg_addr,
    output wire [7:0] dbg_data
);
    // Force Distributed RAM for Combinational Read
    (* ram_style = "distributed" *) reg [7:0] ram [0:2549];

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

    // Asynchronous read
    assign rd_data = ram[rd_addr];

    // Debug read port (asynchronous)
    assign dbg_data = ram[dbg_addr];
endmodule

// Combined Conv Biases RAM (L1 + L2)
// L1: 0-5 (6 biases)
// L2: 6-21 (16 biases)
// Total: 22 biases (32-bit each)
module conv_biases_ram (
    input wire clk,
    input wire [4:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [4:0] rd_addr,
    output wire [31:0] rd_data
);
    // Force Distributed RAM
    (* ram_style = "distributed" *) reg [31:0] ram [0:21];

    always @(posedge clk) begin
        if (wr_en) ram[wr_addr] <= wr_data;
    end

    // Asynchronous read
    assign rd_data = ram[rd_addr];
endmodule
