/*
================================================================================
LeNet-5 Fully Connected Weights and Biases RAM
================================================================================
FC1: 120*400 = 48000 weights + 120 biases
FC2: 84*120 = 10080 weights + 84 biases
FC3: 10*84 = 840 weights + 10 biases
Total: 58920 weights + 214 biases

NOTE: Due to large size, FC weights use BRAM instead of distributed RAM.
This adds 1-cycle read latency which the FSM must account for.
================================================================================
*/

// Combined FC Weights RAM (FC1 + FC2 + FC3)
// FC1: 0 - 47999 (48000 bytes)
// FC2: 48000 - 58079 (10080 bytes)
// FC3: 58080 - 58919 (840 bytes)
module fc_weights_ram (
    input wire clk,
    input wire [15:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [15:0] rd_addr,
    output reg [7:0] rd_data
);
    // Use BRAM for large FC weights (58920 bytes)
    // BRAM has synchronous read, so FSM needs to handle 1-cycle latency
    (* ram_style = "block" *) reg [7:0] ram [0:58919];

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

    // Synchronous read (BRAM style)
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end
endmodule

// Combined FC Biases RAM (FC1 + FC2 + FC3)
// FC1: 0 - 119 (120 biases)
// FC2: 120 - 203 (84 biases)
// FC3: 204 - 213 (10 biases)
// Total: 214 biases (32-bit each)
module fc_biases_ram (
    input wire clk,
    input wire [7:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [7:0] rd_addr,
    output wire [31:0] rd_data
);
    // Use distributed RAM for biases (small enough)
    (* ram_style = "distributed" *) reg [31:0] ram [0:213];

    always @(posedge clk) begin
        if (wr_en) ram[wr_addr] <= wr_data;
    end

    // Asynchronous read
    assign rd_data = ram[rd_addr];
endmodule
