/*
================================================================================
LeNet-5 Working Memory Module
================================================================================
Buffer sizes for LeNet-5:
- Buffer A: 6*28*28 = 4704 bytes (Conv1 output) or 16*10*10 = 1600 (Conv2 output)
- Buffer B: 6*14*14 = 1176 bytes (Pool1 output) or 120 bytes (FC1 output)
- Buffer C: 16*5*5 = 400 bytes (Pool2 output) or 84 bytes (FC2 output)
================================================================================
*/

module ram_cnn (
    input clk,

    // Buffer A: Holds Conv outputs
    // Max: 6*28*28 = 4704 bytes (Conv1 output)
    input [12:0] buf_a_addr,
    input [7:0] buf_a_wr_data,
    input buf_a_wr_en,
    output wire [7:0] buf_a_rd_data,

    // Buffer B: Holds Pool1 output (6*14*14=1176) or FC1 output (120)
    input [10:0] buf_b_addr,
    input [7:0] buf_b_wr_data,
    input buf_b_wr_en,
    output wire [7:0] buf_b_rd_data,

    // Buffer C: Holds Pool2 output (400) or FC2 output (84)
    input [8:0] buf_c_addr,
    input [7:0] buf_c_wr_data,
    input buf_c_wr_en,
    output wire [7:0] buf_c_rd_data
);

    // Force Distributed RAM (LUTRAM) for 0-cycle read latency
    (* ram_style = "distributed" *) reg [7:0] buffer_a [0:4703];   // 6*28*28 = 4704
    (* ram_style = "distributed" *) reg [7:0] buffer_b [0:1175];   // 6*14*14 = 1176
    (* ram_style = "distributed" *) reg [7:0] buffer_c [0:399];    // 16*5*5 = 400

    // Buffer A: Synchronous Write, Asynchronous Read
    always @(posedge clk) begin
        if (buf_a_wr_en) buffer_a[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buffer_a[buf_a_addr];

    // Buffer B: Synchronous Write, Asynchronous Read
    always @(posedge clk) begin
        if (buf_b_wr_en) buffer_b[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buffer_b[buf_b_addr];

    // Buffer C: Synchronous Write, Asynchronous Read
    always @(posedge clk) begin
        if (buf_c_wr_en) buffer_c[buf_c_addr] <= buf_c_wr_data;
    end
    assign buf_c_rd_data = buffer_c[buf_c_addr];

endmodule
