/*
================================================================================
RAM Modules
================================================================================
Block RAM definitions for image storage, predicted digit storage, and scores.
================================================================================
*/

// =============================================================================
// Image RAM (784 bytes)
// =============================================================================
module image_ram (
    input wire clk,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data
);

    (* ram_style = "block" *) reg [7:0] ram [0:783];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


// =============================================================================
// Scores RAM - Stores all 10 class scores for accuracy analysis
// =============================================================================
// Memory Layout:
//   Address 0-3:   Class 0 score (32-bit signed, little-endian)
//   Address 4-7:   Class 1 score
//   Address 8-11:  Class 2 score
//   Address 12-15: Class 3 score
//   Address 16-19: Class 4 score
//   Address 20-23: Class 5 score
//   Address 24-27: Class 6 score
//   Address 28-31: Class 7 score
//   Address 32-35: Class 8 score
//   Address 36-39: Class 9 score
//
// Total: 40 bytes (10 scores × 4 bytes each)
//
// This BRAM stores all class scores when inference completes.
// Scores are written in little-endian format for easy reading via UART.
// =============================================================================
module scores_ram (
    input wire clk,
    input wire wr_en,                    // Write enable (pulse when inference_done)
    input wire signed [31:0] score_0,    // Class 0 score
    input wire signed [31:0] score_1,    // Class 1 score
    input wire signed [31:0] score_2,    // Class 2 score
    input wire signed [31:0] score_3,    // Class 3 score
    input wire signed [31:0] score_4,    // Class 4 score
    input wire signed [31:0] score_5,    // Class 5 score
    input wire signed [31:0] score_6,    // Class 6 score
    input wire signed [31:0] score_7,    // Class 7 score
    input wire signed [31:0] score_8,    // Class 8 score
    input wire signed [31:0] score_9,    // Class 9 score
    input wire [5:0] rd_addr,            // Read address (0-39)
    output reg [7:0] rd_data             // Read data (1 byte)
);

    // 40-byte BRAM (10 scores × 4 bytes each)
    (* ram_style = "block" *) reg [7:0] ram [0:39];
    
    // Synchronous write - write all 10 scores at once (little-endian)
    always @(posedge clk) begin
        if (wr_en) begin
            // Class 0 (addresses 0-3)
            ram[0]  <= score_0[7:0];
            ram[1]  <= score_0[15:8];
            ram[2]  <= score_0[23:16];
            ram[3]  <= score_0[31:24];
            
            // Class 1 (addresses 4-7)
            ram[4]  <= score_1[7:0];
            ram[5]  <= score_1[15:8];
            ram[6]  <= score_1[23:16];
            ram[7]  <= score_1[31:24];
            
            // Class 2 (addresses 8-11)
            ram[8]  <= score_2[7:0];
            ram[9]  <= score_2[15:8];
            ram[10] <= score_2[23:16];
            ram[11] <= score_2[31:24];
            
            // Class 3 (addresses 12-15)
            ram[12] <= score_3[7:0];
            ram[13] <= score_3[15:8];
            ram[14] <= score_3[23:16];
            ram[15] <= score_3[31:24];
            
            // Class 4 (addresses 16-19)
            ram[16] <= score_4[7:0];
            ram[17] <= score_4[15:8];
            ram[18] <= score_4[23:16];
            ram[19] <= score_4[31:24];
            
            // Class 5 (addresses 20-23)
            ram[20] <= score_5[7:0];
            ram[21] <= score_5[15:8];
            ram[22] <= score_5[23:16];
            ram[23] <= score_5[31:24];
            
            // Class 6 (addresses 24-27)
            ram[24] <= score_6[7:0];
            ram[25] <= score_6[15:8];
            ram[26] <= score_6[23:16];
            ram[27] <= score_6[31:24];
            
            // Class 7 (addresses 28-31)
            ram[28] <= score_7[7:0];
            ram[29] <= score_7[15:8];
            ram[30] <= score_7[23:16];
            ram[31] <= score_7[31:24];
            
            // Class 8 (addresses 32-35)
            ram[32] <= score_8[7:0];
            ram[33] <= score_8[15:8];
            ram[34] <= score_8[23:16];
            ram[35] <= score_8[31:24];
            
            // Class 9 (addresses 36-39)
            ram[36] <= score_9[7:0];
            ram[37] <= score_9[15:8];
            ram[38] <= score_9[23:16];
            ram[39] <= score_9[31:24];
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule


// =============================================================================
// Predicted Digit RAM - Stores the predicted digit at a fixed address
// =============================================================================
// Memory Layout:
//   Address 0: Predicted digit (4-bit value, stored as 8-bit byte)
//
// This BRAM is separate from weight/bias/image memories.
// When inference completes, the predicted digit (0-9) is written to address 0.
// =============================================================================
module predicted_digit_ram (
    input wire clk,
    input wire wr_en,           // Write enable (pulse when inference_done)
    input wire [3:0] wr_data,   // Predicted digit value (0-9)
    input wire rd_addr,         // Read address (always 0 for now)
    output reg [7:0] rd_data    // Read data (digit in lower 4 bits)
);

    // Single byte BRAM
    (* ram_style = "block" *) reg [7:0] ram [0:0];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[0] <= {4'b0, wr_data};  // Store 4-bit digit in lower nibble
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end

endmodule

