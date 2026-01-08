/*
================================================================================
Predicted Digit RAM - Stores the predicted digit at a fixed address
================================================================================
Memory Layout:
  Address 0: Predicted digit (4-bit value, stored as 8-bit byte)
================================================================================
*/

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


