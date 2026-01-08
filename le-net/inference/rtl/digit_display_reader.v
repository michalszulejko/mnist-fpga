/*
================================================================================
Digit Display Reader - Reads digit from predicted_digit_ram
================================================================================
*/

module digit_display_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] digit_ram_data,  // Data from predicted_digit_ram
    output reg [3:0] display_digit    // 4-bit digit extracted from memory
);

    always @(posedge clk) begin
        if (rst) begin
            display_digit <= 0;
        end else begin
            // Extract lower 4 bits (digit is stored in lower nibble)
            display_digit <= digit_ram_data[3:0];
        end
    end

endmodule



