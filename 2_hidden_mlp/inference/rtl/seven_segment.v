/*
================================================================================
7-Segment Display Controller
================================================================================
Displays two digits:
  - Leftmost digit (an[3]): Stored predicted digit from BRAM
  - Rightmost digit (an[0]): Current real-time predicted digit
Uses time-multiplexing to alternate between digits at ~1kHz refresh rate.
================================================================================
*/

module seven_segment_display (
    input wire clk,
    input wire rst,
    input wire [3:0] digit_left,   // Leftmost digit (from memory)
    input wire [3:0] digit_right,  // Rightmost digit (predicted digit)
    output reg [6:0] seg,           // Segments a-g (active low)
    output reg [3:0] an             // Anodes (active low)
);
    
    // Refresh counter for multiplexing (100MHz / 50000 = 2kHz, 1kHz per digit)
    localparam REFRESH_COUNT = 50000;
    reg [15:0] refresh_counter;
    reg display_select;  // 0 = display left, 1 = display right
    
    // 7-segment encoding function
    function [6:0] seg_encode;
        input [3:0] digit;
        begin
            case (digit)
                4'd0: seg_encode = 7'b1000000;
                4'd1: seg_encode = 7'b1111001;
                4'd2: seg_encode = 7'b0100100;
                4'd3: seg_encode = 7'b0110000;
                4'd4: seg_encode = 7'b0011001;
                4'd5: seg_encode = 7'b0010010;
                4'd6: seg_encode = 7'b0000010;
                4'd7: seg_encode = 7'b1111000;
                4'd8: seg_encode = 7'b0000000;
                4'd9: seg_encode = 7'b0010000;
                default: seg_encode = 7'b0111111;  // Dash for invalid
            endcase
        end
    endfunction

    // Refresh counter for multiplexing
    always @(posedge clk) begin
        if (rst) begin
            refresh_counter <= 0;
            display_select <= 0;
        end else begin
            if (refresh_counter >= REFRESH_COUNT - 1) begin
                refresh_counter <= 0;
                display_select <= ~display_select;  // Toggle between displays
            end else begin
                refresh_counter <= refresh_counter + 1;
            end
        end
    end

    // Display multiplexer
    always @(posedge clk) begin
        if (rst) begin
            an <= 4'b1111;  // All off
            seg <= 7'b1111111;
        end else begin
            if (display_select == 0) begin
                // Display leftmost digit (an[3])
                an <= 4'b0111;  // an[3] active (leftmost), others off
                seg <= seg_encode(digit_left);
            end else begin
                // Display rightmost digit (an[0])
                an <= 4'b1110;  // an[0] active (rightmost), others off
                seg <= seg_encode(digit_right);
            end
        end
    end

endmodule

