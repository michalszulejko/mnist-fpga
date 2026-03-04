/*
================================================================================
Image Loader - Fixed Alignment
================================================================================
Update: Added `first_byte_flag` to drop the 0x66 protocol marker that the 
router passes through at the start of transmission.
This ensures Pixel 0 lands in Address 0, not Address 1.
================================================================================
*/

module image_loader (
    input wire clk,
    input wire rst,
    input wire weights_loaded,        // Only accept images after weights loaded
    
    // Routed UART interface (from uart_router)
    input wire [7:0] rx_data,         // Routed RX data
    input wire rx_ready,              // Routed RX ready signal
    
    // Image RAM write interface
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded
);

    // Protocol: Data bytes only (markers handled by uart_router)
    // End marker detection: 0x66 0xBB after 784 bytes
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    // States
    localparam STATE_RECEIVING = 2'd0;
    localparam STATE_DONE      = 2'd1;

    reg [1:0] state;
    reg [9:0] byte_count;
    reg [7:0] prev_byte;
    reg first_byte_flag;  // <--- NEW: Flag to drop the start marker

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_RECEIVING;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_count <= 0;
            prev_byte <= 0;
            image_loaded <= 0;
            first_byte_flag <= 1; // <--- Initialize to 1
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            // Only process if weights are loaded
            if (weights_loaded) begin
                case (state)
                    STATE_RECEIVING: begin
                        if (rx_ready) begin
                            // === FIX START: Drop the first byte (0x66 marker) ===
                            if (first_byte_flag) begin
                                first_byte_flag <= 0;
                                // Do NOT increment byte_count or write anything.
                                // We just consume the marker and wait for real data.
                            end 
                            else begin
                                // === Normal Processing ===
                                
                                // First, store the previous byte if we have data pending
                                if (byte_count > 0 && byte_count <= IMG_SIZE) begin
                                    wr_addr <= byte_count - 1;
                                    wr_data <= prev_byte;
                                    wr_en <= 1;
                                end
                                
                                // Check for end marker after storing
                                // We check byte_count >= IMG_SIZE to ensure binary safety
                                if (byte_count >= IMG_SIZE && prev_byte == IMG_END1 && rx_data == IMG_END2) begin
                                    state <= STATE_DONE;
                                    image_loaded <= 1;
                                end else begin
                                    // Advance counter and store current byte for next iteration
                                    byte_count <= byte_count + 1;
                                    prev_byte <= rx_data;
                                end
                            end
                            // === FIX END ===
                        end
                    end
                    
                    STATE_DONE: begin
                        // Go back to waiting for next image
                        state <= STATE_RECEIVING;
                        byte_count <= 0;
                        prev_byte <= 0;
                        first_byte_flag <= 1; // <--- Reset flag for next image
                    end
                endcase
            end
        end
    end

endmodule