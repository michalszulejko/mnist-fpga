module image_loader (
    input wire clk,
    input wire rst,
    input wire weights_loaded,
    input wire [7:0] rx_data,
    input wire rx_ready,
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded,
    output reg [9:0] debug_rx_count
);
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    localparam STATE_RECEIVING = 2'd0;
    localparam STATE_DONE      = 2'd1;

    reg [1:0] state;
    reg [9:0] byte_count;
    reg [7:0] prev_byte;

    reg [7:0] rx_data_reg;
    reg rx_ready_reg;

    always @(posedge clk) begin
        rx_data_reg <= rx_data;
        rx_ready_reg <= rx_ready;
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_RECEIVING;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_count <= 0;
            prev_byte <= 0;
            image_loaded <= 0;
            debug_rx_count <= 0;
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            if (weights_loaded) begin
                case (state)
                    STATE_RECEIVING: begin
                        if (rx_ready_reg) begin
                            debug_rx_count <= debug_rx_count + 1;
                            
                            // 1. Write Valid Payload Bytes ONLY
                            // Stop writing if we hit 784 bytes (don't write markers to RAM)
                            if (byte_count < IMG_SIZE) begin
                                wr_addr <= byte_count;
                                wr_data <= rx_data_reg;
                                wr_en <= 1;
                                byte_count <= byte_count + 1;
                            end
                            
                            // 2. Track for markers
                            prev_byte <= rx_data_reg;

                            // 3. Check for End Sequence
                            if (byte_count >= IMG_SIZE && prev_byte == IMG_END1 && rx_data_reg == IMG_END2) begin
                                state <= STATE_DONE;
                                image_loaded <= 1;
                            end
                        end
                    end
                    
                    STATE_DONE: begin
                        image_loaded <= 0;
                        // Auto-reset logic for next image
                        state <= STATE_RECEIVING;
                        byte_count <= 0;
                        prev_byte <= 0;  // Reset prev_byte for next image
                        debug_rx_count <= 0;
                    end
                endcase
            end
        end
    end
endmodule
