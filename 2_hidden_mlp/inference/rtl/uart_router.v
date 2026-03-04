/*
================================================================================
UART Router - Binary Safe Version
================================================================================
Fix: Implementation of Byte Counting to ensure Binary Safety.
We strictly ignore "End Markers" until the expected number of data bytes 
has been received. This prevents random weights/pixels that resemble 
protocol markers from terminating the transfer early.
================================================================================
*/

module uart_router (
    input wire clk,
    input wire rst,
    input wire rx,                    // UART RX line (from PC)
    
    // Status inputs
    input wire weights_loaded,        // HIGH when weights are fully loaded
    
    // Weight loader interface
    output reg [7:0] weight_rx_data,
    output reg weight_rx_ready,
    
    // Image loader interface
    output reg [7:0] image_rx_data,
    output reg image_rx_ready,
    
    // Command interface (for digit_reader and scores_reader)
    output reg [7:0] cmd_rx_data,
    output reg cmd_rx_ready
);

    // Protocol markers
    localparam WEIGHT_START1 = 8'hAA;
    localparam WEIGHT_START2 = 8'h55;
    localparam WEIGHT_END1   = 8'h55;
    localparam WEIGHT_END2   = 8'hAA;
    
    localparam IMAGE_START1 = 8'hBB;
    localparam IMAGE_START2 = 8'h66;
    localparam IMAGE_END1   = 8'h66;
    localparam IMAGE_END2   = 8'hBB;
    
    localparam CMD_DIGIT_READ  = 8'hCC;
    localparam CMD_SCORES_READ = 8'hCD;
    
    // Data sizes
    localparam WEIGHT_DATA_SIZE = 13128;  // 12,960 weights + 168 bias bytes
    localparam IMAGE_DATA_SIZE = 784;     // 784 pixels
    
    // Router states
    localparam STATE_IDLE          = 4'd0;
    localparam STATE_WAIT_WEIGHT2  = 4'd1;
    localparam STATE_RECEIVING_WEIGHTS = 4'd2;
    localparam STATE_WAIT_IMAGE2   = 4'd3;
    localparam STATE_RECEIVING_IMAGE = 4'd4;
    
    // Shared UART RX
    wire [7:0] rx_data;
    wire rx_ready;
    
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_uart_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );
    
    // State registers
    reg [3:0] state;
    reg [14:0] byte_count;  // Counter for data bytes
    reg [7:0] prev_byte;    // For detecting end markers
    
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            byte_count <= 0;
            prev_byte <= 0;
            weight_rx_data <= 0;
            weight_rx_ready <= 0;
            image_rx_data <= 0;
            image_rx_ready <= 0;
            cmd_rx_data <= 0;
            cmd_rx_ready <= 0;
        end else begin
            // Default: all ready signals are pulses (1 cycle only)
            weight_rx_ready <= 0;
            image_rx_ready <= 0;
            cmd_rx_ready <= 0;
            
            case (state)
                // ============================================================
                // IDLE: Detect protocol start markers or single-byte commands
                // ============================================================
                STATE_IDLE: begin
                    if (rx_ready) begin
                        case (rx_data)
                            // Weight loading start marker (first byte)
                            WEIGHT_START1: begin
                                if (!weights_loaded) begin
                                    state <= STATE_WAIT_WEIGHT2;
                                end
                            end
                            
                            // Image loading start marker (first byte)
                            IMAGE_START1: begin
                                if (weights_loaded) begin
                                    state <= STATE_WAIT_IMAGE2;
                                end
                            end
                            
                            // Single-byte commands
                            CMD_DIGIT_READ, CMD_SCORES_READ: begin
                                if (weights_loaded) begin
                                    cmd_rx_data <= rx_data;
                                    cmd_rx_ready <= 1;
                                end
                            end
                            
                            default: begin
                                // Unknown byte, ignore
                            end
                        endcase
                    end
                end
                
                // ============================================================
                // Wait for second weight start marker (0x55)
                // ============================================================
                STATE_WAIT_WEIGHT2: begin
                    if (rx_ready) begin
                        if (rx_data == WEIGHT_START2) begin
                            // Valid start sequence, begin receiving weights
                            state <= STATE_RECEIVING_WEIGHTS;
                            byte_count <= 0;
                            prev_byte <= 0;
                            
                            // Forward start markers to weight_loader
                            weight_rx_data <= WEIGHT_START2;
                            weight_rx_ready <= 1;
                        end else if (rx_data == WEIGHT_START1) begin
                            // Another 0xAA, stay waiting
                            state <= STATE_WAIT_WEIGHT2;
                        end else begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Receiving weight data (BINARY SAFE)
                // ============================================================
                STATE_RECEIVING_WEIGHTS: begin
                    if (rx_ready) begin
                        // Forward byte to weight_loader
                        weight_rx_data <= rx_data;
                        weight_rx_ready <= 1;
                        
                        // Increment counter
                        byte_count <= byte_count + 1;
                        prev_byte <= rx_data;

                        // === BINARY SAFETY FIX ===
                        // We strictly verify that we have received at least the expected 
                        // amount of data BEFORE we start looking for the End Marker.
                        // byte_count updates non-blocking, so we use +1 logic or > comparison.
                        // We wait until byte_count > WEIGHT_DATA_SIZE to ensure we are 
                        // looking at the marker bytes, not the data bytes.
                        if (byte_count >= WEIGHT_DATA_SIZE) begin
                            if (prev_byte == WEIGHT_END1 && rx_data == WEIGHT_END2) begin
                                // End of weight transfer
                                state <= STATE_IDLE;
                            end
                        end
                        
                        // Safety timeout (overflow protection)
                        if (byte_count > WEIGHT_DATA_SIZE + 10) begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Wait for second image start marker (0x66)
                // ============================================================
                STATE_WAIT_IMAGE2: begin
                    if (rx_ready) begin
                        if (rx_data == IMAGE_START2) begin
                            state <= STATE_RECEIVING_IMAGE;
                            byte_count <= 0;
                            prev_byte <= 0;
                            
                            image_rx_data <= IMAGE_START2;
                            image_rx_ready <= 1;
                        end else if (rx_data == IMAGE_START1) begin
                            state <= STATE_WAIT_IMAGE2;
                        end else begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                // ============================================================
                // Receiving image data (BINARY SAFE)
                // ============================================================
                STATE_RECEIVING_IMAGE: begin
                    if (rx_ready) begin
                        image_rx_data <= rx_data;
                        image_rx_ready <= 1;
                        
                        byte_count <= byte_count + 1;
                        prev_byte <= rx_data;

                        // === BINARY SAFETY FIX ===
                        if (byte_count >= IMAGE_DATA_SIZE) begin
                            if (prev_byte == IMAGE_END1 && rx_data == IMAGE_END2) begin
                                state <= STATE_IDLE;
                            end
                        end

                        if (byte_count > IMAGE_DATA_SIZE + 10) begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule