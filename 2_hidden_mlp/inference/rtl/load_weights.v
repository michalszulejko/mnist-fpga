/*
================================================================================
Two-Hidden-Layer Neural Network Weight Loader via Routed UART
================================================================================

Loads trained neural network weights and biases from PC to FPGA BRAM.

Model: Two-Hidden-Layer Neural Network for MNIST
  - Input:  784 pixels (28x28 image, 8-bit unsigned)
  - Layer 1: 784 -> 16 neurons (ReLU)
  - Layer 2: 16 -> 16 neurons (ReLU)
  - Layer 3: 16 -> 10 outputs (softmax)
  
Layer Parameters:
  - L1 Weights: 784 x 16 = 12,544 bytes (8-bit signed)
  - L1 Biases:  16 values (32-bit signed, 4 bytes each little-endian)
  - L2 Weights: 16 x 16 = 256 bytes (8-bit signed)
  - L2 Biases:  16 values (32-bit signed, 4 bytes each little-endian)
  - L3 Weights: 16 x 10 = 160 bytes (8-bit signed)
  - L3 Biases:  10 values (32-bit signed, 4 bytes each little-endian)
  
Total: 13,128 bytes (12,960 weights + 168 biases)

Protocol:
  - Start marker: 0xAA 0x55 (handled by uart_router)
  - Data bytes: 13,128 bytes in order:
      L1_weights (12,544) + L1_biases (64) +
      L2_weights (256) + L2_biases (64) +
      L3_weights (160) + L3_biases (40)
  - End marker: 0x55 0xAA (handled by uart_router)

NOTE: This module receives pre-routed data from uart_router.

Memory Layout:
  Address 0-12543:     L1_weights (12,544 bytes, 8-bit signed)
  Address 12544-12607: L1_biases (64 bytes, 16 x 4-byte little-endian)
  Address 12608-12863: L2_weights (256 bytes, 8-bit signed)
  Address 12864-12927: L2_biases (64 bytes, 16 x 4-byte little-endian)
  Address 12928-13087: L3_weights (160 bytes, 8-bit signed)
  Address 13088-13127: L3_biases (40 bytes, 10 x 4-byte little-endian)

LED Status:
  - led[0]:    Blinks when receiving bytes
  - led[1]:    HIGH when waiting for start marker
  - led[2]:    HIGH when receiving data
  - led[3]:    HIGH when transfer complete (success)
  - led[4]:    HIGH if error (overflow)
  - led[7:5]:  Unused
  - led[15:8]: Lower 8 bits of current address (progress indicator)

================================================================================
*/

// =============================================================================
// TOP MODULE - Use this for synthesis (only external pins)
// =============================================================================
module load_weights_top (
    input wire clk,              // 100 MHz System Clock
    input wire rst,              // Reset Button (active high)
    input wire rx,               // UART RX Line (from PC)
    output wire [15:0] led       // Debug LEDs
);

    // Internal signals for inference module connection (directly exposed here)
    wire [13:0] L1_weight_addr;
    wire [7:0]  L1_weight_data;
    wire [3:0]  L1_bias_addr;
    wire [31:0] L1_bias_data;
    
    wire [7:0]  L2_weight_addr;
    wire [7:0]  L2_weight_data;
    wire [3:0]  L2_bias_addr;
    wire [31:0] L2_bias_data;
    
    wire [7:0]  L3_weight_addr;
    wire [7:0]  L3_weight_data;
    wire [3:0]  L3_bias_addr;
    wire [31:0] L3_bias_data;
    
    wire        transfer_done;
    
    // Routed UART signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    
    // For testing: tie read addresses to 0
    assign L1_weight_addr = 14'd0;
    assign L1_bias_addr = 4'd0;
    assign L2_weight_addr = 8'd0;
    assign L2_bias_addr = 4'd0;
    assign L3_weight_addr = 8'd0;
    assign L3_bias_addr = 4'd0;

    // UART Router (single uart_rx for the system)
    uart_router u_uart_router (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(transfer_done),
        .weight_rx_data(weight_rx_data),
        .weight_rx_ready(weight_rx_ready),
        .image_rx_data(),      // Not used in standalone test
        .image_rx_ready(),     // Not used in standalone test
        .cmd_rx_data(),        // Not used in standalone test
        .cmd_rx_ready()        // Not used in standalone test
    );

    // Instantiate the weight loader
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
        
        .L1_weight_rd_addr(L1_weight_addr),
        .L1_weight_rd_data(L1_weight_data),
        .L1_bias_rd_addr(L1_bias_addr),
        .L1_bias_rd_data(L1_bias_data),
        
        .L2_weight_rd_addr(L2_weight_addr),
        .L2_weight_rd_data(L2_weight_data),
        .L2_bias_rd_addr(L2_bias_addr),
        .L2_bias_rd_data(L2_bias_data),
        
        .L3_weight_rd_addr(L3_weight_addr),
        .L3_weight_rd_data(L3_weight_data),
        .L3_bias_rd_addr(L3_bias_addr),
        .L3_bias_rd_data(L3_bias_data),
        
        .transfer_done(transfer_done),
        .led(led)
    );

endmodule


// =============================================================================
// WEIGHT LOADER MODULE - Has read ports for inference module
// =============================================================================
module weight_loader (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    
    // Routed UART interface (from uart_router)
    input wire [7:0] rx_data,     // Routed RX data
    input wire rx_ready,          // Routed RX ready signal
    
    // Read ports for L1 (inference module)
    input wire [13:0] L1_weight_rd_addr,  // 0 to 12543 (14 bits needed)
    output reg [7:0]  L1_weight_rd_data,  // 8-bit signed weight
    input wire [3:0]  L1_bias_rd_addr,    // 0 to 15 (4 bits)
    output reg [31:0] L1_bias_rd_data,    // 32-bit signed bias
    
    // Read ports for L2 (inference module)
    input wire [7:0]  L2_weight_rd_addr,  // 0 to 255 (8 bits)
    output reg [7:0]  L2_weight_rd_data,  // 8-bit signed weight
    input wire [3:0]  L2_bias_rd_addr,    // 0 to 15 (4 bits)
    output reg [31:0] L2_bias_rd_data,    // 32-bit signed bias
    
    // Read ports for L3 (inference module)
    input wire [7:0]  L3_weight_rd_addr,  // 0 to 159 (8 bits)
    output reg [7:0]  L3_weight_rd_data,  // 8-bit signed weight
    input wire [3:0]  L3_bias_rd_addr,    // 0 to 9 (4 bits)
    output reg [31:0] L3_bias_rd_data,    // 32-bit signed bias
    
    // Status
    output reg transfer_done,
    
    // Debug LEDs
    output reg [15:0] led
);

    // Memory sizes
    localparam L1_WEIGHT_SIZE = 12544;   // 784 x 16
    localparam L1_BIAS_SIZE = 16;
    localparam L2_WEIGHT_SIZE = 256;     // 16 x 16
    localparam L2_BIAS_SIZE = 16;
    localparam L3_WEIGHT_SIZE = 160;     // 16 x 10
    localparam L3_BIAS_SIZE = 10;
    localparam TOTAL_BYTES = 13128;      // Total data bytes
    
    // Memory address boundaries
    localparam L1_WEIGHT_START = 0;
    localparam L1_WEIGHT_END = 12543;
    localparam L1_BIAS_START = 12544;
    localparam L1_BIAS_END = 12607;
    localparam L2_WEIGHT_START = 12608;
    localparam L2_WEIGHT_END = 12863;
    localparam L2_BIAS_START = 12864;
    localparam L2_BIAS_END = 12927;
    localparam L3_WEIGHT_START = 12928;
    localparam L3_WEIGHT_END = 13087;
    localparam L3_BIAS_START = 13088;
    localparam L3_BIAS_END = 13127;
    
    // Protocol markers (for end detection)
    localparam END_BYTE1 = 8'h55;
    localparam END_BYTE2 = 8'hAA;
    
    // State machine states
    localparam STATE_WAIT_DATA   = 3'd0;  // Waiting for first data byte
    localparam STATE_RECEIVING   = 3'd1;
    localparam STATE_DONE        = 3'd2;
    localparam STATE_ERROR       = 3'd3;

    // Block RAM for weights (8-bit values)
    (* ram_style = "block" *) reg [7:0] L1_weight_bram [0:L1_WEIGHT_SIZE-1];
    (* ram_style = "block" *) reg [7:0] L2_weight_bram [0:L2_WEIGHT_SIZE-1];
    (* ram_style = "block" *) reg [7:0] L3_weight_bram [0:L3_WEIGHT_SIZE-1];
    
    // Block RAM for biases (32-bit values)
    (* ram_style = "block" *) reg [31:0] L1_bias_bram [0:L1_BIAS_SIZE-1];
    (* ram_style = "block" *) reg [31:0] L2_bias_bram [0:L2_BIAS_SIZE-1];
    (* ram_style = "block" *) reg [31:0] L3_bias_bram [0:L3_BIAS_SIZE-1];
    
    // State and control registers
    reg [2:0] state;
    reg [14:0] write_addr;       // Address for writing (0 to 13127, needs 15 bits)
    reg [7:0] prev_byte;         // Previous byte for end marker detection
    reg blink_toggle;            // For LED blinking
    reg first_byte;              // Flag to track if we've received the first data byte
    
    // Bias assembly registers (4 bytes -> 32 bits)
    reg [1:0] bias_byte_cnt;     // Which byte of bias we're receiving (0-3)
    reg [31:0] bias_temp;        // Temporary register for assembling bias

    // Synchronous read from weight BRAMs
    always @(posedge clk) begin
        L1_weight_rd_data <= L1_weight_bram[L1_weight_rd_addr];
        L2_weight_rd_data <= L2_weight_bram[L2_weight_rd_addr];
        L3_weight_rd_data <= L3_weight_bram[L3_weight_rd_addr];
    end
    
    // Synchronous read from bias BRAMs
    always @(posedge clk) begin
        L1_bias_rd_data <= L1_bias_bram[L1_bias_rd_addr];
        L2_bias_rd_data <= L2_bias_bram[L2_bias_rd_addr];
        L3_bias_rd_data <= L3_bias_bram[L3_bias_rd_addr];
    end

    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_DATA;
            write_addr <= 0;
            prev_byte <= 0;
            transfer_done <= 0;
            bias_byte_cnt <= 0;
            bias_temp <= 0;
            blink_toggle <= 0;
            first_byte <= 1;
            led <= 16'b0000_0000_0000_0010;  // led[1] = waiting for start
        end else begin
            
            case (state)
                // ============================================
                // Wait for first data byte (start marker handled by router)
                // ============================================
                STATE_WAIT_DATA: begin
                    led[1] <= 1;  // Waiting indicator
                    led[2] <= 0;
                    led[3] <= 0;
                    led[4] <= 0;
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // First byte received from router means start sequence was valid
                        // The router sends us the 0x55 (second start marker) first,
                        // then the actual data bytes
                        if (first_byte) begin
                            // This is the 0x55 from start marker, skip it
                            first_byte <= 0;
                        end else begin
                            // Real data starts here
                            state <= STATE_RECEIVING;
                            write_addr <= 0;
                            bias_byte_cnt <= 0;
                            bias_temp <= 0;
                            led[1] <= 0;
                            led[2] <= 1;  // Receiving indicator
                            
                            // Store first real data byte
                            prev_byte <= rx_data;
                            write_addr <= 1;
                        end
                    end
                end
                
                // ============================================
                // Receiving data bytes
                // ============================================
                STATE_RECEIVING: begin
                    led[2] <= 1;
                    led[15:8] <= write_addr[7:0];  // Show progress
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // First, store the previous byte if we have data pending
                        if (write_addr > 0 && write_addr <= TOTAL_BYTES) begin
                            // Determine which memory to write to based on address
                            if (write_addr - 1 >= L1_WEIGHT_START && write_addr - 1 <= L1_WEIGHT_END) begin
                                // Storing L1 weight
                                L1_weight_bram[write_addr - 1 - L1_WEIGHT_START] <= prev_byte;
                            end
                            else if (write_addr - 1 >= L1_BIAS_START && write_addr - 1 <= L1_BIAS_END) begin
                                // Storing L1 bias byte
                                // Assemble 4 bytes into 32-bit value (little-endian)
                                case (bias_byte_cnt)
                                    2'd0: bias_temp[7:0]   <= prev_byte;
                                    2'd1: bias_temp[15:8]  <= prev_byte;
                                    2'd2: bias_temp[23:16] <= prev_byte;
                                    2'd3: begin
                                        bias_temp[31:24] <= prev_byte;
                                        // Write complete bias to BRAM
                                        L1_bias_bram[(write_addr - 1 - L1_BIAS_START) >> 2] <= 
                                            {prev_byte, bias_temp[23:0]};
                                    end
                                endcase
                                bias_byte_cnt <= bias_byte_cnt + 1;
                            end
                            else if (write_addr - 1 >= L2_WEIGHT_START && write_addr - 1 <= L2_WEIGHT_END) begin
                                // Storing L2 weight
                                L2_weight_bram[write_addr - 1 - L2_WEIGHT_START] <= prev_byte;
                            end
                            else if (write_addr - 1 >= L2_BIAS_START && write_addr - 1 <= L2_BIAS_END) begin
                                // Storing L2 bias byte
                                case (bias_byte_cnt)
                                    2'd0: bias_temp[7:0]   <= prev_byte;
                                    2'd1: bias_temp[15:8]  <= prev_byte;
                                    2'd2: bias_temp[23:16] <= prev_byte;
                                    2'd3: begin
                                        bias_temp[31:24] <= prev_byte;
                                        L2_bias_bram[(write_addr - 1 - L2_BIAS_START) >> 2] <= 
                                            {prev_byte, bias_temp[23:0]};
                                    end
                                endcase
                                bias_byte_cnt <= bias_byte_cnt + 1;
                            end
                            else if (write_addr - 1 >= L3_WEIGHT_START && write_addr - 1 <= L3_WEIGHT_END) begin
                                // Storing L3 weight
                                L3_weight_bram[write_addr - 1 - L3_WEIGHT_START] <= prev_byte;
                            end
                            else if (write_addr - 1 >= L3_BIAS_START && write_addr - 1 <= L3_BIAS_END) begin
                                // Storing L3 bias byte
                                case (bias_byte_cnt)
                                    2'd0: bias_temp[7:0]   <= prev_byte;
                                    2'd1: bias_temp[15:8]  <= prev_byte;
                                    2'd2: bias_temp[23:16] <= prev_byte;
                                    2'd3: begin
                                        bias_temp[31:24] <= prev_byte;
                                        L3_bias_bram[(write_addr - 1 - L3_BIAS_START) >> 2] <= 
                                            {prev_byte, bias_temp[23:0]};
                                    end
                                endcase
                                bias_byte_cnt <= bias_byte_cnt + 1;
                            end
                        end
                        
                        // Check for end marker after storing
                        // CRITICAL FIX: Only check for end marker if we have received enough data
                        // This prevents random weight values that look like 0x55 0xAA from stopping the loader
                        if (write_addr >= TOTAL_BYTES && prev_byte == END_BYTE1 && rx_data == END_BYTE2) begin
                            // End marker detected!
                            state <= STATE_DONE;
                            transfer_done <= 1;
                            led[2] <= 0;
                            led[3] <= 1;  // Success indicator
                        end else begin
                            // Check for overflow
                            if (write_addr >= TOTAL_BYTES + 10) begin
                                state <= STATE_ERROR;
                                led[4] <= 1;
                            end else begin
                                write_addr <= write_addr + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                end
                
                // ============================================
                // Transfer complete
                // ============================================
                STATE_DONE: begin
                    led[3] <= 1;  // Success - stays on
                    led[15:8] <= 8'hFF;  // All progress LEDs on
                    transfer_done <= 1;
                    // Stay in this state until reset
                end
                
                // ============================================
                // Error state
                // ============================================
                STATE_ERROR: begin
                    led[4] <= 1;  // Error indicator stays on
                    // Stay in this state until reset
                end
                
                default: begin
                    state <= STATE_WAIT_DATA;
                end
            endcase
        end
    end

endmodule