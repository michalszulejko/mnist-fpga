/*
================================================================================
Top Module - Complete Two-Hidden-Layer Inference System with Unified UART Router
================================================================================
This module instantiates all sub-modules and connects them together.
Uses a SINGLE uart_rx instance (in uart_router) for the entire system.

Model Architecture:
  - Input: 784 pixels (28x28 image)
  - Layer 1: 784 → 16 neurons (ReLU activation)
  - Layer 2: 16 → 16 neurons (ReLU activation)
  - Layer 3: 16 → 10 outputs (softmax/logits)

Key difference from regresja model:
  - 3 layers instead of 1 layer
  - 6 BRAMs for weights/biases (L1, L2, L3) instead of 2
  - ReLU activations between layers
  - More complex inference state machine

================================================================================
*/

module top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line
    output wire [15:0] led,       // Status LEDs
    output wire [6:0] seg,        // 7-segment display segments
    output wire [3:0] an,         // 7-segment display anodes
    input wire [15:0] sw,         // Switches (for control/debug)
    input wire btnU,              // Up button (optional)
    input wire btnD,              // Down button (optional)
    input wire btnL,              // Left button (optional)
    input wire btnR               // Right button (optional)
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // UART Router signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    wire [7:0] image_rx_data;
    wire image_rx_ready;
    wire [7:0] cmd_rx_data;
    wire cmd_rx_ready;
    
    // Weight loader signals (6 read ports for 3 layers)
    wire [13:0] L1_weight_rd_addr;
    wire [7:0]  L1_weight_rd_data;
    wire [3:0]  L1_bias_rd_addr;
    wire [31:0] L1_bias_rd_data;
    
    wire [7:0]  L2_weight_rd_addr;
    wire [7:0]  L2_weight_rd_data;
    wire [3:0]  L2_bias_rd_addr;
    wire [31:0] L2_bias_rd_data;
    
    wire [7:0]  L3_weight_rd_addr;
    wire [7:0]  L3_weight_rd_data;
    wire [3:0]  L3_bias_rd_addr;
    wire [31:0] L3_bias_rd_data;
    
    wire        weights_loaded;
    wire [15:0] loader_led;
    
    // Inference signals
    wire [13:0] inf_L1_weight_addr;
    wire [3:0]  inf_L1_bias_addr;
    wire [7:0]  inf_L2_weight_addr;
    wire [3:0]  inf_L2_bias_addr;
    wire [7:0]  inf_L3_weight_addr;
    wire [3:0]  inf_L3_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
    
    // Class scores from inference module
    wire signed [31:0] class_score_0;
    wire signed [31:0] class_score_1;
    wire signed [31:0] class_score_2;
    wire signed [31:0] class_score_3;
    wire signed [31:0] class_score_4;
    wire signed [31:0] class_score_5;
    wire signed [31:0] class_score_6;
    wire signed [31:0] class_score_7;
    wire signed [31:0] class_score_8;
    wire signed [31:0] class_score_9;
    
    // Image RAM signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    
    // Control signals
    wire start_inference_pulse;
    
    // Predicted digit RAM signals
    wire digit_ram_wr_en;
    wire [7:0] digit_ram_rd_data;
    
    // Scores RAM signals
    wire scores_ram_wr_en;
    wire [5:0] scores_ram_rd_addr;
    wire [7:0] scores_ram_rd_data;
    
    // UART TX signals - shared between digit and scores readers
    wire [7:0] digit_tx_data;
    wire digit_tx_send;
    wire [7:0] scores_tx_data;
    wire scores_tx_send;
    wire [7:0] tx_data;
    wire tx_send;
    wire tx_busy;
    wire tx_out;
    
    // LED and display registers
    reg [15:0] led_reg;
    
    // Display digit signals
    wire [3:0] digit_left;   // From memory (persistent)
    wire [3:0] digit_right;  // Current predicted_digit
    
    // Edge detector for inference_done (to create write pulse)
    reg inference_done_prev;
    
    // =========================================================================
    // UART Router - SINGLE uart_rx for the entire system
    // =========================================================================
    uart_router u_uart_router (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(weights_loaded),
        .weight_rx_data(weight_rx_data),
        .weight_rx_ready(weight_rx_ready),
        .image_rx_data(image_rx_data),
        .image_rx_ready(image_rx_ready),
        .cmd_rx_data(cmd_rx_data),
        .cmd_rx_ready(cmd_rx_ready)
    );
    
    // =========================================================================
    // Weight Loader (receives routed data from uart_router)
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
        
        .L1_weight_rd_addr(L1_weight_rd_addr),
        .L1_weight_rd_data(L1_weight_rd_data),
        .L1_bias_rd_addr(L1_bias_rd_addr),
        .L1_bias_rd_data(L1_bias_rd_data),
        
        .L2_weight_rd_addr(L2_weight_rd_addr),
        .L2_weight_rd_data(L2_weight_rd_data),
        .L2_bias_rd_addr(L2_bias_rd_addr),
        .L2_bias_rd_data(L2_bias_rd_data),
        
        .L3_weight_rd_addr(L3_weight_rd_addr),
        .L3_weight_rd_data(L3_weight_rd_data),
        .L3_bias_rd_addr(L3_bias_rd_addr),
        .L3_bias_rd_data(L3_bias_rd_data),
        
        .transfer_done(weights_loaded),
        .led(loader_led)
    );
    
    // =========================================================================
    // Image Loader (receives routed data from uart_router)
    // =========================================================================
    image_loader u_image_loader (
        .clk(clk),
        .rst(rst),
        .weights_loaded(weights_loaded),
        .rx_data(image_rx_data),
        .rx_ready(image_rx_ready),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .image_loaded(img_loaded)
    );
    
    // =========================================================================
    // Image RAM (784 bytes)
    // =========================================================================
    image_ram u_image_ram (
        .clk(clk),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .rd_addr(inf_input_addr),
        .rd_data(inf_input_pixel)
    );
    
    // =========================================================================
    // Inference Module (3-layer neural network)
    // =========================================================================
    
    // Connect weight/bias addresses from inference to loader
    assign L1_weight_rd_addr = inf_L1_weight_addr;
    assign L1_bias_rd_addr = inf_L1_bias_addr;
    assign L2_weight_rd_addr = inf_L2_weight_addr;
    assign L2_bias_rd_addr = inf_L2_bias_addr;
    assign L3_weight_rd_addr = inf_L3_weight_addr;
    assign L3_bias_rd_addr = inf_L3_bias_addr;
    
    inference u_inference (
        .clk(clk),
        .rst(rst),
        
        // Layer 1 interface
        .L1_weight_addr(inf_L1_weight_addr),
        .L1_weight_data(L1_weight_rd_data),
        .L1_bias_addr(inf_L1_bias_addr),
        .L1_bias_data(L1_bias_rd_data),
        
        // Layer 2 interface
        .L2_weight_addr(inf_L2_weight_addr),
        .L2_weight_data(L2_weight_rd_data),
        .L2_bias_addr(inf_L2_bias_addr),
        .L2_bias_data(L2_bias_rd_data),
        
        // Layer 3 interface
        .L3_weight_addr(inf_L3_weight_addr),
        .L3_weight_data(L3_weight_rd_data),
        .L3_bias_addr(inf_L3_bias_addr),
        .L3_bias_data(L3_bias_rd_data),
        
        // Control
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        
        // Outputs
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy),
        
        // Class scores
        .class_score_0(class_score_0),
        .class_score_1(class_score_1),
        .class_score_2(class_score_2),
        .class_score_3(class_score_3),
        .class_score_4(class_score_4),
        .class_score_5(class_score_5),
        .class_score_6(class_score_6),
        .class_score_7(class_score_7),
        .class_score_8(class_score_8),
        .class_score_9(class_score_9)
    );
    
    // =========================================================================
    // Auto-start inference when image is loaded
    // =========================================================================
    reg img_loaded_prev;
    always @(posedge clk) begin
        if (rst)
            img_loaded_prev <= 0;
        else
            img_loaded_prev <= img_loaded;
    end
    
    assign start_inference_pulse = img_loaded && !img_loaded_prev;
    
    // =========================================================================
    // Edge detector for inference_done (to write predicted digit to RAM)
    // =========================================================================
    always @(posedge clk) begin
        if (rst)
            inference_done_prev <= 0;
        else
            inference_done_prev <= inference_done;
    end
    
    assign digit_ram_wr_en = inference_done && !inference_done_prev;
    assign scores_ram_wr_en = inference_done && !inference_done_prev;
    
    // =========================================================================
    // Predicted Digit RAM
    // =========================================================================
    predicted_digit_ram u_predicted_digit_ram (
        .clk(clk),
        .wr_en(digit_ram_wr_en),
        .wr_data(predicted_digit),
        .rd_addr(1'b0),  // Always read from address 0
        .rd_data(digit_ram_rd_data)
    );
    
    // =========================================================================
    // Scores RAM
    // =========================================================================
    scores_ram u_scores_ram (
        .clk(clk),
        .wr_en(scores_ram_wr_en),
        .score_0(class_score_0),
        .score_1(class_score_1),
        .score_2(class_score_2),
        .score_3(class_score_3),
        .score_4(class_score_4),
        .score_5(class_score_5),
        .score_6(class_score_6),
        .score_7(class_score_7),
        .score_8(class_score_8),
        .score_9(class_score_9),
        .rd_addr(scores_ram_rd_addr),
        .rd_data(scores_ram_rd_data)
    );
    
    // =========================================================================
    // UART TX for responses
    // =========================================================================
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_uart_tx (
        .clk(clk),
        .rst(rst),
        .data(tx_data),
        .send(tx_send),
        .tx(tx_out),
        .busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Reader (0xCC protocol) - uses routed command interface
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(cmd_rx_data),
        .rx_ready(cmd_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(digit_tx_data),
        .tx_send(digit_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // Scores Reader (0xCD protocol) - uses routed command interface
    // =========================================================================
    scores_reader u_scores_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(cmd_rx_data),
        .rx_ready(cmd_rx_ready),
        .scores_data(scores_ram_rd_data),
        .scores_addr(scores_ram_rd_addr),
        .tx_data(scores_tx_data),
        .tx_send(scores_tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // TX Arbiter - Multiplex between digit_reader and scores_reader
    // =========================================================================
    // Simple OR-based arbiter (only one will be active at a time due to
    // different request bytes 0xCC vs 0xCD)
    assign tx_data = digit_tx_send ? digit_tx_data : scores_tx_data;
    assign tx_send = digit_tx_send | scores_tx_send;
    
    // =========================================================================
    // Digit Display Reader - Reads from predicted_digit_ram
    // =========================================================================
    digit_display_reader u_digit_display_reader (
        .clk(clk),
        .rst(rst),
        .digit_ram_data(digit_ram_rd_data),
        .display_digit(digit_left)
    );
    
    // =========================================================================
    // LED Control
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
        end else begin
            // Show loader status when loading, inference result when done
            if (!weights_loaded) begin
                led_reg <= loader_led;
            end else begin
                led_reg[3:0] <= predicted_digit;
                led_reg[4] <= inference_busy;
                led_reg[5] <= inference_done;
                led_reg[6] <= img_loaded;
                led_reg[7] <= weights_loaded;
                led_reg[15:8] <= loader_led[15:8];
            end
        end
    end
    
    assign led = led_reg;
    
    // =========================================================================
    // 7-Segment Display
    // =========================================================================
    // Rightmost digit shows current predicted_digit, leftmost shows stored value from memory
    assign digit_right = predicted_digit;
    
    seven_segment_display u_display (
        .clk(clk),
        .rst(rst),
        .digit_left(digit_left),
        .digit_right(digit_right),
        .seg(seg),
        .an(an)
    );
    
    // Connect TX output
    assign tx = tx_out;

endmodule

