module top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line
    output wire [15:0] led,       // Status LEDs
    output wire [6:0] seg,        // 7-segment display segments
    output wire [3:0] an,         // 7-segment display anodes
    input wire [15:0] sw,         // Switches (optional)
    input wire btnU,              // Buttons (optional)
    input wire btnD,
    input wire btnL,
    input wire btnR
);

    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // UART Router signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    wire [7:0] image_rx_data;
    wire image_rx_ready;
    wire [7:0] cmd_rx_data;
    wire cmd_rx_ready;
    
    // Weight Loader Signals
    wire [11:0] conv_w_wr_addr;
    wire [7:0]  conv_w_wr_data;
    wire        conv_w_wr_en;

    wire [4:0]  conv_b_wr_addr;
    wire [31:0] conv_b_wr_data;
    wire        conv_b_wr_en;

    wire [15:0] fc_w_wr_addr;
    wire [7:0]  fc_w_wr_data;
    wire        fc_w_wr_en;

    wire [7:0]  fc_b_wr_addr;
    wire [31:0] fc_b_wr_data;
    wire        fc_b_wr_en;
    
    wire weights_loaded;
    wire [15:0] loader_led; // DEBUG LEDs
    
    // Inference Signals (Read Side)
    wire [11:0] conv_w_rd_addr;
    wire [7:0]  conv_w_rd_data;

    wire [4:0]  conv_b_rd_addr;
    wire [31:0] conv_b_rd_data;

    wire [15:0] fc_w_rd_addr;
    wire [7:0]  fc_w_rd_data;

    wire [7:0]  fc_b_rd_addr;
    wire [31:0] fc_b_rd_data;

    // Intermediate Buffers
    wire [12:0] buf_a_addr;
    wire [7:0]  buf_a_wr_data;
    wire        buf_a_wr_en;
    wire [7:0]  buf_a_rd_data;

    wire [10:0] buf_b_addr;
    wire [7:0]  buf_b_wr_data;
    wire        buf_b_wr_en;
    wire [7:0]  buf_b_rd_data;

    wire [8:0]  buf_c_addr;
    wire [7:0]  buf_c_wr_data;
    wire        buf_c_wr_en;
    wire [7:0]  buf_c_rd_data;

    // Tanh LUT Signals
    wire [7:0]  tanh_addr;
    wire [7:0]  tanh_rd_data;
    wire [7:0]  tanh_wr_addr;
    wire [7:0]  tanh_wr_data;
    wire        tanh_wr_en;

    // Debug reader signals
    wire [7:0]  dbg_tanh_addr;
    wire [7:0]  dbg_tanh_data;
    wire [11:0] dbg_conv_addr;
    wire [7:0]  dbg_conv_data;
    wire [9:0]  dbg_img_addr;
    wire [7:0]  dbg_img_data;
    wire [7:0]  dbg_tx_data;
    wire        dbg_tx_send;
    
    // Image RAM Signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    wire [15:0] image_loader_debug_rx_count;
    
    wire [9:0]  img_rd_addr;
    wire [7:0]  img_rd_data;
    
    // Control & Results
    wire        start_inference_pulse;
    wire        inference_done;
    wire [3:0]  predicted_digit;
    
    // Class Scores
    wire signed [31:0] class_score_0, class_score_1, class_score_2, class_score_3, class_score_4;
    wire signed [31:0] class_score_5, class_score_6, class_score_7, class_score_8, class_score_9;

    // Edge Detectors
    reg img_loaded_prev;
    reg inference_done_prev;
    
    // RAM Write Enables
    wire digit_ram_wr_en;
    wire scores_ram_wr_en;
    
    // UART TX & Readers
    wire [7:0] digit_ram_rd_data;
    wire [5:0] scores_ram_rd_addr;
    wire [7:0] scores_ram_rd_data;
    
    wire [7:0] tx_data;
    wire       tx_send;
    wire       tx_busy;
    wire       tx_out;
    
    wire [7:0] digit_tx_data;
    wire       digit_tx_send;
    wire [7:0] scores_tx_data;
    wire       scores_tx_send;
    
    // Display
    wire [3:0] digit_left;
    
    // DEBUG: Heartbeat counter
    reg [26:0] heartbeat_cnt;

    // =========================================================================
    // 1. UART ROUTER
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
    // 2. WEIGHT LOADER
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
        .transfer_done(weights_loaded),

        // Conv (L1 + L2 combined)
        .conv_w_addr(conv_w_wr_addr),
        .conv_w_data(conv_w_wr_data),
        .conv_w_en(conv_w_wr_en),
        .conv_b_addr(conv_b_wr_addr),
        .conv_b_data(conv_b_wr_data),
        .conv_b_en(conv_b_wr_en),

        // FC (Fully Connected layers)
        .fc_w_addr(fc_w_wr_addr),
        .fc_w_data(fc_w_wr_data),
        .fc_w_en(fc_w_wr_en),
        .fc_b_addr(fc_b_wr_addr),
        .fc_b_data(fc_b_wr_data),
        .fc_b_en(fc_b_wr_en),

        // Tanh LUT
        .tanh_addr(tanh_wr_addr),
        .tanh_data(tanh_wr_data),
        .tanh_en(tanh_wr_en),

        // Debug
        .led(loader_led)
    );

    // =========================================================================
    // 3. CONV MEMORIES (Weights & Biases)
    // =========================================================================
    conv_weights_ram u_conv_w_ram (
        .clk(clk),
        .wr_addr(conv_w_wr_addr),
        .wr_data(conv_w_wr_data),
        .wr_en(conv_w_wr_en),
        .rd_addr(conv_w_rd_addr),
        .rd_data(conv_w_rd_data),
        .dbg_addr(dbg_conv_addr),
        .dbg_data(dbg_conv_data)
    );

    conv_biases_ram u_conv_b_ram (
        .clk(clk),
        .wr_addr(conv_b_wr_addr),
        .wr_data(conv_b_wr_data),
        .wr_en(conv_b_wr_en),
        .rd_addr(conv_b_rd_addr),
        .rd_data(conv_b_rd_data)
    );

    // =========================================================================
    // 4. FC (FULLY CONNECTED) BIASES
    // =========================================================================
    fc_biases_ram u_fc_b_ram (
        .clk(clk),
        .wr_addr(fc_b_wr_addr),
        .wr_data(fc_b_wr_data),
        .wr_en(fc_b_wr_en),
        .rd_addr(fc_b_rd_addr),
        .rd_data(fc_b_rd_data)
    );

    // =========================================================================
    // 4B. TANH LOOKUP TABLE
    // =========================================================================
    tanh_lut u_tanh_lut (
        .clk(clk),
        .addr(tanh_addr),
        .data(tanh_rd_data),
        .dbg_addr(dbg_tanh_addr),
        .dbg_data(dbg_tanh_data),
        .wr_addr(tanh_wr_addr),
        .wr_data(tanh_wr_data),
        .wr_en(tanh_wr_en)
    );

    // =========================================================================
    // 5. FC WEIGHTS RAM
    // =========================================================================
    fc_weights_ram u_fc_w_ram (
        .clk(clk),
        .wr_addr(fc_w_wr_addr),
        .wr_data(fc_w_wr_data),
        .wr_en(fc_w_wr_en),
        .rd_addr(fc_w_rd_addr),
        .rd_data(fc_w_rd_data)
    );

    // =========================================================================
    // 6. CNN RAM (Buffers A/B/C)
    // =========================================================================
    ram_cnn u_ram_cnn (
        .clk(clk),

        // Buffer A
        .buf_a_addr(buf_a_addr),
        .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en),
        .buf_a_rd_data(buf_a_rd_data),

        // Buffer B
        .buf_b_addr(buf_b_addr),
        .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en),
        .buf_b_rd_data(buf_b_rd_data),

        // Buffer C
        .buf_c_addr(buf_c_addr),
        .buf_c_wr_data(buf_c_wr_data),
        .buf_c_wr_en(buf_c_wr_en),
        .buf_c_rd_data(buf_c_rd_data)
    );

    // =========================================================================
    // 6. IMAGE LOADING & STORAGE
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
        .image_loaded(img_loaded),
        .debug_rx_count(image_loader_debug_rx_count)
    );

    image_ram u_image_ram (
        .clk(clk),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .rd_addr(img_rd_addr),
        .rd_data(img_rd_data),
        .dbg_addr(dbg_img_addr),
        .dbg_data(dbg_img_data)
    );

    // =========================================================================
    // 7. INFERENCE ENGINE
    // =========================================================================
    always @(posedge clk) begin
        if (rst) img_loaded_prev <= 0;
        else img_loaded_prev <= img_loaded;
    end
    assign start_inference_pulse = img_loaded && !img_loaded_prev;

    inference u_inference (
        .clk(clk),
        .rst(rst),
        .start(start_inference_pulse),
        .done(inference_done),
        .predicted_digit(predicted_digit),
        
        // Image Read
        .img_addr(img_rd_addr),
        .img_data(img_rd_data),
        
        // Conv Read
        .conv_w_addr(conv_w_rd_addr),
        .conv_w_data(conv_w_rd_data),
        .conv_b_addr(conv_b_rd_addr),
        .conv_b_data(conv_b_rd_data),

        // FC Read
        .fc_w_addr(fc_w_rd_addr),
        .fc_w_data(fc_w_rd_data),
        .fc_b_addr(fc_b_rd_addr),
        .fc_b_data(fc_b_rd_data),

        // Tanh LUT
        .tanh_addr(tanh_addr),
        .tanh_data(tanh_rd_data),

        // Buffers
        .buf_a_addr(buf_a_addr),
        .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en),
        .buf_a_rd_data(buf_a_rd_data),

        .buf_b_addr(buf_b_addr),
        .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en),
        .buf_b_rd_data(buf_b_rd_data),

        .buf_c_addr(buf_c_addr),
        .buf_c_wr_data(buf_c_wr_data),
        .buf_c_wr_en(buf_c_wr_en),
        .buf_c_rd_data(buf_c_rd_data),

        // Scores
        .class_score_0(class_score_0), .class_score_1(class_score_1),
        .class_score_2(class_score_2), .class_score_3(class_score_3),
        .class_score_4(class_score_4), .class_score_5(class_score_5),
        .class_score_6(class_score_6), .class_score_7(class_score_7),
        .class_score_8(class_score_8), .class_score_9(class_score_9)
    );

    // =========================================================================
    // 8. RESULT STORAGE & READBACK
    // =========================================================================
    always @(posedge clk) begin
        if (rst) inference_done_prev <= 0;
        else inference_done_prev <= inference_done;
    end
    assign digit_ram_wr_en  = inference_done && !inference_done_prev;
    assign scores_ram_wr_en = inference_done && !inference_done_prev;

    predicted_digit_ram u_pred_ram (
        .clk(clk),
        .wr_en(digit_ram_wr_en),
        .wr_data(predicted_digit),
        .rd_addr(1'b0),
        .rd_data(digit_ram_rd_data)
    );

    scores_ram u_scores_ram (
        .clk(clk),
        .wr_en(scores_ram_wr_en),
        .score_0(class_score_0), .score_1(class_score_1),
        .score_2(class_score_2), .score_3(class_score_3),
        .score_4(class_score_4), .score_5(class_score_5),
        .score_6(class_score_6), .score_7(class_score_7),
        .score_8(class_score_8), .score_9(class_score_9),
        .rd_addr(scores_ram_rd_addr),
        .rd_data(scores_ram_rd_data)
    );

    // =========================================================================
    // 9. UART TX & DISPLAY
    // =========================================================================
    uart_tx #(.CLK_FREQ(100_000_000), .BAUD_RATE(115200)) u_uart_tx (
        .clk(clk), .rst(rst),
        .data(tx_data), .send(tx_send), .tx(tx_out), .busy(tx_busy)
    );
    assign tx = tx_out;

    digit_reader u_digit_reader (
        .clk(clk), .rst(rst),
        .rx_data(cmd_rx_data), .rx_ready(cmd_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(digit_tx_data), .tx_send(digit_tx_send), .tx_busy(tx_busy)
    );

    scores_reader u_scores_reader (
        .clk(clk), .rst(rst),
        .rx_data(cmd_rx_data), .rx_ready(cmd_rx_ready),
        .scores_data(scores_ram_rd_data), .scores_addr(scores_ram_rd_addr),
        .tx_data(scores_tx_data), .tx_send(scores_tx_send), .tx_busy(tx_busy)
    );

    debug_reader u_debug_reader (
        .clk(clk), .rst(rst),
        .rx_data(cmd_rx_data), .rx_ready(cmd_rx_ready),
        .tanh_dbg_addr(dbg_tanh_addr), .tanh_dbg_data(dbg_tanh_data),
        .conv_dbg_addr(dbg_conv_addr), .conv_dbg_data(dbg_conv_data),
        .img_dbg_addr(dbg_img_addr), .img_dbg_data(dbg_img_data),
        .tx_data(dbg_tx_data), .tx_send(dbg_tx_send), .tx_busy(tx_busy)
    );

    // TX Mux: Priority to digit, then scores, then debug
    assign tx_data = digit_tx_send ? digit_tx_data :
                     scores_tx_send ? scores_tx_data : dbg_tx_data;
    assign tx_send = digit_tx_send | scores_tx_send | dbg_tx_send;

    digit_display_reader u_disp_reader (
        .clk(clk), .rst(rst),
        .digit_ram_data(digit_ram_rd_data),
        .display_digit(digit_left)
    );

    seven_segment_display u_display (
        .clk(clk), .rst(rst),
        .digit_left(digit_left),
        .digit_right(predicted_digit),
        .seg(seg), .an(an)
    );

    // =========================================================================
    // 10. FLIGHT RECORDER LED LOGIC (DEBUG)
    // =========================================================================
    // These LEDs latch ON if a condition is ever met.
    // To reset them, you must press the center button (RST).
    // =========================================================================
    always @(posedge clk) heartbeat_cnt <= heartbeat_cnt + 1;

    reg [15:0] debug_latch;
    
    always @(posedge clk) begin
        if (rst) begin
            debug_latch <= 0;
        end else begin
            // -------------------------------------------------------------
            // SYSTEM STATUS
            // -------------------------------------------------------------
            // [0] Weights Loaded (Steady)
            debug_latch[0] <= weights_loaded;

            // [1] Image Loaded (Steady)
            debug_latch[1] <= img_loaded;

            // NEW DEBUG SIGNALS for weight loading:
            // [11] Latch when ANY weight byte is received
            if (weight_rx_ready) debug_latch[11] <= 1;

            // [12] Latch when weight RAMs are being written
            if (conv_w_wr_en || fc_w_wr_en) debug_latch[12] <= 1;

            // [13] Latch when bias RAMs are being written
            if (conv_b_wr_en || fc_b_wr_en) debug_latch[13] <= 1;

            // [14] Latch when tanh LUT is being written
            if (tanh_wr_en) debug_latch[14] <= 1;
            
            // -------------------------------------------------------------
            // CONTROL SIGNALS (Latched)
            // -------------------------------------------------------------
            // [2] Did START Pulse trigger?
            if (start_inference_pulse) debug_latch[2] <= 1;
            
            // [3] Did DONE Pulse trigger? (If this is OFF, FSM is stuck)
            if (inference_done) debug_latch[3] <= 1;
            
            // -------------------------------------------------------------
            // DATA INTEGRITY (Latched)
            // -------------------------------------------------------------
            // [4] NON-ZERO PIXEL read? (If OFF, image RAM is empty/black)
            if (img_rd_data != 0) debug_latch[4] <= 1;
            
            // [5] NON-ZERO WEIGHT read? (If OFF, weight RAM is empty)
            if (conv_w_rd_data != 0) debug_latch[5] <= 1;
            
            // [6] NON-ZERO BIAS read?
            if (conv_b_rd_data != 0) debug_latch[6] <= 1;
            
            // -------------------------------------------------------------
            // PROCESSING ACTIVITY (Latched)
            // -------------------------------------------------------------
            // [7] L1 Conv Write Enable (Did we write to Buffer A?)
            if (buf_a_wr_en) debug_latch[7] <= 1;
            
            // [8] NON-ZERO L1 Output (Did L1 produce valid data?)
            if (buf_a_wr_en && buf_a_wr_data != 0) debug_latch[8] <= 1;
            
            // [9] Pooling/L2 Write Enable (Did we write to Buffer B?)
            if (buf_b_wr_en) debug_latch[9] <= 1;
            
            // [10] Scores Write Enable (Did we try to save results?)
            if (scores_ram_wr_en) debug_latch[10] <= 1;
            
            // -------------------------------------------------------------
            // HEARTBEAT
            // -------------------------------------------------------------
            debug_latch[15] <= heartbeat_cnt[26];
        end
    end
    
    // LED Mapping:
    // LED[7:0]   = debug status flags (always visible)
    // LED[11:8]  = weight loading debug signals OR image rx count bits [3:0]
    // LED[14:12] = weight loading debug signals (latched) - always visible
    // LED[15]    = heartbeat
    assign led[7:0]   = debug_latch[7:0];
    assign led[11:8]  = weights_loaded ? image_loader_debug_rx_count[3:0] : debug_latch[11:8];
    assign led[14:12] = debug_latch[14:12];
    assign led[15]    = debug_latch[15];

endmodule