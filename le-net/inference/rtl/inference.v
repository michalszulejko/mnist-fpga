/*
================================================================================
LeNet-5 Inference Module
================================================================================
Version 1.0 - LeNet-5 Architecture Implementation

Architecture:
- Conv1: 6 filters, 5x5, padding=2 -> 6x28x28
- Pool1: 2x2 Average Pool -> 6x14x14
- Conv2: 16 filters, 5x5 -> 16x10x10
- Pool2: 2x2 Average Pool -> 16x5x5 = 400 features
- FC1: 400 -> 120 (with Tanh)
- FC2: 120 -> 84 (with Tanh)
- FC3: 84 -> 10 (raw logits)

Key differences from original CNN:
- 5x5 kernels instead of 3x3
- Average pooling instead of max pooling
- Tanh activation via LUT instead of ReLU
- Three FC layers instead of one
================================================================================
*/

module inference (
    input wire clk,
    input wire rst,
    input wire start,
    output reg done,
    output reg [3:0] predicted_digit,

    // Image RAM (28x28 = 784 bytes, but need padding for 5x5 conv)
    output reg [9:0] img_addr,
    input wire signed [7:0] img_data,

    // Conv Weights RAM
    // Conv1: 6*1*5*5 = 150 weights
    // Conv2: 16*6*5*5 = 2400 weights
    // Total: 2550 weights
    output reg [11:0] conv_w_addr,
    input wire signed [7:0] conv_w_data,

    // Conv Biases RAM
    // Conv1: 6 biases
    // Conv2: 16 biases
    // Total: 22 biases (32-bit each)
    output reg [4:0] conv_b_addr,
    input wire signed [31:0] conv_b_data,

    // FC Weights RAM
    // FC1: 120*400 = 48000 weights
    // FC2: 84*120 = 10080 weights
    // FC3: 10*84 = 840 weights
    // Total: 58920 weights
    output reg [15:0] fc_w_addr,
    input wire signed [7:0] fc_w_data,

    // FC Biases RAM
    // FC1: 120 biases
    // FC2: 84 biases
    // FC3: 10 biases
    // Total: 214 biases (32-bit each)
    output reg [7:0] fc_b_addr,
    input wire signed [31:0] fc_b_data,

    // Tanh LUT (256 entries)
    output reg [7:0] tanh_addr,
    input wire signed [7:0] tanh_data,

    // Buffer A - for conv outputs
    // Max size: 6*28*28 = 4704 bytes (Conv1 output)
    output reg [12:0] buf_a_addr,
    output reg [7:0] buf_a_wr_data,
    output reg buf_a_wr_en,
    input wire signed [7:0] buf_a_rd_data,

    // Buffer B - for pooled outputs and FC intermediate
    // Max size: 16*10*10 = 1600 bytes (Conv2 output before pool)
    output reg [10:0] buf_b_addr,
    output reg [7:0] buf_b_wr_data,
    output reg buf_b_wr_en,
    input wire signed [7:0] buf_b_rd_data,

    // Buffer C - for FC intermediate results
    // Max size: 400 bytes (Pool2 output) or 120 bytes (FC1 output)
    output reg [8:0] buf_c_addr,
    output reg [7:0] buf_c_wr_data,
    output reg buf_c_wr_en,
    input wire signed [7:0] buf_c_rd_data,

    // Scores output
    output reg signed [31:0] class_score_0,
    output reg signed [31:0] class_score_1,
    output reg signed [31:0] class_score_2,
    output reg signed [31:0] class_score_3,
    output reg signed [31:0] class_score_4,
    output reg signed [31:0] class_score_5,
    output reg signed [31:0] class_score_6,
    output reg signed [31:0] class_score_7,
    output reg signed [31:0] class_score_8,
    output reg signed [31:0] class_score_9
);

    // FSM States
    localparam IDLE                 = 6'd0;

    // Layer 1 Conv (5x5, 6 filters, padding=2)
    localparam L1_LOAD_BIAS         = 6'd1;
    localparam L1_LOAD_BIAS_WAIT    = 6'd2;
    localparam L1_PREFETCH          = 6'd3;
    localparam L1_CONV              = 6'd4;
    localparam L1_TANH              = 6'd5;
    localparam L1_SAVE              = 6'd6;

    // Layer 1 Average Pool
    localparam L1_POOL              = 6'd7;
    localparam L1_POOL_CALC         = 6'd8;

    // Layer 2 Conv (5x5, 16 filters)
    localparam L2_LOAD_BIAS         = 6'd9;
    localparam L2_LOAD_BIAS_WAIT    = 6'd10;
    localparam L2_PREFETCH          = 6'd11;
    localparam L2_CONV              = 6'd12;
    localparam L2_TANH              = 6'd13;
    localparam L2_SAVE              = 6'd14;

    // Layer 2 Average Pool
    localparam L2_POOL              = 6'd15;
    localparam L2_POOL_CALC         = 6'd16;

    // FC1 Layer (400 -> 120)
    localparam FC1_LOAD_BIAS        = 6'd17;
    localparam FC1_LOAD_BIAS_WAIT   = 6'd18;
    localparam FC1_PREFETCH         = 6'd19;
    localparam FC1_PREFETCH2        = 6'd41;  // Extra wait for 2-cycle BRAM latency
    localparam FC1_MULT             = 6'd20;
    localparam FC1_MULT_WAIT        = 6'd35;  // Wait cycle 1 for BRAM data
    localparam FC1_MULT_WAIT2       = 6'd38;  // Wait cycle 2 for BRAM data (if 2-cycle latency)
    localparam FC1_TANH             = 6'd21;
    localparam FC1_SAVE             = 6'd22;

    // FC2 Layer (120 -> 84)
    localparam FC2_LOAD_BIAS        = 6'd23;
    localparam FC2_LOAD_BIAS_WAIT   = 6'd24;
    localparam FC2_PREFETCH         = 6'd25;
    localparam FC2_PREFETCH2        = 6'd42;  // Extra wait for 2-cycle BRAM latency
    localparam FC2_MULT             = 6'd26;
    localparam FC2_MULT_WAIT        = 6'd36;  // Wait cycle 1 for BRAM data
    localparam FC2_MULT_WAIT2       = 6'd39;  // Wait cycle 2 for BRAM data (if 2-cycle latency)
    localparam FC2_TANH             = 6'd27;
    localparam FC2_SAVE             = 6'd28;

    // FC3 Layer (84 -> 10)
    localparam FC3_LOAD_BIAS        = 6'd29;
    localparam FC3_LOAD_BIAS_WAIT   = 6'd30;
    localparam FC3_PREFETCH         = 6'd31;
    localparam FC3_PREFETCH2        = 6'd43;  // Extra wait for 2-cycle BRAM latency
    localparam FC3_MULT             = 6'd32;
    localparam FC3_MULT_WAIT        = 6'd37;  // Wait cycle 1 for BRAM data
    localparam FC3_MULT_WAIT2       = 6'd40;  // Wait cycle 2 for BRAM data (if 2-cycle latency)
    localparam FC3_NEXT             = 6'd33;

    localparam DONE_STATE           = 6'd34;

    reg [5:0] state;

    // Iterators
    reg [4:0] f_idx;        // Filter index (max 16)
    reg [3:0] ch_idx;       // Channel index (max 6)
    reg [4:0] r, c;         // Row, column
    reg [2:0] kr, kc;       // Kernel row, col (0-4 for 5x5)
    reg [6:0] neuron_idx;   // FC neuron index (max 120)
    reg [3:0] class_idx;    // Output class (0-9)

    reg signed [31:0] acc;
    reg signed [31:0] temp_val;
    reg signed [31:0] pool_sum;
    reg [9:0] flat_idx;     // Flattened index for FC layers
    reg [2:0] pool_step;
    reg signed [31:0] max_score;

    // Address calculation helpers for current position
    wire signed [5:0] img_row_signed;
    wire signed [5:0] img_col_signed;
    wire [9:0] img_addr_calc;
    wire img_in_bounds;

    assign img_row_signed = $signed({1'b0, r}) + $signed({1'b0, kr}) - 2; // -2 for padding
    assign img_col_signed = $signed({1'b0, c}) + $signed({1'b0, kc}) - 2;
    assign img_in_bounds = (img_row_signed >= 0) && (img_row_signed < 28) &&
                           (img_col_signed >= 0) && (img_col_signed < 28);
    assign img_addr_calc = img_row_signed * 28 + img_col_signed;

    // Address calculation helpers for NEXT position (needed for L1_CONV pipelining)
    wire [2:0] next_kr_l1 = (kc == 4) ? kr + 1 : kr;
    wire [2:0] next_kc_l1 = (kc == 4) ? 3'd0 : kc + 1;
    wire signed [5:0] next_img_row = $signed({1'b0, r}) + $signed({1'b0, next_kr_l1}) - 2;
    wire signed [5:0] next_img_col = $signed({1'b0, c}) + $signed({1'b0, next_kc_l1}) - 2;
    wire next_img_in_bounds = (next_img_row >= 0) && (next_img_row < 28) &&
                              (next_img_col >= 0) && (next_img_col < 28);
    wire [9:0] next_img_addr = next_img_row * 28 + next_img_col;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            buf_a_wr_en <= 0;
            buf_b_wr_en <= 0;
            buf_c_wr_en <= 0;
            predicted_digit <= 0;
            img_addr <= 0;
            conv_w_addr <= 0; conv_b_addr <= 0;
            fc_w_addr <= 0; fc_b_addr <= 0;
            tanh_addr <= 0;
            buf_a_addr <= 0; buf_a_wr_data <= 0;
            buf_b_addr <= 0; buf_b_wr_data <= 0;
            buf_c_addr <= 0; buf_c_wr_data <= 0;
            class_score_0 <= 0; class_score_1 <= 0; class_score_2 <= 0;
            class_score_3 <= 0; class_score_4 <= 0; class_score_5 <= 0;
            class_score_6 <= 0; class_score_7 <= 0; class_score_8 <= 0;
            class_score_9 <= 0;
            f_idx <= 0; ch_idx <= 0; r <= 0; c <= 0; kr <= 0; kc <= 0;
            neuron_idx <= 0; class_idx <= 0;
            acc <= 0; temp_val <= 0; pool_sum <= 0;
            flat_idx <= 0; pool_step <= 0;
            max_score <= 32'h80000000;
        end else begin
            buf_a_wr_en <= 0;
            buf_b_wr_en <= 0;
            buf_c_wr_en <= 0;

            case (state)

                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= L1_LOAD_BIAS;
                        f_idx <= 0; r <= 0; c <= 0;
                        max_score <= 32'h80000000;
                    end
                end

                // =============================================================
                // LAYER 1 CONV (5x5, padding=2, 6 filters)
                // Input: 1x28x28, Output: 6x28x28
                // =============================================================

                L1_LOAD_BIAS: begin
                    conv_b_addr <= f_idx;
                    state <= L1_LOAD_BIAS_WAIT;
                end

                L1_LOAD_BIAS_WAIT: begin
                    acc <= $signed(conv_b_data);
                    kr <= 0; kc <= 0;
                    // Set up first pixel address (with padding consideration)
                    if (r >= 2 && c >= 2)
                        img_addr <= (r - 2) * 28 + (c - 2);
                    conv_w_addr <= f_idx * 25;  // 5x5 = 25 weights per filter
                    state <= L1_PREFETCH;
                end

                L1_PREFETCH: begin
                    state <= L1_CONV;
                end

                L1_CONV: begin
                    // Check if current pixel is in bounds (padding handling)
                    if (img_in_bounds) begin
                        acc <= acc + $signed(img_data) * $signed(conv_w_data);
                    end
                    // Otherwise add 0 (padding)

                    if (kr == 4 && kc == 4) begin
                        // Done with 5x5 kernel
                        state <= L1_TANH;
                    end else begin
                        if (kc == 4) begin
                            kc <= 0;
                            kr <= kr + 1;
                        end else begin
                            kc <= kc + 1;
                        end
                        // Calculate next addresses (use NEXT kr/kc, not current!)
                        conv_w_addr <= f_idx * 25 + next_kr_l1 * 5 + next_kc_l1;
                        // Image address for NEXT position (with padding check)
                        if (next_img_in_bounds)
                            img_addr <= next_img_addr;
                    end
                end

                L1_TANH: begin
                    // Shift and prepare for tanh LUT (SHIFT=10, input scale=127.0)
                    temp_val <= acc >>> 10;
                    // Set tanh address here to allow one cycle for LUT read
                    if ((acc >>> 10) < -128)
                        tanh_addr <= 8'd0;  // Index for -128
                    else if ((acc >>> 10) > 127)
                        tanh_addr <= 8'd255; // Index for 127
                    else
                        tanh_addr <= (acc >>> 10) + 8'd128; // Offset to 0-255
                    state <= L1_SAVE;
                end

                L1_SAVE: begin
                    // Apply tanh via LUT (tanh_data now valid from previous cycle)
                    // Store result
                    buf_a_addr <= f_idx * 784 + r * 28 + c;
                    buf_a_wr_data <= tanh_data;  // Tanh LUT output (ready now)
                    buf_a_wr_en <= 1;

                    // Move to next position
                    if (c == 27) begin
                        c <= 0;
                        if (r == 27) begin
                            r <= 0;
                            if (f_idx == 5) begin
                                // Done with Conv1, move to Pool1
                                state <= L1_POOL;
                                f_idx <= 0; r <= 0; c <= 0; pool_step <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L1_LOAD_BIAS;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L1_LOAD_BIAS;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L1_LOAD_BIAS;
                    end
                end

                // =============================================================
                // L1 AVERAGE POOL (2x2)
                // Input: 6x28x28, Output: 6x14x14
                // =============================================================

                L1_POOL: begin
                    case(pool_step)
                        0: begin
                            // Read first value
                            buf_a_addr <= f_idx*784 + (r*2)*28 + (c*2);
                            pool_step <= 1;
                        end
                        1: begin
                            pool_sum <= $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*784 + (r*2)*28 + (c*2+1);
                            pool_step <= 2;
                        end
                        2: begin
                            pool_sum <= pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*784 + (r*2+1)*28 + (c*2);
                            pool_step <= 3;
                        end
                        3: begin
                            pool_sum <= pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*784 + (r*2+1)*28 + (c*2+1);
                            pool_step <= 4;
                        end
                        4: begin
                            // Average = sum / 4 (arithmetic right shift by 2)
                            temp_val <= (pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data})) >>> 2;
                            state <= L1_POOL_CALC;
                        end
                    endcase
                end

                L1_POOL_CALC: begin
                    // Store pooled result
                    buf_b_addr <= f_idx*196 + r*14 + c;  // 14*14 = 196
                    buf_b_wr_data <= temp_val[7:0];
                    buf_b_wr_en <= 1;
                    pool_step <= 0;

                    // Advance to next pool position
                    if (c == 13) begin
                        c <= 0;
                        if (r == 13) begin
                            r <= 0;
                            if (f_idx == 5) begin
                                state <= L2_LOAD_BIAS;
                                f_idx <= 0; r <= 0; c <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L1_POOL;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L1_POOL;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L1_POOL;
                    end
                end

                // =============================================================
                // LAYER 2 CONV (5x5, 16 filters, 6 input channels)
                // Input: 6x14x14, Output: 16x10x10
                // =============================================================

                L2_LOAD_BIAS: begin
                    conv_b_addr <= 6 + f_idx;  // Offset by Conv1 biases
                    state <= L2_LOAD_BIAS_WAIT;
                end

                L2_LOAD_BIAS_WAIT: begin
                    acc <= $signed(conv_b_data);
                    ch_idx <= 0; kr <= 0; kc <= 0;
                    buf_b_addr <= 0 * 196 + (r + 0) * 14 + (c + 0);
                    // Conv2 weights start at offset 150 (Conv1 weights)
                    conv_w_addr <= 150 + f_idx * 150 + 0 * 25 + 0;  // 6*25=150 per filter
                    state <= L2_PREFETCH;
                end

                L2_PREFETCH: begin
                    state <= L2_CONV;
                end

                L2_CONV: begin
                    acc <= acc + $signed(buf_b_rd_data) * $signed(conv_w_data);

                    if (kr == 4 && kc == 4 && ch_idx == 5) begin
                        state <= L2_TANH;
                    end else begin
                        if (kc == 4) begin
                            kc <= 0;
                            if (kr == 4) begin
                                kr <= 0;
                                ch_idx <= ch_idx + 1;
                                buf_b_addr <= (ch_idx + 1) * 196 + (r + 0) * 14 + (c + 0);
                                conv_w_addr <= 150 + f_idx * 150 + (ch_idx + 1) * 25 + 0;
                            end else begin
                                kr <= kr + 1;
                                buf_b_addr <= ch_idx * 196 + (r + kr + 1) * 14 + (c + 0);
                                conv_w_addr <= 150 + f_idx * 150 + ch_idx * 25 + (kr + 1) * 5 + 0;
                            end
                        end else begin
                            kc <= kc + 1;
                            buf_b_addr <= ch_idx * 196 + (r + kr) * 14 + (c + kc + 1);
                            conv_w_addr <= 150 + f_idx * 150 + ch_idx * 25 + kr * 5 + (kc + 1);
                        end
                    end
                end

                L2_TANH: begin
                    // SHIFT=8, input scale=31.75 (post-pool, 127/4)
                    temp_val <= acc >>> 8;
                    // Set tanh address here to allow one cycle for LUT read
                    if ((acc >>> 8) < -128)
                        tanh_addr <= 8'd0;
                    else if ((acc >>> 8) > 127)
                        tanh_addr <= 8'd255;
                    else
                        tanh_addr <= (acc >>> 8) + 8'd128;
                    state <= L2_SAVE;
                end

                L2_SAVE: begin
                    // Apply tanh (tanh_data now valid from previous cycle)
                    buf_a_addr <= f_idx * 100 + r * 10 + c;  // 10*10 = 100
                    buf_a_wr_data <= tanh_data;
                    buf_a_wr_en <= 1;

                    if (c == 9) begin
                        c <= 0;
                        if (r == 9) begin
                            r <= 0;
                            if (f_idx == 15) begin
                                state <= L2_POOL;
                                f_idx <= 0; r <= 0; c <= 0; pool_step <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L2_LOAD_BIAS;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L2_LOAD_BIAS;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L2_LOAD_BIAS;
                    end
                end

                // =============================================================
                // L2 AVERAGE POOL (2x2)
                // Input: 16x10x10, Output: 16x5x5 = 400 features
                // =============================================================

                L2_POOL: begin
                    case(pool_step)
                        0: begin
                            buf_a_addr <= f_idx*100 + (r*2)*10 + (c*2);
                            pool_step <= 1;
                        end
                        1: begin
                            pool_sum <= $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*100 + (r*2)*10 + (c*2+1);
                            pool_step <= 2;
                        end
                        2: begin
                            pool_sum <= pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*100 + (r*2+1)*10 + (c*2);
                            pool_step <= 3;
                        end
                        3: begin
                            pool_sum <= pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data});
                            buf_a_addr <= f_idx*100 + (r*2+1)*10 + (c*2+1);
                            pool_step <= 4;
                        end
                        4: begin
                            temp_val <= (pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data})) >>> 2;
                            state <= L2_POOL_CALC;
                        end
                    endcase
                end

                L2_POOL_CALC: begin
                    // Store to Buffer C (400 features for FC1)
                    buf_c_addr <= f_idx*25 + r*5 + c;  // 5*5 = 25
                    buf_c_wr_data <= temp_val[7:0];
                    buf_c_wr_en <= 1;
                    pool_step <= 0;

                    if (c == 4) begin
                        c <= 0;
                        if (r == 4) begin
                            r <= 0;
                            if (f_idx == 15) begin
                                state <= FC1_LOAD_BIAS;
                                neuron_idx <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L2_POOL;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L2_POOL;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L2_POOL;
                    end
                end

                // =============================================================
                // FC1 LAYER (400 -> 120 with Tanh)
                // =============================================================

                FC1_LOAD_BIAS: begin
                    fc_b_addr <= neuron_idx;
                    state <= FC1_LOAD_BIAS_WAIT;
                end

                FC1_LOAD_BIAS_WAIT: begin
                    acc <= $signed(fc_b_data);
                    flat_idx <= 0;
                    buf_c_addr <= 0;
                    fc_w_addr <= neuron_idx * 400;
                    state <= FC1_PREFETCH;
                end

                FC1_PREFETCH: begin
                    // Wait cycle 1 for first BRAM data
                    state <= FC1_PREFETCH2;
                end

                FC1_PREFETCH2: begin
                    // Wait cycle 2 for first BRAM data (2-cycle latency)
                    state <= FC1_MULT;
                end

                FC1_MULT: begin
                    acc <= acc + $signed(buf_c_rd_data) * $signed(fc_w_data);

                    if (flat_idx == 399) begin
                        state <= FC1_TANH;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_c_addr <= flat_idx + 1;
                        fc_w_addr <= neuron_idx * 400 + (flat_idx + 1);
                        state <= FC1_MULT_WAIT;  // Wait for BRAM latency
                    end
                end

                FC1_MULT_WAIT: begin
                    // Wait cycle 1 for BRAM data
                    state <= FC1_MULT_WAIT2;
                end

                FC1_MULT_WAIT2: begin
                    // Wait cycle 2 for BRAM data (2-cycle BRAM latency)
                    state <= FC1_MULT;
                end

                FC1_TANH: begin
                    // SHIFT=9, input scale=31.75 (post-pool)
                    temp_val <= acc >>> 9;
                    // Set tanh address here to allow one cycle for LUT read
                    if ((acc >>> 9) < -128)
                        tanh_addr <= 8'd0;
                    else if ((acc >>> 9) > 127)
                        tanh_addr <= 8'd255;
                    else
                        tanh_addr <= (acc >>> 9) + 8'd128;
                    state <= FC1_SAVE;
                end

                FC1_SAVE: begin
                    // Apply tanh (tanh_data now valid from previous cycle)
                    // Store FC1 output in Buffer B (reuse)
                    buf_b_addr <= neuron_idx;
                    buf_b_wr_data <= tanh_data;
                    buf_b_wr_en <= 1;

                    if (neuron_idx == 119) begin
                        state <= FC2_LOAD_BIAS;
                        neuron_idx <= 0;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                        state <= FC1_LOAD_BIAS;
                    end
                end

                // =============================================================
                // FC2 LAYER (120 -> 84 with Tanh)
                // =============================================================

                FC2_LOAD_BIAS: begin
                    fc_b_addr <= 120 + neuron_idx;  // Offset by FC1 biases
                    state <= FC2_LOAD_BIAS_WAIT;
                end

                FC2_LOAD_BIAS_WAIT: begin
                    acc <= $signed(fc_b_data);
                    flat_idx <= 0;
                    buf_b_addr <= 0;
                    fc_w_addr <= 48000 + neuron_idx * 120;  // FC2 weights start after FC1
                    state <= FC2_PREFETCH;
                end

                FC2_PREFETCH: begin
                    // Wait cycle 1 for first BRAM data
                    state <= FC2_PREFETCH2;
                end

                FC2_PREFETCH2: begin
                    // Wait cycle 2 for first BRAM data (2-cycle latency)
                    state <= FC2_MULT;
                end

                FC2_MULT: begin
                    acc <= acc + $signed(buf_b_rd_data) * $signed(fc_w_data);

                    if (flat_idx == 119) begin
                        state <= FC2_TANH;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_b_addr <= flat_idx + 1;
                        fc_w_addr <= 48000 + neuron_idx * 120 + (flat_idx + 1);
                        state <= FC2_MULT_WAIT;  // Wait for BRAM latency
                    end
                end

                FC2_MULT_WAIT: begin
                    // Wait cycle 1 for BRAM data
                    state <= FC2_MULT_WAIT2;
                end

                FC2_MULT_WAIT2: begin
                    // Wait cycle 2 for BRAM data (2-cycle BRAM latency)
                    state <= FC2_MULT;
                end

                FC2_TANH: begin
                    // SHIFT=10, input scale=127.0 (post-tanh)
                    temp_val <= acc >>> 10;
                    // Set tanh address here to allow one cycle for LUT read
                    if ((acc >>> 10) < -128)
                        tanh_addr <= 8'd0;
                    else if ((acc >>> 10) > 127)
                        tanh_addr <= 8'd255;
                    else
                        tanh_addr <= (acc >>> 10) + 8'd128;
                    state <= FC2_SAVE;
                end

                FC2_SAVE: begin
                    // Apply tanh (tanh_data now valid from previous cycle)
                    // Store FC2 output in Buffer C (reuse)
                    buf_c_addr <= neuron_idx;
                    buf_c_wr_data <= tanh_data;
                    buf_c_wr_en <= 1;

                    if (neuron_idx == 83) begin
                        state <= FC3_LOAD_BIAS;
                        class_idx <= 0;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                        state <= FC2_LOAD_BIAS;
                    end
                end

                // =============================================================
                // FC3 LAYER (84 -> 10, no activation)
                // =============================================================

                FC3_LOAD_BIAS: begin
                    fc_b_addr <= 204 + class_idx;  // 120 + 84 = 204
                    state <= FC3_LOAD_BIAS_WAIT;
                end

                FC3_LOAD_BIAS_WAIT: begin
                    acc <= $signed(fc_b_data);
                    flat_idx <= 0;
                    buf_c_addr <= 0;
                    // FC3 weights start after FC1 + FC2
                    fc_w_addr <= 48000 + 10080 + class_idx * 84;
                    state <= FC3_PREFETCH;
                end

                FC3_PREFETCH: begin
                    // Wait cycle 1 for first BRAM data
                    state <= FC3_PREFETCH2;
                end

                FC3_PREFETCH2: begin
                    // Wait cycle 2 for first BRAM data (2-cycle latency)
                    state <= FC3_MULT;
                end

                FC3_MULT: begin
                    acc <= acc + $signed(buf_c_rd_data) * $signed(fc_w_data);

                    if (flat_idx == 83) begin
                        state <= FC3_NEXT;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_c_addr <= flat_idx + 1;
                        fc_w_addr <= 48000 + 10080 + class_idx * 84 + (flat_idx + 1);
                        state <= FC3_MULT_WAIT;  // Wait for BRAM latency
                    end
                end

                FC3_MULT_WAIT: begin
                    // Wait cycle 1 for BRAM data
                    state <= FC3_MULT_WAIT2;
                end

                FC3_MULT_WAIT2: begin
                    // Wait cycle 2 for BRAM data (2-cycle BRAM latency)
                    state <= FC3_MULT;
                end

                FC3_NEXT: begin
                    // Store class score
                    case (class_idx)
                        0: class_score_0 <= acc;
                        1: class_score_1 <= acc;
                        2: class_score_2 <= acc;
                        3: class_score_3 <= acc;
                        4: class_score_4 <= acc;
                        5: class_score_5 <= acc;
                        6: class_score_6 <= acc;
                        7: class_score_7 <= acc;
                        8: class_score_8 <= acc;
                        9: class_score_9 <= acc;
                    endcase

                    // Track argmax
                    if (class_idx == 0 || acc > max_score) begin
                        max_score <= acc;
                        predicted_digit <= class_idx;
                    end

                    if (class_idx == 9)
                        state <= DONE_STATE;
                    else begin
                        class_idx <= class_idx + 1;
                        state <= FC3_LOAD_BIAS;
                    end
                end

                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

    // Debug
    // synthesis translate_off
    initial begin
        $display("");
        $display("==========================================================");
        $display("[INFERENCE] LeNet-5 Version 1.0");
        $display("[INFERENCE] Conv: 6@5x5 -> Pool -> 16@5x5 -> Pool");
        $display("[INFERENCE] FC: 400 -> 120 -> 84 -> 10");
        $display("[INFERENCE] Activation: Tanh (LUT-based)");
        $display("==========================================================");
        $display("");
    end
    // synthesis translate_on

endmodule
