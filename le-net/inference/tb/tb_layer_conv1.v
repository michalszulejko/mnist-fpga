`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Conv1 Layer Testbench
================================================================================
Tests ONLY the Conv1 layer: 1x28x28 -> 6x28x28

Loads:
- Input image (layer0_input.mem)
- Conv1 weights/biases
- Tanh LUT
- Golden reference (layer1_conv1.mem)

Compares:
- FPGA Buffer A output vs Python golden Conv1 output
- Reports mismatches per filter, per pixel
================================================================================
*/

module tb_layer_conv1;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Image RAM (28x28 = 784 bytes)
    reg [7:0] image_ram [0:783];
    reg [9:0] img_addr;
    wire signed [7:0] img_data;
    assign img_data = image_ram[img_addr];
    
    // Conv1 Weights RAM (6*1*5*5 = 150 weights)
    reg [7:0] conv1_weights_ram [0:149];
    reg [7:0] conv_w_addr;
    wire signed [7:0] conv_w_data;
    assign conv_w_data = conv1_weights_ram[conv_w_addr];
    
    // Conv1 Biases RAM (6 biases)
    reg [31:0] conv1_biases_ram [0:5];
    reg [2:0] conv_b_addr;
    wire signed [31:0] conv_b_data;
    assign conv_b_data = conv1_biases_ram[conv_b_addr];
    
    // Tanh LUT
    reg [7:0] tanh_lut_ram [0:255];
    reg [7:0] tanh_addr;
    wire signed [7:0] tanh_data;
    assign tanh_data = tanh_lut_ram[tanh_addr];
    
    // Buffer A (6*28*28 = 4704 bytes)
    reg [7:0] buf_a_ram [0:4703];
    reg [12:0] buf_a_addr;
    reg [7:0] buf_a_wr_data;
    reg buf_a_wr_en;
    wire signed [7:0] buf_a_rd_data;
    
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    
    // Golden reference
    reg [7:0] golden_conv1 [0:4703];
    
    // =========================================================================
    // Conv1 FSM (Simplified from inference.v)
    // =========================================================================
    localparam IDLE             = 3'd0;
    localparam LOAD_BIAS        = 3'd1;
    localparam LOAD_BIAS_WAIT   = 3'd2;
    localparam CONV             = 3'd3;
    localparam TANH             = 3'd4;
    localparam SAVE             = 3'd5;
    localparam DONE             = 3'd6;
    
    reg [2:0] state;
    reg [2:0] f_idx;        // Filter index (0-5)
    reg [4:0] r, c;         // Row, column (0-27)
    reg [2:0] kr, kc;       // Kernel row, col (0-4)
    reg signed [31:0] acc;
    
    // Temporary variables for next address calculation
    reg [2:0] next_kr, next_kc;
    reg signed [5:0] next_img_row, next_img_col;
    reg next_in_bounds;
    
    // Address calculation for padding
    wire signed [5:0] img_row_signed;
    wire signed [5:0] img_col_signed;
    wire [9:0] img_addr_calc;
    wire img_in_bounds;
    
    assign img_row_signed = $signed({1'b0, r}) + $signed({1'b0, kr}) - 2;
    assign img_col_signed = $signed({1'b0, c}) + $signed({1'b0, kc}) - 2;
    assign img_in_bounds = (img_row_signed >= 0) && (img_row_signed < 28) &&
                           (img_col_signed >= 0) && (img_col_signed < 28);
    assign img_addr_calc = img_row_signed * 28 + img_col_signed;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done_manual <= 0;
            f_idx <= 0;
            r <= 0;
            c <= 0;
            kr <= 0;
            kc <= 0;
            acc <= 0;
            buf_a_wr_en <= 0;
            img_addr <= 0;
            conv_w_addr <= 0;
            conv_b_addr <= 0;
            tanh_addr <= 0;
        end else begin
            buf_a_wr_en <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LOAD_BIAS;
                        f_idx <= 0;
                        r <= 0;
                        c <= 0;
                    end
                end
                
                LOAD_BIAS: begin
                    conv_b_addr <= f_idx;
                    state <= LOAD_BIAS_WAIT;
                end
                
                LOAD_BIAS_WAIT: begin
                    acc <= $signed(conv_b_data);
                    kr <= 0;
                    kc <= 0;
                    if (r >= 2 && c >= 2)
                        img_addr <= (r - 2) * 28 + (c - 2);
                    conv_w_addr <= f_idx * 25;
                    state <= CONV;
                end
                
                CONV: begin
                    if (img_in_bounds) begin
                        acc <= acc + $signed(img_data) * $signed(conv_w_data);
                    end
                    
                    if (kr == 4 && kc == 4) begin
                        state <= TANH;
                    end else begin
                        // Calculate next kr, kc
                        if (kc == 4) begin
                            kc <= 0;
                            kr <= kr + 1;
                            next_kr = kr + 1;
                            next_kc = 0;
                        end else begin
                            kc <= kc + 1;
                            next_kr = kr;
                            next_kc = kc + 1;
                        end
                        
                        // Calculate next weight address
                        conv_w_addr <= f_idx * 25 + next_kr * 5 + next_kc;
                        
                        // Calculate next image address using next_kr and next_kc
                        next_img_row = $signed({1'b0, r}) + $signed({1'b0, next_kr}) - 2;
                        next_img_col = $signed({1'b0, c}) + $signed({1'b0, next_kc}) - 2;
                        next_in_bounds = (next_img_row >= 0) && (next_img_row < 28) &&
                                        (next_img_col >= 0) && (next_img_col < 28);
                        if (next_in_bounds)
                            img_addr <= next_img_row * 28 + next_img_col;
                    end
                end
                
                TANH: begin
                    if ((acc >>> 9) < -128)
                        tanh_addr <= 8'd0;
                    else if ((acc >>> 9) > 127)
                        tanh_addr <= 8'd255;
                    else
                        tanh_addr <= (acc >>> 9) + 8'd128;
                    state <= SAVE;
                end
                
                SAVE: begin
                    buf_a_addr <= f_idx * 784 + r * 28 + c;
                    buf_a_wr_data <= tanh_data;
                    buf_a_wr_en <= 1;
                    
                    if (c == 27) begin
                        c <= 0;
                        if (r == 27) begin
                            r <= 0;
                            if (f_idx == 5) begin
                                state <= DONE;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= LOAD_BIAS;
                            end
                        end else begin
                            r <= r + 1;
                            state <= LOAD_BIAS;
                        end
                    end else begin
                        c <= c + 1;
                        state <= LOAD_BIAS;
                    end
                end
                
                DONE: begin
                    done_manual <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    // =========================================================================
    // Clock Generation
    // =========================================================================
    always #5 clk = ~clk;
    
    // =========================================================================
    // Test Procedure
    // =========================================================================
    integer i, mismatches, max_diff, diff;
    integer fpga_val, gold_val;
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        
        $display("\n" + {"="}*80);
        $display("LeNet-5 Conv1 Layer Testbench");
        $display("Testing: 1x28x28 -> 6x28x28 (Conv + Tanh + Shift)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer0_input.mem", image_ram);
        $readmemh("conv1_weights.mem", conv1_weights_ram);
        $readmemh("conv1_biases.mem", conv1_biases_ram);
        $readmemh("tanh_lut.mem", tanh_lut_ram);
        $readmemh("layer1_conv1.mem", golden_conv1);
        $display("  ✓ Input image loaded (784 bytes)");
        $display("  ✓ Conv1 weights loaded (150 bytes)");
        $display("  ✓ Conv1 biases loaded (6 entries)");
        $display("  ✓ Tanh LUT loaded (256 entries)");
        $display("  ✓ Golden Conv1 output loaded (4704 bytes)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start inference
        $display("\nStarting Conv1 layer...");
        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;
        
        // Wait for completion
        wait(done_manual);
        @(posedge clk);
        #100;
        
        $display("  ✓ Conv1 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 4704; i = i + 1) begin
            fpga_val = buf_a_ram[i];
            gold_val = golden_conv1[i];
            
            // Sign extend for comparison
            if (fpga_val > 127) fpga_val = fpga_val - 256;
            if (gold_val > 127) gold_val = gold_val - 256;
            
            diff = fpga_val - gold_val;
            if (diff < 0) diff = -diff;
            
            if (diff > max_diff) max_diff = diff;
            
            if (fpga_val !== gold_val) begin
                if (mismatches < 10) begin
                    $display("  [MISMATCH] Index %0d: FPGA=%0d, Golden=%0d, Diff=%0d",
                             i, fpga_val, gold_val, fpga_val - gold_val);
                end
                mismatches = mismatches + 1;
            end
        end
        
        // Summary
        $display("\n" + {"="}*80);
        $display("RESULTS");
        $display({"="}*80);
        $display("Total elements: 4704 (6 filters × 28 × 28)");
        $display("Exact matches:  %0d (%.2f%%)", 4704 - mismatches,
                 100.0 * (4704 - mismatches) / 4704);
        $display("Mismatches:     %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] Conv1 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] Conv1 layer has mismatches.");
            $display("This indicates a quantization or implementation issue in Conv1.");
        end
        
        $display({"="}*80 + "\n");
        $finish;
    end
    
    // Timeout
    initial begin
        #10_000_000;  // 10ms timeout
        $display("\n[ERROR] Timeout!");
        $finish;
    end

endmodule
