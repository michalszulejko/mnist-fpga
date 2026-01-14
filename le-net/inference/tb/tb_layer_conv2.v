`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Conv2 Layer Testbench
================================================================================
Tests ONLY the Conv2 layer: 6x14x14 -> 16x10x10

Loads:
- Pool1 output (layer2_pool1.mem) as input
- Conv2 weights/biases
- Tanh LUT
- Golden reference (layer3_conv2.mem)

Compares:
- FPGA Buffer A output vs Python golden Conv2 output
================================================================================
*/

module tb_layer_conv2;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Buffer B (input: 6*14*14 = 1176 bytes)
    reg [7:0] buf_b_ram [0:1175];
    reg [10:0] buf_b_addr;
    wire signed [7:0] buf_b_rd_data;
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    // Conv2 Weights RAM (16*6*5*5 = 2400 weights)
    reg [7:0] conv2_weights_ram [0:2399];
    reg [11:0] conv_w_addr;
    wire signed [7:0] conv_w_data;
    assign conv_w_data = conv2_weights_ram[conv_w_addr];
    
    // Conv2 Biases RAM (16 biases)
    reg [31:0] conv2_biases_ram [0:15];
    reg [3:0] conv_b_addr;
    wire signed [31:0] conv_b_data;
    assign conv_b_data = conv2_biases_ram[conv_b_addr];
    
    // Tanh LUT
    reg [7:0] tanh_lut_ram [0:255];
    reg [7:0] tanh_addr;
    wire signed [7:0] tanh_data;
    assign tanh_data = tanh_lut_ram[tanh_addr];
    
    // Buffer A (output: 16*10*10 = 1600 bytes)
    reg [7:0] buf_a_ram [0:1599];
    reg [10:0] buf_a_addr;
    reg [7:0] buf_a_wr_data;
    reg buf_a_wr_en;
    wire signed [7:0] buf_a_rd_data;
    
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    
    // Golden reference
    reg [7:0] golden_conv2 [0:1599];
    
    // =========================================================================
    // Conv2 FSM
    // =========================================================================
    localparam IDLE             = 3'd0;
    localparam LOAD_BIAS        = 3'd1;
    localparam LOAD_BIAS_WAIT   = 3'd2;
    localparam CONV             = 3'd3;
    localparam TANH             = 3'd4;
    localparam SAVE             = 3'd5;
    localparam DONE             = 3'd6;
    
    reg [2:0] state;
    reg [4:0] f_idx;        // Filter index (0-15)
    reg [3:0] ch_idx;       // Channel index (0-5)
    reg [3:0] r, c;         // Row, column (0-9)
    reg [2:0] kr, kc;       // Kernel row, col (0-4)
    reg signed [31:0] acc;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done_manual <= 0;
            f_idx <= 0;
            ch_idx <= 0;
            r <= 0;
            c <= 0;
            kr <= 0;
            kc <= 0;
            acc <= 0;
            buf_a_wr_en <= 0;
            buf_b_addr <= 0;
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
                    ch_idx <= 0;
                    kr <= 0;
                    kc <= 0;
                    buf_b_addr <= 0 * 196 + (r + 0) * 14 + (c + 0);
                    conv_w_addr <= f_idx * 150 + 0 * 25 + 0;
                    state <= CONV;
                end
                
                CONV: begin
                    acc <= acc + $signed(buf_b_rd_data) * $signed(conv_w_data);
                    
                    if (kr == 4 && kc == 4 && ch_idx == 5) begin
                        state <= TANH;
                    end else begin
                        if (kc == 4) begin
                            kc <= 0;
                            if (kr == 4) begin
                                kr <= 0;
                                ch_idx <= ch_idx + 1;
                                buf_b_addr <= (ch_idx + 1) * 196 + (r + 0) * 14 + (c + 0);
                                conv_w_addr <= f_idx * 150 + (ch_idx + 1) * 25 + 0;
                            end else begin
                                kr <= kr + 1;
                                buf_b_addr <= ch_idx * 196 + (r + kr + 1) * 14 + (c + 0);
                                conv_w_addr <= f_idx * 150 + ch_idx * 25 + (kr + 1) * 5 + 0;
                            end
                        end else begin
                            kc <= kc + 1;
                            buf_b_addr <= ch_idx * 196 + (r + kr) * 14 + (c + kc + 1);
                            conv_w_addr <= f_idx * 150 + ch_idx * 25 + kr * 5 + (kc + 1);
                        end
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
                    buf_a_addr <= f_idx * 100 + r * 10 + c;
                    buf_a_wr_data <= tanh_data;
                    buf_a_wr_en <= 1;
                    
                    if (c == 9) begin
                        c <= 0;
                        if (r == 9) begin
                            r <= 0;
                            if (f_idx == 15) begin
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
        $display("LeNet-5 Conv2 Layer Testbench");
        $display("Testing: 6x14x14 -> 16x10x10 (Conv + Tanh + Shift)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer2_pool1.mem", buf_b_ram);
        $readmemh("conv2_weights.mem", conv2_weights_ram);
        $readmemh("conv2_biases.mem", conv2_biases_ram);
        $readmemh("tanh_lut.mem", tanh_lut_ram);
        $readmemh("layer3_conv2.mem", golden_conv2);
        $display("  ✓ Pool1 output loaded (1176 bytes)");
        $display("  ✓ Conv2 weights loaded (2400 bytes)");
        $display("  ✓ Conv2 biases loaded (16 entries)");
        $display("  ✓ Tanh LUT loaded (256 entries)");
        $display("  ✓ Golden Conv2 output loaded (1600 bytes)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start inference
        $display("\nStarting Conv2 layer...");
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
        
        $display("  ✓ Conv2 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 1600; i = i + 1) begin
            fpga_val = buf_a_ram[i];
            gold_val = golden_conv2[i];
            
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
        $display("Total elements: 1600 (16 filters × 10 × 10)");
        $display("Exact matches:  %0d (%.2f%%)", 1600 - mismatches,
                 100.0 * (1600 - mismatches) / 1600);
        $display("Mismatches:     %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] Conv2 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] Conv2 layer has mismatches.");
            $display("This indicates a quantization or implementation issue in Conv2.");
        end
        
        $display({"="}*80 + "\n");
        $finish;
    end
    
    // Timeout
    initial begin
        #50_000_000;  // 50ms timeout
        $display("\n[ERROR] Timeout!");
        $finish;
    end

endmodule
