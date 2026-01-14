`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 FC1 Layer Testbench
================================================================================
Tests ONLY the FC1 layer: 400 -> 120

Loads:
- Pool2 output (layer4_pool2.mem) as input (400 features)
- FC1 weights/biases
- Tanh LUT
- Golden reference (layer5_fc1.mem)

Compares:
- FPGA Buffer B output vs Python golden FC1 output
================================================================================
*/

module tb_layer_fc1;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Buffer C (input: 400 bytes)
    reg [7:0] buf_c_ram [0:399];
    reg [8:0] buf_c_addr;
    wire signed [7:0] buf_c_rd_data;
    assign buf_c_rd_data = buf_c_ram[buf_c_addr];
    
    // FC1 Weights RAM (120*400 = 48000 weights)
    reg [7:0] fc1_weights_ram [0:47999];
    reg [15:0] fc_w_addr;
    wire signed [7:0] fc_w_data;
    assign fc_w_data = fc1_weights_ram[fc_w_addr];
    
    // FC1 Biases RAM (120 biases)
    reg [31:0] fc1_biases_ram [0:119];
    reg [6:0] fc_b_addr;
    wire signed [31:0] fc_b_data;
    assign fc_b_data = fc1_biases_ram[fc_b_addr];
    
    // Tanh LUT
    reg [7:0] tanh_lut_ram [0:255];
    reg [7:0] tanh_addr;
    wire signed [7:0] tanh_data;
    assign tanh_data = tanh_lut_ram[tanh_addr];
    
    // Buffer B (output: 120 bytes)
    reg [7:0] buf_b_ram [0:119];
    reg [6:0] buf_b_addr;
    reg [7:0] buf_b_wr_data;
    reg buf_b_wr_en;
    wire signed [7:0] buf_b_rd_data;
    
    always @(posedge clk) begin
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    // Golden reference
    reg [7:0] golden_fc1 [0:119];
    
    // =========================================================================
    // FC1 FSM
    // =========================================================================
    localparam IDLE             = 3'd0;
    localparam LOAD_BIAS        = 3'd1;
    localparam LOAD_BIAS_WAIT   = 3'd2;
    localparam MULT             = 3'd3;
    localparam TANH             = 3'd4;
    localparam SAVE             = 3'd5;
    localparam DONE             = 3'd6;
    
    reg [2:0] state;
    reg [6:0] neuron_idx;       // Neuron index (0-119)
    reg [9:0] flat_idx;         // Input index (0-399)
    reg signed [31:0] acc;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done_manual <= 0;
            neuron_idx <= 0;
            flat_idx <= 0;
            acc <= 0;
            buf_b_wr_en <= 0;
            buf_c_addr <= 0;
            fc_w_addr <= 0;
            fc_b_addr <= 0;
            tanh_addr <= 0;
        end else begin
            buf_b_wr_en <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LOAD_BIAS;
                        neuron_idx <= 0;
                    end
                end
                
                LOAD_BIAS: begin
                    fc_b_addr <= neuron_idx;
                    state <= LOAD_BIAS_WAIT;
                end
                
                LOAD_BIAS_WAIT: begin
                    acc <= $signed(fc_b_data);
                    flat_idx <= 0;
                    buf_c_addr <= 0;
                    fc_w_addr <= neuron_idx * 400;
                    state <= MULT;
                end
                
                MULT: begin
                    acc <= acc + $signed(buf_c_rd_data) * $signed(fc_w_data);
                    
                    if (flat_idx == 399) begin
                        state <= TANH;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_c_addr <= flat_idx + 1;
                        fc_w_addr <= neuron_idx * 400 + (flat_idx + 1);
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
                    buf_b_addr <= neuron_idx;
                    buf_b_wr_data <= tanh_data;
                    buf_b_wr_en <= 1;
                    
                    if (neuron_idx == 119) begin
                        state <= DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
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
        $display("LeNet-5 FC1 Layer Testbench");
        $display("Testing: 400 -> 120 (FC + Tanh + Shift)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer4_pool2.mem", buf_c_ram);
        $readmemh("fc1_weights.mem", fc1_weights_ram);
        $readmemh("fc1_biases.mem", fc1_biases_ram);
        $readmemh("tanh_lut.mem", tanh_lut_ram);
        $readmemh("layer5_fc1.mem", golden_fc1);
        $display("  ✓ Pool2 output loaded (400 bytes)");
        $display("  ✓ FC1 weights loaded (48000 bytes)");
        $display("  ✓ FC1 biases loaded (120 entries)");
        $display("  ✓ Tanh LUT loaded (256 entries)");
        $display("  ✓ Golden FC1 output loaded (120 bytes)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start inference
        $display("\nStarting FC1 layer...");
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
        
        $display("  ✓ FC1 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 120; i = i + 1) begin
            fpga_val = buf_b_ram[i];
            gold_val = golden_fc1[i];
            
            // Sign extend for comparison
            if (fpga_val > 127) fpga_val = fpga_val - 256;
            if (gold_val > 127) gold_val = gold_val - 256;
            
            diff = fpga_val - gold_val;
            if (diff < 0) diff = -diff;
            
            if (diff > max_diff) max_diff = diff;
            
            if (fpga_val !== gold_val) begin
                if (mismatches < 10) begin
                    $display("  [MISMATCH] Neuron %0d: FPGA=%0d, Golden=%0d, Diff=%0d",
                             i, fpga_val, gold_val, fpga_val - gold_val);
                end
                mismatches = mismatches + 1;
            end
        end
        
        // Summary
        $display("\n" + {"="}*80);
        $display("RESULTS");
        $display({"="}*80);
        $display("Total neurons: 120");
        $display("Exact matches: %0d (%.2f%%)", 120 - mismatches,
                 100.0 * (120 - mismatches) / 120);
        $display("Mismatches:    %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] FC1 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] FC1 layer has mismatches.");
            $display("This indicates a quantization or implementation issue in FC1.");
        end
        
        $display({"="}*80 + "\n");
        $finish;
    end
    
    // Timeout
    initial begin
        #100_000_000;  // 100ms timeout
        $display("\n[ERROR] Timeout!");
        $finish;
    end

endmodule
