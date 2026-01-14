`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 FC3 Layer Testbench
================================================================================
Tests ONLY the FC3 layer: 84 -> 10 (no activation)

Loads:
- FC2 output (layer6_fc2.mem) as input
- FC3 weights/biases
- Golden reference (layer7_fc3.mem)

Compares:
- FPGA class scores vs Python golden FC3 output
================================================================================
*/

module tb_layer_fc3;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Buffer C (input: 84 bytes)
    reg [7:0] buf_c_ram [0:83];
    reg [6:0] buf_c_addr;
    wire signed [7:0] buf_c_rd_data;
    assign buf_c_rd_data = buf_c_ram[buf_c_addr];
    
    // FC3 Weights RAM (10*84 = 840 weights)
    reg [7:0] fc3_weights_ram [0:839];
    reg [9:0] fc_w_addr;
    wire signed [7:0] fc_w_data;
    assign fc_w_data = fc3_weights_ram[fc_w_addr];
    
    // FC3 Biases RAM (10 biases)
    reg [31:0] fc3_biases_ram [0:9];
    reg [3:0] fc_b_addr;
    wire signed [31:0] fc_b_data;
    assign fc_b_data = fc3_biases_ram[fc_b_addr];
    
    // Class scores output
    reg signed [31:0] class_scores [0:9];
    
    // Golden reference
    reg [31:0] golden_fc3 [0:9];
    
    // =========================================================================
    // FC3 FSM
    // =========================================================================
    localparam IDLE             = 3'd0;
    localparam LOAD_BIAS        = 3'd1;
    localparam LOAD_BIAS_WAIT   = 3'd2;
    localparam MULT             = 3'd3;
    localparam SAVE             = 3'd4;
    localparam DONE             = 3'd5;
    
    reg [2:0] state;
    reg [3:0] class_idx;        // Class index (0-9)
    reg [6:0] flat_idx;         // Input index (0-83)
    reg signed [31:0] acc;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done_manual <= 0;
            class_idx <= 0;
            flat_idx <= 0;
            acc <= 0;
            buf_c_addr <= 0;
            fc_w_addr <= 0;
            fc_b_addr <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LOAD_BIAS;
                        class_idx <= 0;
                    end
                end
                
                LOAD_BIAS: begin
                    fc_b_addr <= class_idx;
                    state <= LOAD_BIAS_WAIT;
                end
                
                LOAD_BIAS_WAIT: begin
                    acc <= $signed(fc_b_data);
                    flat_idx <= 0;
                    buf_c_addr <= 0;
                    fc_w_addr <= class_idx * 84;
                    state <= MULT;
                end
                
                MULT: begin
                    acc <= acc + $signed(buf_c_rd_data) * $signed(fc_w_data);
                    
                    if (flat_idx == 83) begin
                        state <= SAVE;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_c_addr <= flat_idx + 1;
                        fc_w_addr <= class_idx * 84 + (flat_idx + 1);
                    end
                end
                
                SAVE: begin
                    class_scores[class_idx] <= acc;
                    
                    if (class_idx == 9) begin
                        state <= DONE;
                    end else begin
                        class_idx <= class_idx + 1;
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
    integer i, mismatches, max_diff;
    integer fpga_val, gold_val, diff;
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        
        $display("\n" + {"="}*80);
        $display("LeNet-5 FC3 Layer Testbench");
        $display("Testing: 84 -> 10 (FC, no activation)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer6_fc2.mem", buf_c_ram);
        $readmemh("fc3_weights.mem", fc3_weights_ram);
        $readmemh("fc3_biases.mem", fc3_biases_ram);
        $readmemh("layer7_fc3.mem", golden_fc3);
        $display("  ✓ FC2 output loaded (84 bytes)");
        $display("  ✓ FC3 weights loaded (840 bytes)");
        $display("  ✓ FC3 biases loaded (10 entries)");
        $display("  ✓ Golden FC3 output loaded (10 int32s)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start inference
        $display("\nStarting FC3 layer...");
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
        
        $display("  ✓ FC3 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        $display("  Class |      FPGA      |     Golden     | Difference");
        $display("  ------|----------------|----------------|------------");
        
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 10; i = i + 1) begin
            fpga_val = class_scores[i];
            gold_val = $signed(golden_fc3[i]);
            
            diff = fpga_val - gold_val;
            if (diff < 0) diff = -diff;
            
            if (diff > max_diff) max_diff = diff;
            
            if (fpga_val !== gold_val) begin
                mismatches = mismatches + 1;
                $display("    %0d   | %12d | %12d | %10d [MM]", i, fpga_val, gold_val, fpga_val - gold_val);
            end else begin
                $display("    %0d   | %12d | %12d | %10d [OK]", i, fpga_val, gold_val, 0);
            end
        end
        
        // Summary
        $display("\n" + {"="}*80);
        $display("RESULTS");
        $display({"="}*80);
        $display("Total classes: 10");
        $display("Exact matches: %0d", 10 - mismatches);
        $display("Mismatches:    %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] FC3 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] FC3 layer has mismatches.");
            $display("This indicates a quantization or accumulation issue in FC3.");
            $display("Since FC3 has no activation/shift, mismatches suggest:");
            $display("  1. Accumulated error from previous layers, OR");
            $display("  2. Bias pre-scaling mismatch");
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
