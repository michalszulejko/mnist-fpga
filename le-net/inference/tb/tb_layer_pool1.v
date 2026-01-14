`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Pool1 Layer Testbench
================================================================================
Tests ONLY the Pool1 layer: 6x28x28 -> 6x14x14 (2x2 average pooling)

Loads:
- Conv1 output (layer1_conv1.mem) as input
- Golden reference (layer2_pool1.mem)

Compares:
- FPGA Buffer B output vs Python golden Pool1 output
================================================================================
*/

module tb_layer_pool1;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Buffer A (input: 6*28*28 = 4704 bytes)
    reg [7:0] buf_a_ram [0:4703];
    reg [12:0] buf_a_addr;
    wire signed [7:0] buf_a_rd_data;
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    
    // Buffer B (output: 6*14*14 = 1176 bytes)
    reg [7:0] buf_b_ram [0:1175];
    reg [10:0] buf_b_addr;
    reg [7:0] buf_b_wr_data;
    reg buf_b_wr_en;
    wire signed [7:0] buf_b_rd_data;
    
    always @(posedge clk) begin
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    // Golden reference
    reg [7:0] golden_pool1 [0:1175];
    
    // =========================================================================
    // Pool1 FSM
    // =========================================================================
    localparam IDLE        = 2'd0;
    localparam POOL        = 2'd1;
    localparam POOL_CALC   = 2'd2;
    localparam DONE        = 2'd3;
    
    reg [1:0] state;
    reg [2:0] f_idx;        // Filter index (0-5)
    reg [3:0] r, c;         // Output position (0-13)
    reg [2:0] pool_step;
    reg signed [31:0] pool_sum;
    reg signed [31:0] temp_val;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done_manual <= 0;
            f_idx <= 0;
            r <= 0;
            c <= 0;
            pool_step <= 0;
            pool_sum <= 0;
            buf_b_wr_en <= 0;
        end else begin
            buf_b_wr_en <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= POOL;
                        f_idx <= 0;
                        r <= 0;
                        c <= 0;
                        pool_step <= 0;
                    end
                end
                
                POOL: begin
                    case(pool_step)
                        0: begin
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
                            temp_val <= (pool_sum + $signed({{24{buf_a_rd_data[7]}}, buf_a_rd_data})) >>> 2;
                            state <= POOL_CALC;
                        end
                    endcase
                end
                
                POOL_CALC: begin
                    buf_b_addr <= f_idx*196 + r*14 + c;
                    buf_b_wr_data <= temp_val[7:0];
                    buf_b_wr_en <= 1;
                    pool_step <= 0;
                    
                    if (c == 13) begin
                        c <= 0;
                        if (r == 13) begin
                            r <= 0;
                            if (f_idx == 5) begin
                                state <= DONE;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= POOL;
                            end
                        end else begin
                            r <= r + 1;
                            state <= POOL;
                        end
                    end else begin
                        c <= c + 1;
                        state <= POOL;
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
        $display("LeNet-5 Pool1 Layer Testbench");
        $display("Testing: 6x28x28 -> 6x14x14 (2x2 Average Pool)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer1_conv1.mem", buf_a_ram);
        $readmemh("layer2_pool1.mem", golden_pool1);
        $display("  ✓ Conv1 output loaded (4704 bytes)");
        $display("  ✓ Golden Pool1 output loaded (1176 bytes)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start pooling
        $display("\nStarting Pool1 layer...");
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
        
        $display("  ✓ Pool1 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 1176; i = i + 1) begin
            fpga_val = buf_b_ram[i];
            gold_val = golden_pool1[i];
            
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
        $display("Total elements: 1176 (6 filters × 14 × 14)");
        $display("Exact matches:  %0d (%.2f%%)", 1176 - mismatches,
                 100.0 * (1176 - mismatches) / 1176);
        $display("Mismatches:     %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] Pool1 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] Pool1 layer has mismatches.");
            $display("This indicates an issue in the average pooling implementation.");
        end
        
        $display({"="}*80 + "\n");
        $finish;
    end
    
    // Timeout
    initial begin
        #5_000_000;  // 5ms timeout
        $display("\n[ERROR] Timeout!");
        $finish;
    end

endmodule

