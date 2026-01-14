`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Pool2 Layer Testbench
================================================================================
Tests ONLY the Pool2 layer: 16x10x10 -> 16x5x5 (2x2 average pooling)

Loads:
- Conv2 output (layer3_conv2.mem) as input
- Golden reference (layer4_pool2.mem)

Compares:
- FPGA Buffer C output vs Python golden Pool2 output
================================================================================
*/

module tb_layer_pool2;

    // =========================================================================
    // Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    reg done_manual;
    
    // Buffer A (input: 16*10*10 = 1600 bytes)
    reg [7:0] buf_a_ram [0:1599];
    reg [10:0] buf_a_addr;
    wire signed [7:0] buf_a_rd_data;
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    
    // Buffer C (output: 16*5*5 = 400 bytes)
    reg [7:0] buf_c_ram [0:399];
    reg [8:0] buf_c_addr;
    reg [7:0] buf_c_wr_data;
    reg buf_c_wr_en;
    wire signed [7:0] buf_c_rd_data;
    
    always @(posedge clk) begin
        if (buf_c_wr_en) buf_c_ram[buf_c_addr] <= buf_c_wr_data;
    end
    assign buf_c_rd_data = buf_c_ram[buf_c_addr];
    
    // Golden reference
    reg [7:0] golden_pool2 [0:399];
    
    // =========================================================================
    // Pool2 FSM
    // =========================================================================
    localparam IDLE        = 2'd0;
    localparam POOL        = 2'd1;
    localparam POOL_CALC   = 2'd2;
    localparam DONE        = 2'd3;
    
    reg [1:0] state;
    reg [4:0] f_idx;        // Filter index (0-15)
    reg [2:0] r, c;         // Output position (0-4)
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
            buf_c_wr_en <= 0;
        end else begin
            buf_c_wr_en <= 0;
            
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
                            state <= POOL_CALC;
                        end
                    endcase
                end
                
                POOL_CALC: begin
                    buf_c_addr <= f_idx*25 + r*5 + c;
                    buf_c_wr_data <= temp_val[7:0];
                    buf_c_wr_en <= 1;
                    pool_step <= 0;
                    
                    if (c == 4) begin
                        c <= 0;
                        if (r == 4) begin
                            r <= 0;
                            if (f_idx == 15) begin
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
        $display("LeNet-5 Pool2 Layer Testbench");
        $display("Testing: 16x10x10 -> 16x5x5 (2x2 Average Pool)");
        $display({"="}*80);
        
        // Load memory files
        $display("\nLoading memory files...");
        $readmemh("layer3_conv2.mem", buf_a_ram);
        $readmemh("layer4_pool2.mem", golden_pool2);
        $display("  ✓ Conv2 output loaded (1600 bytes)");
        $display("  ✓ Golden Pool2 output loaded (400 bytes)");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Start pooling
        $display("\nStarting Pool2 layer...");
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
        
        $display("  ✓ Pool2 completed");
        
        // Compare results
        $display("\nComparing FPGA output vs Golden reference...");
        mismatches = 0;
        max_diff = 0;
        
        for (i = 0; i < 400; i = i + 1) begin
            fpga_val = buf_c_ram[i];
            gold_val = golden_pool2[i];
            
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
        $display("Total elements: 400 (16 filters × 5 × 5)");
        $display("Exact matches:  %0d (%.2f%%)", 400 - mismatches,
                 100.0 * (400 - mismatches) / 400);
        $display("Mismatches:     %0d", mismatches);
        $display("Max difference: %0d", max_diff);
        
        if (mismatches == 0) begin
            $display("\n[SUCCESS] Pool2 layer output matches golden reference exactly!");
        end else begin
            $display("\n[FAILURE] Pool2 layer has mismatches.");
            $display("This indicates an issue in the average pooling implementation.");
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
