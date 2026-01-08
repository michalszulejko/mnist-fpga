`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Inference Debug Testbench
================================================================================
Simplified testbench with extensive debugging to identify where inference hangs.
Tests a SINGLE image with detailed state monitoring and early timeout.
================================================================================
*/

module tb_inference_debug;

    // =========================================================================
    // Constants
    // =========================================================================
    localparam IMAGE_SIZE = 784;
    localparam NUM_CLASSES = 10;
    localparam TIMEOUT_CYCLES = 100000; // 100K cycles = 1ms @ 100MHz

    // =========================================================================
    // DUT Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    wire done;
    wire [3:0] predicted_digit;

    // Image RAM
    wire [9:0] img_addr;
    wire signed [7:0] img_data;

    // Conv Weights RAM (2550 bytes)
    wire [11:0] conv_w_addr;
    wire signed [7:0] conv_w_data;

    // Conv Biases RAM (22 biases)
    wire [4:0] conv_b_addr;
    wire signed [31:0] conv_b_data;

    // FC Weights RAM (58920 bytes)
    wire [15:0] fc_w_addr;
    wire signed [7:0] fc_w_data;

    // FC Biases RAM (214 biases)
    wire [7:0] fc_b_addr;
    wire signed [31:0] fc_b_data;

    // Tanh LUT
    wire [7:0] tanh_addr;
    wire signed [7:0] tanh_data;

    // Buffer A (4704 bytes)
    wire [12:0] buf_a_addr;
    wire [7:0] buf_a_wr_data;
    wire buf_a_wr_en;
    wire signed [7:0] buf_a_rd_data;

    // Buffer B (1176 bytes)
    wire [10:0] buf_b_addr;
    wire [7:0] buf_b_wr_data;
    wire buf_b_wr_en;
    wire signed [7:0] buf_b_rd_data;

    // Buffer C (400 bytes)
    wire [8:0] buf_c_addr;
    wire [7:0] buf_c_wr_data;
    wire buf_c_wr_en;
    wire signed [7:0] buf_c_rd_data;

    // Class Scores
    wire signed [31:0] class_score_0, class_score_1, class_score_2, class_score_3, class_score_4;
    wire signed [31:0] class_score_5, class_score_6, class_score_7, class_score_8, class_score_9;

    // =========================================================================
    // RAM Instances
    // =========================================================================

    // Image RAM
    reg [7:0] image_ram [0:783];
    assign img_data = image_ram[img_addr];

    // Conv Weights RAM
    reg [7:0] conv_weights_ram [0:2549];
    assign conv_w_data = conv_weights_ram[conv_w_addr];

    // Conv Biases RAM
    reg [31:0] conv_biases_ram [0:21];
    assign conv_b_data = conv_biases_ram[conv_b_addr];

    // FC Weights RAM
    reg [7:0] fc_weights_ram [0:58919];
    assign fc_w_data = fc_weights_ram[fc_w_addr];

    // FC Biases RAM
    reg [31:0] fc_biases_ram [0:213];
    assign fc_b_data = fc_biases_ram[fc_b_addr];

    // Tanh LUT
    reg [7:0] tanh_lut_ram [0:255];
    assign tanh_data = tanh_lut_ram[tanh_addr];

    // Buffer A
    reg [7:0] buf_a_ram [0:4703];
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];

    // Buffer B
    reg [7:0] buf_b_ram [0:1175];
    always @(posedge clk) begin
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];

    // Buffer C
    reg [7:0] buf_c_ram [0:399];
    always @(posedge clk) begin
        if (buf_c_wr_en) buf_c_ram[buf_c_addr] <= buf_c_wr_data;
    end
    assign buf_c_rd_data = buf_c_ram[buf_c_addr];

    // =========================================================================
    // DUT Instantiation
    // =========================================================================
    inference dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_digit(predicted_digit),

        .img_addr(img_addr),
        .img_data(img_data),

        .conv_w_addr(conv_w_addr),
        .conv_w_data(conv_w_data),
        .conv_b_addr(conv_b_addr),
        .conv_b_data(conv_b_data),

        .fc_w_addr(fc_w_addr),
        .fc_w_data(fc_w_data),
        .fc_b_addr(fc_b_addr),
        .fc_b_data(fc_b_data),

        .tanh_addr(tanh_addr),
        .tanh_data(tanh_data),

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
    // Clock Generation (100 MHz)
    // =========================================================================
    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // Monitoring Variables
    // =========================================================================
    integer cycle_count;
    reg [11:0] prev_conv_w_addr;
    reg [4:0] prev_conv_b_addr;
    reg [15:0] prev_fc_w_addr;
    reg [7:0] prev_fc_b_addr;
    reg prev_buf_a_wr_en, prev_buf_b_wr_en, prev_buf_c_wr_en;
    integer idle_cycles;
    integer last_activity_cycle;

    // =========================================================================
    // Activity Monitor - Detects if hardware is stuck
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            idle_cycles <= 0;
            last_activity_cycle <= 0;
            prev_conv_w_addr <= 0;
            prev_conv_b_addr <= 0;
            prev_fc_w_addr <= 0;
            prev_fc_b_addr <= 0;
            prev_buf_a_wr_en <= 0;
            prev_buf_b_wr_en <= 0;
            prev_buf_c_wr_en <= 0;
        end else if (start || !done) begin
            // Check if any activity happened
            if (conv_w_addr != prev_conv_w_addr ||
                conv_b_addr != prev_conv_b_addr ||
                fc_w_addr != prev_fc_w_addr ||
                fc_b_addr != prev_fc_b_addr ||
                buf_a_wr_en || buf_b_wr_en || buf_c_wr_en ||
                img_addr != 0) begin
                
                idle_cycles <= 0;
                last_activity_cycle <= cycle_count;
            end else begin
                idle_cycles <= idle_cycles + 1;
            end

            prev_conv_w_addr <= conv_w_addr;
            prev_conv_b_addr <= conv_b_addr;
            prev_fc_w_addr <= fc_w_addr;
            prev_fc_b_addr <= fc_b_addr;
            prev_buf_a_wr_en <= buf_a_wr_en;
            prev_buf_b_wr_en <= buf_b_wr_en;
            prev_buf_c_wr_en <= buf_c_wr_en;

            // Report if stuck for 1000 cycles
            if (idle_cycles == 1000) begin
                $display("\n[WARNING] No activity detected for 1000 cycles!");
                $display("  Last activity at cycle: %0d", last_activity_cycle);
                $display("  Current signals:");
                $display("    done=%b start=%b", done, start);
                $display("    img_addr=%0d", img_addr);
                $display("    conv_w_addr=%0d conv_b_addr=%0d", conv_w_addr, conv_b_addr);
                $display("    fc_w_addr=%0d fc_b_addr=%0d", fc_w_addr, fc_b_addr);
                $display("    buf_a_wr_en=%b buf_b_wr_en=%b buf_c_wr_en=%b", 
                    buf_a_wr_en, buf_b_wr_en, buf_c_wr_en);
            end
        end
    end

    // =========================================================================
    // Cycle Counter & Progress Monitor
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            cycle_count <= 0;
        end else if (!done && cycle_count > 0) begin
            cycle_count <= cycle_count + 1;
            
            // Progress updates every 10K cycles
            if (cycle_count % 10000 == 0) begin
                $display("[PROGRESS] Cycle %0d - done=%b idle_cycles=%0d", 
                    cycle_count, done, idle_cycles);
            end
        end else if (start) begin
            cycle_count <= 1;
        end
    end

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;

        $display("\n================================================================================");
        $display(" LeNet-5 Inference DEBUG Testbench");
        $display(" Single Image Test with Activity Monitoring");
        $display("================================================================================\n");

        // Load memory files
        $display("Loading memory files...");

        // Try to load, but don't fail if files don't exist
        $readmemh("sim_conv_weights.mem", conv_weights_ram);
        $readmemh("sim_conv_biases.mem", conv_biases_ram);
        $readmemh("sim_fc_weights.mem", fc_weights_ram);
        $readmemh("sim_fc_biases.mem", fc_biases_ram);
        $readmemh("tanh_lut.mem", tanh_lut_ram);

        // Load a test image (or create dummy data)
        $readmemh("test_pixels.mem", image_ram);

        $display("  > Memory files loaded\n");

        // Reset
        #100;
        rst = 0;
        #100;

        $display("========================================");
        $display("Starting Single Image Inference Test");
        $display("========================================\n");

        // Start inference
        @(posedge clk);
        #1;
        start = 1;
        $display("[CYCLE %0d] START signal asserted", cycle_count);
        
        @(posedge clk);
        #1;
        start = 0;
        $display("[CYCLE %0d] START signal deasserted, waiting for DONE...\n", cycle_count);

        // Wait for completion with timeout
        fork
            begin
                // Wait for done
                wait(done);
                $display("\n[CYCLE %0d] DONE signal received!", cycle_count);
                $display("  Predicted digit: %0d", predicted_digit);
                $display("  Class scores:");
                $display("    [0]: %0d", class_score_0);
                $display("    [1]: %0d", class_score_1);
                $display("    [2]: %0d", class_score_2);
                $display("    [3]: %0d", class_score_3);
                $display("    [4]: %0d", class_score_4);
                $display("    [5]: %0d", class_score_5);
                $display("    [6]: %0d", class_score_6);
                $display("    [7]: %0d", class_score_7);
                $display("    [8]: %0d", class_score_8);
                $display("    [9]: %0d", class_score_9);
                
                $display("\n[SUCCESS] Inference completed in %0d cycles", cycle_count);
                $display("           (%.2f μs @ 100MHz)\n", cycle_count * 0.01);
                disable timeout_block;
            end

            begin: timeout_block
                repeat(TIMEOUT_CYCLES) @(posedge clk);
                $display("\n\n================================================================================");
                $display("[ERROR] TIMEOUT after %0d cycles (%.1f μs)", TIMEOUT_CYCLES, TIMEOUT_CYCLES * 0.01);
                $display("================================================================================");
                $display("\nInference module appears to be STUCK or UNIMPLEMENTED");
                $display("\nFinal Signal State:");
                $display("  done=%b start=%b rst=%b", done, start, rst);
                $display("  idle_cycles=%0d (cycles since last activity)", idle_cycles);
                $display("  last_activity_cycle=%0d", last_activity_cycle);
                $display("\nMemory Access Signals:");
                $display("  img_addr=%0d (of 0-783)", img_addr);
                $display("  conv_w_addr=%0d (of 0-2549)", conv_w_addr);
                $display("  conv_b_addr=%0d (of 0-21)", conv_b_addr);
                $display("  fc_w_addr=%0d (of 0-58919)", fc_w_addr);
                $display("  fc_b_addr=%0d (of 0-213)", fc_b_addr);
                $display("\nBuffer Write Enables:");
                $display("  buf_a_wr_en=%b buf_b_wr_en=%b buf_c_wr_en=%b", 
                    buf_a_wr_en, buf_b_wr_en, buf_c_wr_en);
                $display("\nDiagnostics:");
                
                if (idle_cycles > 900) begin
                    $display("  >> Module appears COMPLETELY STUCK (no activity for %0d cycles)", idle_cycles);
                    $display("  >> Possible causes:");
                    $display("     - State machine not implemented or missing transitions");
                    $display("     - Missing 'done' signal assertion logic");
                    $display("     - Module is just a skeleton/placeholder");
                end else if (last_activity_cycle < 100) begin
                    $display("  >> Module stopped very early (cycle %0d)", last_activity_cycle);
                    $display("  >> Possible causes:");
                    $display("     - Initialization issue");
                    $display("     - Early state machine exit");
                    $display("     - Missing start signal handling");
                end else begin
                    $display("  >> Module was active until cycle %0d, then stopped", last_activity_cycle);
                    $display("  >> Possible causes:");
                    $display("     - Stuck in a loop/state");
                    $display("     - Counter overflow or boundary condition");
                    $display("     - Missing state transition condition");
                end
                
                $display("\n================================================================================\n");
            end
        join

        $display("\n================================================================================\n");
        $finish;
    end

endmodule
