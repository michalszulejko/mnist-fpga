`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Inference Quick Testbench
================================================================================
Minimal testbench for rapid iteration and debugging:
- Tests only 3 images by default (configurable)
- Verbose state-by-state output
- Shorter timeout (5ms per test)
- Useful for verifying FSM fixes and initial debugging

Usage:
- Change NUM_TESTS parameter to 1, 2, or 3 for faster testing
- Enable VERBOSE_STATE for cycle-by-cycle state monitoring
- Use this for quick validation before running full test suite
================================================================================
*/

module tb_inference_lenet_quick;

    // =========================================================================
    // Configuration Parameters
    // =========================================================================
    parameter NUM_TESTS = 3;  // Test only first 3 images
    parameter VERBOSE_STATE = 0;  // 1 = print every state change
    parameter SHOW_EVERY_N_CYCLES = 5000;  // Progress every N cycles
    parameter PER_TEST_TIMEOUT = 5_000_000;   // 5ms per test
    
    localparam IMAGE_SIZE = 784;
    localparam NUM_CLASSES = 10;

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
    // Golden Reference Storage
    // =========================================================================
    reg [7:0] golden_pixels [0:NUM_TESTS*IMAGE_SIZE-1];
    reg [31:0] golden_scores [0:NUM_TESTS*NUM_CLASSES-1];
    reg [3:0] golden_preds [0:NUM_TESTS-1];
    reg [3:0] golden_labels [0:NUM_TESTS-1];

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
    always #5 clk = ~clk;

    // =========================================================================
    // State Monitoring
    // =========================================================================
    reg [5:0] last_state;
    integer cycle_count;
    integer test_start_cycle;
    reg monitoring_active;

    // State name decoder
    function [95:0] state_name;
        input [5:0] state;
        begin
            case(state)
                6'd0: state_name = "IDLE";
                6'd1: state_name = "L1_LOAD_BIAS";
                6'd2: state_name = "L1_LOAD_BIAS_W";
                6'd3: state_name = "L1_PREFETCH";
                6'd4: state_name = "L1_CONV";
                6'd5: state_name = "L1_TANH";
                6'd6: state_name = "L1_SAVE";
                6'd7: state_name = "L1_POOL";
                6'd8: state_name = "L1_POOL_CALC";
                6'd9: state_name = "L2_LOAD_BIAS";
                6'd10: state_name = "L2_LOAD_BIAS_W";
                6'd11: state_name = "L2_PREFETCH";
                6'd12: state_name = "L2_CONV";
                6'd13: state_name = "L2_TANH";
                6'd14: state_name = "L2_SAVE";
                6'd15: state_name = "L2_POOL";
                6'd16: state_name = "L2_POOL_CALC";
                6'd17: state_name = "FC1_LOAD_BIAS";
                6'd18: state_name = "FC1_LOAD_BIAS_W";
                6'd19: state_name = "FC1_PREFETCH";
                6'd20: state_name = "FC1_MULT";
                6'd21: state_name = "FC1_TANH";
                6'd22: state_name = "FC1_SAVE";
                6'd23: state_name = "FC2_LOAD_BIAS";
                6'd24: state_name = "FC2_LOAD_BIAS_W";
                6'd25: state_name = "FC2_PREFETCH";
                6'd26: state_name = "FC2_MULT";
                6'd27: state_name = "FC2_TANH";
                6'd28: state_name = "FC2_SAVE";
                6'd29: state_name = "FC3_LOAD_BIAS";
                6'd30: state_name = "FC3_LOAD_BIAS_W";
                6'd31: state_name = "FC3_PREFETCH";
                6'd32: state_name = "FC3_MULT";
                6'd33: state_name = "FC3_NEXT";
                6'd34: state_name = "DONE_STATE";
                default: state_name = "UNKNOWN";
            endcase
        end
    endfunction

    // Monitor state changes
    always @(posedge clk) begin
        if (monitoring_active) begin
            cycle_count <= cycle_count + 1;
            
            // Verbose state change reporting
            if (VERBOSE_STATE && (dut.state !== last_state)) begin
                $display("    [CYCLE %0d] State: %0s -> %0s",
                    cycle_count, state_name(last_state), state_name(dut.state));
                last_state <= dut.state;
            end else if (!VERBOSE_STATE && (dut.state !== last_state)) begin
                last_state <= dut.state;
            end
            
            // Periodic progress
            if ((cycle_count % SHOW_EVERY_N_CYCLES) == 0) begin
                $display("    [PROGRESS] Cycle %0d | State: %0s | f_idx=%0d r=%0d c=%0d",
                    cycle_count, state_name(dut.state), dut.f_idx, dut.r, dut.c);
            end
            
            // Timeout check
            if ((cycle_count - test_start_cycle) > PER_TEST_TIMEOUT) begin
                $display("\n[ERROR] Test timeout at cycle %0d!", cycle_count);
                $display("  State: %0s", state_name(dut.state));
                $display("  Position: f_idx=%0d r=%0d c=%0d kr=%0d kc=%0d",
                    dut.f_idx, dut.r, dut.c, dut.kr, dut.kc);
                $finish;
            end
        end
    end

    // =========================================================================
    // Test Statistics
    // =========================================================================
    integer tests_passed;
    integer tests_failed;
    integer pred_correct;
    integer pred_incorrect;

    // =========================================================================
    // Helper Tasks
    // =========================================================================

    task load_image;
        input integer img_idx;
        integer i;
        integer src_addr;
        begin
            src_addr = img_idx * IMAGE_SIZE;
            for (i = 0; i < IMAGE_SIZE; i = i + 1) begin
                image_ram[i] = golden_pixels[src_addr + i];
            end
            $display("  [LOAD] Image %0d loaded", img_idx);
        end
    endtask

    task compare_scores;
        input integer img_idx;
        reg signed [31:0] fpga_scores [0:9];
        reg signed [31:0] gold_scores [0:9];
        integer base_addr;
        integer i;
        integer mismatches;
        integer max_diff;
        integer diff;
        begin
            fpga_scores[0] = class_score_0;
            fpga_scores[1] = class_score_1;
            fpga_scores[2] = class_score_2;
            fpga_scores[3] = class_score_3;
            fpga_scores[4] = class_score_4;
            fpga_scores[5] = class_score_5;
            fpga_scores[6] = class_score_6;
            fpga_scores[7] = class_score_7;
            fpga_scores[8] = class_score_8;
            fpga_scores[9] = class_score_9;

            base_addr = img_idx * NUM_CLASSES;
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                gold_scores[i] = golden_scores[base_addr + i];
            end

            mismatches = 0;
            max_diff = 0;

            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                diff = fpga_scores[i] - gold_scores[i];
                if (diff < 0) diff = -diff;
                if (diff > max_diff) max_diff = diff;
                if (fpga_scores[i] !== gold_scores[i]) mismatches = mismatches + 1;
            end

            if (mismatches == 0) begin
                tests_passed = tests_passed + 1;
                $display("  [OK] Scores match perfectly (max diff: %0d)", max_diff);
            end else begin
                tests_failed = tests_failed + 1;
                $display("  [MM] %0d score mismatches", mismatches);
                $display("\n  Class |      FPGA      |     Golden     | Difference");
                $display("  ------|----------------|----------------|------------");
                for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                    $display("    %0d   | %12d | %12d | %10d %s",
                        i, fpga_scores[i], gold_scores[i], fpga_scores[i] - gold_scores[i],
                        (fpga_scores[i] === gold_scores[i]) ? "[OK]" : "[MM]");
                end
                $display("");
            end
        end
    endtask

    task run_inference;
        input integer img_idx;
        integer test_cycles;
        begin
            $display("\n========================================");
            $display("TEST %0d: Image Index %0d", img_idx + 1, img_idx);
            $display("========================================");

            load_image(img_idx);
            repeat(10) @(posedge clk);

            test_start_cycle = cycle_count;
            monitoring_active = 1;

            @(posedge clk);
            #1;
            start = 1;
            @(posedge clk);
            #1;
            start = 0;

            $display("  [START] Inference started at cycle %0d", cycle_count);

            wait(done);
            @(posedge clk);
            
            monitoring_active = 0;
            test_cycles = cycle_count - test_start_cycle;

            $display("  [DONE] Inference completed in %0d cycles (%.2f us @ 100MHz)",
                test_cycles, test_cycles * 0.01);
            $display("  [RESULT] Expected=%0d | Predicted=%0d | Label=%0d",
                golden_preds[img_idx], predicted_digit, golden_labels[img_idx]);

            if (predicted_digit === golden_preds[img_idx]) begin
                pred_correct = pred_correct + 1;
                $display("  [OK] Prediction correct!");
            end else begin
                $display("  [ERROR] PREDICTION MISMATCH!");
                pred_incorrect = pred_incorrect + 1;
            end

            compare_scores(img_idx);
            repeat(10) @(posedge clk);
        end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    integer i;

    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        tests_passed = 0;
        tests_failed = 0;
        pred_correct = 0;
        pred_incorrect = 0;
        cycle_count = 0;
        test_start_cycle = 0;
        monitoring_active = 0;
        last_state = 6'd0;

        $display("\n================================================================================");
        $display(" LeNet-5 Quick Test - Testing %0d Images", NUM_TESTS);
        $display("================================================================================");
        $display(" Mode: Quick validation");
        $display(" Verbose state: %0s", VERBOSE_STATE ? "ENABLED" : "DISABLED");
        $display(" Progress updates: Every %0d cycles", SHOW_EVERY_N_CYCLES);
        $display(" Timeout: %0d cycles per test", PER_TEST_TIMEOUT);
        $display("================================================================================\n");

        $display("Loading memory files...");
        $readmemh("sim_conv_weights.mem", conv_weights_ram);
        $readmemh("sim_conv_biases.mem", conv_biases_ram);
        $readmemh("sim_fc_weights.mem", fc_weights_ram);
        $readmemh("sim_fc_biases.mem", fc_biases_ram);
        $readmemh("tanh_lut.mem", tanh_lut_ram);
        $readmemh("test_pixels.mem", golden_pixels);
        $readmemh("test_scores.mem", golden_scores);
        $readmemh("test_preds.mem", golden_preds);
        $readmemh("test_labels.mem", golden_labels);
        $display("  > All memory files loaded\n");

        #100;
        rst = 0;
        #100;

        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            run_inference(i);
        end

        $display("\n\n================================================================================");
        $display(" QUICK TEST RESULTS (%0d IMAGES)", NUM_TESTS);
        $display("================================================================================");
        $display("Prediction: %0d/%0d correct (%.1f%%)", 
            pred_correct, NUM_TESTS, (pred_correct * 100.0) / NUM_TESTS);
        $display("Scores:     %0d/%0d exact matches", tests_passed, NUM_TESTS);
        $display("Cycles:     %0d total, %0d avg per image", 
            cycle_count, cycle_count / NUM_TESTS);
        $display("Time:       %.2f us (@ 100MHz)", cycle_count * 0.01);

        if (tests_passed === NUM_TESTS && pred_correct === NUM_TESTS) begin
            $display("\n[SUCCESS] All quick tests passed!");
        end else begin
            $display("\n[WARNING] Some tests failed - check details above");
        end

        $display("================================================================================\n");
        $finish;
    end

    // Global timeout
    initial begin
        #50_000_000; // 50ms total
        $display("\n[ERROR] Global timeout - simulation too long!");
        $finish;
    end

endmodule
