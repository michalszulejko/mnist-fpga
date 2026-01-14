`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Inference Testbench with Debug Features
================================================================================
Enhanced testbench with:
- Progress monitoring (state tracking, cycle counting)
- Per-test timeout detection
- Configurable test count (1, 10, or 100)
- State change detection for debugging hung FSM
- Verbose progress indicators

Tests MNIST images and compares:
  1. Predicted digit (FPGA vs Python)
  2. All 10 class scores/logits (FPGA vs Python)
  3. Reports mismatches and statistics

LeNet-5 Architecture:
  Conv1: 6 filters, 5x5, padding=2 -> 6x28x28
  Pool1: 2x2 Average -> 6x14x14
  Conv2: 16 filters, 5x5 -> 16x10x10
  Pool2: 2x2 Average -> 16x5x5 = 400 features
  FC1: 400 -> 120 (Tanh)
  FC2: 120 -> 84 (Tanh)
  FC3: 84 -> 10 (raw logits)
================================================================================
*/

module tb_inference_lenet_debug;

    // =========================================================================
    // Configuration Parameters
    // =========================================================================
    parameter NUM_TESTS = 5;  // Change to 1, 10, or 100
    parameter PROGRESS_UPDATE_CYCLES = 10000;  // Show progress every N cycles
    parameter PER_TEST_TIMEOUT = 10_000_000;   // 10ms per test timeout
    
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

    // Conv Weights RAM (Conv1: 150, Conv2: 2400 = 2550 total)
    reg [7:0] conv_weights_ram [0:2549];
    assign conv_w_data = conv_weights_ram[conv_w_addr];

    // Conv Biases RAM (Conv1: 6, Conv2: 16 = 22 total)
    reg [31:0] conv_biases_ram [0:21];
    assign conv_b_data = conv_biases_ram[conv_b_addr];

    // FC Weights RAM (FC1: 48000, FC2: 10080, FC3: 840 = 58920 total)
    reg [7:0] fc_weights_ram [0:58919];
    assign fc_w_data = fc_weights_ram[fc_w_addr];

    // FC Biases RAM (FC1: 120, FC2: 84, FC3: 10 = 214 total)
    reg [31:0] fc_biases_ram [0:213];
    assign fc_b_data = fc_biases_ram[fc_b_addr];

    // Tanh LUT (256 entries)
    reg [7:0] tanh_lut_ram [0:255];
    assign tanh_data = tanh_lut_ram[tanh_addr];

    // Buffer A (6*28*28 = 4704)
    reg [7:0] buf_a_ram [0:4703];
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];

    // Buffer B (6*14*14 = 1176)
    reg [7:0] buf_b_ram [0:1175];
    always @(posedge clk) begin
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];

    // Buffer C (16*5*5 = 400)
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
    // Debug Monitoring
    // =========================================================================
    reg [5:0] last_state;
    integer cycle_count;
    integer test_start_cycle;
    integer state_change_cycle;
    reg monitoring_active;

    // State name decoder for debugging
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

    // Progress monitor
    always @(posedge clk) begin
        if (monitoring_active) begin
            cycle_count <= cycle_count + 1;
            
            // Detect state changes
            if (dut.state !== last_state) begin
                state_change_cycle <= cycle_count;
                last_state <= dut.state;
            end
            
            // Periodic progress update
            if ((cycle_count % PROGRESS_UPDATE_CYCLES) == 0) begin
                $display("  [PROGRESS] Cycle %0d | State: %0s | Cycles since state change: %0d",
                    cycle_count, state_name(dut.state), cycle_count - state_change_cycle);
            end
            
            // Check for hung state (no state change in long time)
            if ((cycle_count - state_change_cycle) > 50000) begin
                $display("  [WARNING] State %0s has been active for %0d cycles - possible hang!",
                    state_name(dut.state), cycle_count - state_change_cycle);
                $display("  [DEBUG] f_idx=%0d, r=%0d, c=%0d, kr=%0d, kc=%0d, neuron_idx=%0d",
                    dut.f_idx, dut.r, dut.c, dut.kr, dut.kc, dut.neuron_idx);
                $finish;
            end
            
            // Per-test timeout
            if ((cycle_count - test_start_cycle) > PER_TEST_TIMEOUT) begin
                $display("\n[ERROR] Test timeout! No completion after %0d cycles", 
                    cycle_count - test_start_cycle);
                $display("  Last state: %0s", state_name(dut.state));
                $display("  State has been active for %0d cycles", cycle_count - state_change_cycle);
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

    // Load a single image into image_ram
    task load_image;
        input integer img_idx;
        integer i;
        integer src_addr;
        begin
            src_addr = img_idx * IMAGE_SIZE;
            for (i = 0; i < IMAGE_SIZE; i = i + 1) begin
                image_ram[i] = golden_pixels[src_addr + i];
            end
            $display("[LOAD] Image %0d loaded into RAM", img_idx);
        end
    endtask

    // Compare FPGA scores with golden reference
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
            // Extract FPGA scores
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

            // Extract golden scores
            base_addr = img_idx * NUM_CLASSES;
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                gold_scores[i] = golden_scores[base_addr + i];
            end

            // Compare
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
                $display("  [OK] Scores Match. Max Diff: %0d", max_diff);
            end else begin
                tests_failed = tests_failed + 1;
                $display("  [MM] %0d score mismatches detected", mismatches);
                $display("  Class |      FPGA      |     Golden     | Difference");
                $display("  ------|----------------|----------------|------------");
                for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                    $display("    %0d   | %12d | %12d | %10d %s",
                        i, fpga_scores[i], gold_scores[i], fpga_scores[i] - gold_scores[i],
                        (fpga_scores[i] === gold_scores[i]) ? "[OK]" : "[MM]");
                end
            end
        end
    endtask

    // Run inference on a single image
    task run_inference;
        input integer img_idx;
        integer test_cycles;
        begin
            $display("\n========================================");
            $display("Test %0d: Image Index %0d", img_idx + 1, img_idx);

            // 1. Load image
            load_image(img_idx);

            // Wait for RAM to stabilize
            repeat(10) @(posedge clk);

            // Reset monitoring counters
            test_start_cycle = cycle_count;
            monitoring_active = 1;

            // 2. Start inference
            @(posedge clk);
            #1;
            start = 1;
            @(posedge clk);
            #1;
            start = 0;

            // Wait for completion
            wait(done);
            @(posedge clk);
            
            monitoring_active = 0;
            test_cycles = cycle_count - test_start_cycle;

            // 3. Check prediction
            $display("  Expected: %0d | Predicted: %0d | Label: %0d | Cycles: %0d",
                golden_preds[img_idx], predicted_digit, golden_labels[img_idx], test_cycles);

            if (predicted_digit === golden_preds[img_idx]) begin
                pred_correct = pred_correct + 1;
            end else begin
                $display("  [MM] PREDICTION MISMATCH!");
                pred_incorrect = pred_incorrect + 1;
            end

            // Compare scores
            compare_scores(img_idx);

            // Wait before next test
            repeat(10) @(posedge clk);
        end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    integer i;

    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        tests_passed = 0;
        tests_failed = 0;
        pred_correct = 0;
        pred_incorrect = 0;
        cycle_count = 0;
        test_start_cycle = 0;
        state_change_cycle = 0;
        monitoring_active = 0;
        last_state = 6'd0;

        $display("\n================================================================================");
        $display(" LeNet-5 Inference Testbench (DEBUG MODE) - %0d Image Test", NUM_TESTS);
        $display("================================================================================");
        $display(" Progress updates every %0d cycles", PROGRESS_UPDATE_CYCLES);
        $display(" Per-test timeout: %0d cycles", PER_TEST_TIMEOUT);
        $display("================================================================================\n");

        // Load memory files
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

        $display("  > Conv weights loaded (2550 bytes)");
        $display("  > Conv biases loaded (22 entries)");
        $display("  > FC weights loaded (58920 bytes)");
        $display("  > FC biases loaded (214 entries)");
        $display("  > Tanh LUT loaded (256 entries)");
        $display("  > Test vectors loaded (%0d images)\n", NUM_TESTS);

        // Reset
        #100;
        rst = 0;
        #100;

        // Run tests on all images
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            run_inference(i);
        end

        // Final Summary
        $display("\n\n================================================================================");
        $display(" FINAL RESULTS (%0d IMAGES)", NUM_TESTS);
        $display("================================================================================");
        $display("\nPrediction Accuracy:");
        $display("  Correct:   %0d/%0d (%.1f%%)", pred_correct, NUM_TESTS,
            (pred_correct * 100.0) / NUM_TESTS);
        $display("  Incorrect: %0d/%0d", pred_incorrect, NUM_TESTS);

        $display("\nScore Comparison:");
        $display("  Exact Matches: %0d/%0d tests", tests_passed, NUM_TESTS);
        $display("  Mismatches:    %0d/%0d tests", tests_failed, NUM_TESTS);

        $display("\nPerformance:");
        $display("  Total cycles: %0d", cycle_count);
        $display("  Average cycles per image: %0d", cycle_count / NUM_TESTS);

        if (tests_passed === NUM_TESTS) begin
            $display("\n[SUCCESS] All %0d tests passed! FPGA matches Python exactly.", NUM_TESTS);
        end else begin
            $display("\n[FAILURE] Some tests failed.");
        end

        $display("\n================================================================================\n");
        $finish;
    end

    // =========================================================================
    // Global Timeout Watchdog
    // =========================================================================
    initial begin
        #500_000_000; // 500ms global timeout
        $display("\n[ERROR] Global simulation timeout!");
        $display("Simulation exceeded 500ms - likely a serious issue.");
        $finish;
    end

endmodule
