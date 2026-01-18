`timescale 1ns / 1ps

/*
================================================================================
LeNet-5 Inference Testbench
================================================================================
Tests 100 MNIST images and compares:
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

module tb_inference_lenet;

    // =========================================================================
    // Constants
    // =========================================================================
    localparam NUM_TESTS = 16;  // Testing images 500-2499 from MNIST test set
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
    // NOTE: Must use 2-cycle synchronous read to match BRAM behavior on FPGA!
    // Vivado may add output registers (DOA_REG=1) for better timing.
    reg [7:0] fc_weights_ram [0:58919];
    reg [7:0] fc_w_data_reg1;  // First pipeline stage
    reg [7:0] fc_w_data_reg2;  // Second pipeline stage (output register)
    always @(posedge clk) begin
        fc_w_data_reg1 <= fc_weights_ram[fc_w_addr];  // Cycle 1: RAM read
        fc_w_data_reg2 <= fc_w_data_reg1;             // Cycle 2: Output register
    end
    assign fc_w_data = fc_w_data_reg2;

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
    // Test Statistics
    // =========================================================================
    integer tests_passed;
    integer tests_failed;
    integer pred_match_python;    // FPGA matches Python prediction (bit-exactness)
    integer pred_mismatch_python;
    integer actual_correct;       // FPGA prediction matches TRUE label (real accuracy)
    integer actual_incorrect;

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

            // Always show score table
            $display("  Class |      FPGA      |     Golden     | Difference");
            $display("  ------|----------------|----------------|------------");
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                $display("    %0d   | %12d | %12d | %10d %s",
                    i, fpga_scores[i], gold_scores[i], fpga_scores[i] - gold_scores[i],
                    (fpga_scores[i] === gold_scores[i]) ? "[OK]" : "[MM]");
            end

            if (mismatches == 0) begin
                tests_passed = tests_passed + 1;
                $display("  [PASS] All scores match exactly");
            end else begin
                tests_failed = tests_failed + 1;
                $display("  [FAIL] %0d score mismatches detected", mismatches);
            end
        end
    endtask

    // Run inference on a single image
    task run_inference;
        input integer img_idx;
        begin
            $display("\n========================================");
            $display("Test %0d: Label: %0d", img_idx + 1, golden_labels[img_idx]);

            // 1. Load image
            load_image(img_idx);

            // Wait for RAM to stabilize
            repeat(10) @(posedge clk);

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

            // 3. Check prediction
            $display("  FPGA Predicted: %0d | Python Predicted: %0d | True Label: %0d %s %s",
                predicted_digit, golden_preds[img_idx], golden_labels[img_idx],
                (predicted_digit === golden_preds[img_idx]) ? "" : "[PYTHON MISMATCH]",
                (predicted_digit === golden_labels[img_idx]) ? "[CORRECT]" : "[WRONG]");

            // Track bit-exactness (FPGA vs Python)
            if (predicted_digit === golden_preds[img_idx]) begin
                pred_match_python = pred_match_python + 1;
            end else begin
                pred_mismatch_python = pred_mismatch_python + 1;
            end

            // Track actual accuracy (FPGA vs True Label)
            if (predicted_digit === golden_labels[img_idx]) begin
                actual_correct = actual_correct + 1;
            end else begin
                actual_incorrect = actual_incorrect + 1;
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
        pred_match_python = 0;
        pred_mismatch_python = 0;
        actual_correct = 0;
        actual_incorrect = 0;

        $display("\n================================================================================");
        $display(" LeNet-5 Inference Testbench - %0d Image Test", NUM_TESTS);
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

        $display("\n1. ACTUAL MODEL ACCURACY (FPGA vs True Labels):");
        $display("   Correct:   %0d/%0d (%.2f%%)", actual_correct, NUM_TESTS,
            (actual_correct * 100.0) / NUM_TESTS);
        $display("   Incorrect: %0d/%0d", actual_incorrect, NUM_TESTS);

        $display("\n2. BIT-EXACTNESS (FPGA vs Python Predictions):");
        $display("   Match:     %0d/%0d (%.2f%%)", pred_match_python, NUM_TESTS,
            (pred_match_python * 100.0) / NUM_TESTS);
        $display("   Mismatch:  %0d/%0d", pred_mismatch_python, NUM_TESTS);

        $display("\n3. SCORE COMPARISON (All 10 logits):");
        $display("   Exact Matches: %0d/%0d tests", tests_passed, NUM_TESTS);
        $display("   Mismatches:    %0d/%0d tests", tests_failed, NUM_TESTS);

        if (tests_passed === NUM_TESTS && pred_match_python === NUM_TESTS) begin
            $display("\n[SUCCESS] FPGA matches Python exactly (bit-exact).");
            $display("          Model accuracy on test set: %.2f%%", (actual_correct * 100.0) / NUM_TESTS);
        end else begin
            $display("\n[FAILURE] FPGA does not match Python exactly.");
        end

        $display("\n================================================================================\n");
        $finish;
    end

endmodule
