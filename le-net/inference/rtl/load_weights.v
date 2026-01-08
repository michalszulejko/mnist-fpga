/*
================================================================================
LeNet-5 Weight Loader
================================================================================
Weight Packet Structure (62,670 bytes total):

Offset      Size        Data
----------------------------------------------
0           150         Conv1 weights (6x1x5x5)
150         24          Conv1 biases (6 x int32)
174         2,400       Conv2 weights (16x6x5x5)
2,574       64          Conv2 biases (16 x int32)
2,638       48,000      FC1 weights (120x400)
50,638      480         FC1 biases (120 x int32)
51,118      10,080      FC2 weights (84x120)
61,198      336         FC2 biases (84 x int32)
61,534      840         FC3 weights (10x84)
62,374      40          FC3 biases (10 x int32)
62,414      256         Tanh LUT (256 entries)
----------------------------------------------
Total:      62,670 bytes
================================================================================
*/

module weight_loader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,
    input wire rx_ready,
    output reg transfer_done,
    output reg [15:0] led,

    // Conv Weights (Conv1 + Conv2)
    output reg [11:0] conv_w_addr,
    output reg [7:0] conv_w_data,
    output reg conv_w_en,

    // Conv Biases (Conv1 + Conv2)
    output reg [4:0] conv_b_addr,
    output reg [31:0] conv_b_data,
    output reg conv_b_en,

    // FC Weights (FC1 + FC2 + FC3)
    output reg [15:0] fc_w_addr,
    output reg [7:0] fc_w_data,
    output reg fc_w_en,

    // FC Biases (FC1 + FC2 + FC3)
    output reg [7:0] fc_b_addr,
    output reg [31:0] fc_b_data,
    output reg fc_b_en,

    // Tanh LUT
    output reg [7:0] tanh_addr,
    output reg [7:0] tanh_data,
    output reg tanh_en
);

    // Address Map for LeNet-5
    localparam C1_W_START = 0;
    localparam C1_W_END   = 150;           // Conv1 weights: 6*1*5*5 = 150

    localparam C1_B_START = C1_W_END;
    localparam C1_B_END   = C1_B_START + 24;   // Conv1 biases: 6 * 4 = 24

    localparam C2_W_START = C1_B_END;
    localparam C2_W_END   = C2_W_START + 2400; // Conv2 weights: 16*6*5*5 = 2400

    localparam C2_B_START = C2_W_END;
    localparam C2_B_END   = C2_B_START + 64;   // Conv2 biases: 16 * 4 = 64

    localparam FC1_W_START = C2_B_END;
    localparam FC1_W_END   = FC1_W_START + 48000; // FC1 weights: 120*400 = 48000

    localparam FC1_B_START = FC1_W_END;
    localparam FC1_B_END   = FC1_B_START + 480;   // FC1 biases: 120 * 4 = 480

    localparam FC2_W_START = FC1_B_END;
    localparam FC2_W_END   = FC2_W_START + 10080; // FC2 weights: 84*120 = 10080

    localparam FC2_B_START = FC2_W_END;
    localparam FC2_B_END   = FC2_B_START + 336;   // FC2 biases: 84 * 4 = 336

    localparam FC3_W_START = FC2_B_END;
    localparam FC3_W_END   = FC3_W_START + 840;   // FC3 weights: 10*84 = 840

    localparam FC3_B_START = FC3_W_END;
    localparam FC3_B_END   = FC3_B_START + 40;    // FC3 biases: 10 * 4 = 40

    localparam TANH_START  = FC3_B_END;
    localparam TANH_END    = TANH_START + 256;    // Tanh LUT: 256 entries

    localparam TOTAL       = TANH_END;            // 62,670 bytes total

    reg [16:0] global_addr;  // Need 17 bits for addresses up to 62,670
    reg [1:0] byte_idx;
    reg [31:0] bias_accum;

    always @(posedge clk) begin
        if (rst) begin
            transfer_done <= 0;
            global_addr <= 0;
            byte_idx <= 0;
            conv_w_en <= 0; conv_b_en <= 0;
            fc_w_en <= 0; fc_b_en <= 0;
            tanh_en <= 0;
            led <= 0;
            bias_accum <= 0;
        end else if (rx_ready) begin
            // Default: disable all write enables
            conv_w_en <= 0; conv_b_en <= 0;
            fc_w_en <= 0; fc_b_en <= 0;
            tanh_en <= 0;

            // 1. Conv1 Weights (0-149)
            if (global_addr < C1_W_END) begin
                conv_w_addr <= global_addr[11:0];
                conv_w_data <= rx_data;
                conv_w_en <= 1;
            end
            // 2. Conv1 Biases (150-173)
            else if (global_addr < C1_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    conv_b_addr <= (global_addr - C1_B_START) >> 2;
                    conv_b_data <= {rx_data, bias_accum[23:0]};
                    conv_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 3. Conv2 Weights (174-2573)
            else if (global_addr < C2_W_END) begin
                conv_w_addr <= 150 + (global_addr - C2_W_START);
                conv_w_data <= rx_data;
                conv_w_en <= 1;
            end
            // 4. Conv2 Biases (2574-2637)
            else if (global_addr < C2_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    conv_b_addr <= 6 + ((global_addr - C2_B_START) >> 2);
                    conv_b_data <= {rx_data, bias_accum[23:0]};
                    conv_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 5. FC1 Weights (2638-50637)
            else if (global_addr < FC1_W_END) begin
                fc_w_addr <= (global_addr - FC1_W_START);
                fc_w_data <= rx_data;
                fc_w_en <= 1;
            end
            // 6. FC1 Biases (50638-51117)
            else if (global_addr < FC1_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    fc_b_addr <= (global_addr - FC1_B_START) >> 2;
                    fc_b_data <= {rx_data, bias_accum[23:0]};
                    fc_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 7. FC2 Weights (51118-61197)
            else if (global_addr < FC2_W_END) begin
                fc_w_addr <= 48000 + (global_addr - FC2_W_START);
                fc_w_data <= rx_data;
                fc_w_en <= 1;
            end
            // 8. FC2 Biases (61198-61533)
            else if (global_addr < FC2_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    fc_b_addr <= 120 + ((global_addr - FC2_B_START) >> 2);
                    fc_b_data <= {rx_data, bias_accum[23:0]};
                    fc_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 9. FC3 Weights (61534-62373)
            else if (global_addr < FC3_W_END) begin
                fc_w_addr <= 48000 + 10080 + (global_addr - FC3_W_START);
                fc_w_data <= rx_data;
                fc_w_en <= 1;
            end
            // 10. FC3 Biases (62374-62413)
            else if (global_addr < FC3_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    fc_b_addr <= 204 + ((global_addr - FC3_B_START) >> 2);
                    fc_b_data <= {rx_data, bias_accum[23:0]};
                    fc_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 11. Tanh LUT (62414-62669)
            else if (global_addr < TANH_END) begin
                tanh_addr <= global_addr - TANH_START;
                tanh_data <= rx_data;
                tanh_en <= 1;
            end

            // Update LED progress (show upper bits)
            led <= global_addr[16:1];

            global_addr <= global_addr + 1;
            if (global_addr >= TOTAL - 1) begin
                transfer_done <= 1;
                led <= 16'hFFFF;
            end
        end else begin
            conv_w_en <= 0; conv_b_en <= 0;
            fc_w_en <= 0; fc_b_en <= 0;
            tanh_en <= 0;
        end
    end
endmodule
