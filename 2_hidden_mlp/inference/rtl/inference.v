/*
================================================================================
Three-Layer Neural Network Inference Module (Fixed Pipeline)
================================================================================
*/

module inference (
    input wire clk,
    input wire rst,
    
    // Layer 1 weight/bias memory interface
    output reg [13:0] L1_weight_addr,
    input wire [7:0]  L1_weight_data,
    output reg [3:0]  L1_bias_addr,
    input wire [31:0] L1_bias_data,
    
    // Layer 2 weight/bias memory interface
    output reg [7:0]  L2_weight_addr,
    input wire [7:0]  L2_weight_data,
    output reg [3:0]  L2_bias_addr,
    input wire [31:0] L2_bias_data,
    
    // Layer 3 weight/bias memory interface
    output reg [7:0]  L3_weight_addr,
    input wire [7:0]  L3_weight_data,
    output reg [3:0]  L3_bias_addr,
    input wire [31:0] L3_bias_data,
    
    // Control
    input wire        weights_ready,
    input wire        start_inference,
    input wire signed [7:0]  input_pixel,
    output reg [9:0]  input_addr,
    
    // Outputs
    output reg [3:0]  predicted_digit,
    output reg        inference_done,
    output reg        busy,
    
    // Class scores output
    output reg signed [31:0] class_score_0,
    output reg signed [31:0] class_score_1,
    output reg signed [31:0] class_score_2,
    output reg signed [31:0] class_score_3,
    output reg signed [31:0] class_score_4,
    output reg signed [31:0] class_score_5,
    output reg signed [31:0] class_score_6,
    output reg signed [31:0] class_score_7,
    output reg signed [31:0] class_score_8,
    output reg signed [31:0] class_score_9
);

    // State definitions
    localparam STATE_IDLE              = 5'd0;
    
    // Layer 1 (784 -> 16)
    localparam STATE_L1_WAIT_BIAS      = 5'd1;
    localparam STATE_L1_LOAD_BIAS      = 5'd2;
    localparam STATE_L1_WAIT_DATA      = 5'd3;  // Wait for RAM read
    localparam STATE_L1_COMPUTE        = 5'd4;
    localparam STATE_L1_FLUSH1         = 5'd5;
    localparam STATE_L1_FLUSH2         = 5'd6;  // Fully drain pipeline
    localparam STATE_L1_NEXT_NEURON    = 5'd7;
    
    // Layer 2 (16 -> 16)
    localparam STATE_L2_WAIT_BIAS      = 5'd8;
    localparam STATE_L2_LOAD_BIAS      = 5'd9;
    localparam STATE_L2_WAIT_DATA      = 5'd10; // Wait for RAM read
    localparam STATE_L2_COMPUTE        = 5'd11;
    localparam STATE_L2_FLUSH1         = 5'd12;
    localparam STATE_L2_FLUSH2         = 5'd13; // Fully drain pipeline
    localparam STATE_L2_NEXT_NEURON    = 5'd14;
    
    // Layer 3 (16 -> 10)
    localparam STATE_L3_WAIT_BIAS      = 5'd15;
    localparam STATE_L3_LOAD_BIAS      = 5'd16;
    localparam STATE_L3_WAIT_DATA      = 5'd17; // Wait for RAM read
    localparam STATE_L3_COMPUTE        = 5'd18;
    localparam STATE_L3_FLUSH1         = 5'd19;
    localparam STATE_L3_FLUSH2         = 5'd20; // Fully drain pipeline
    localparam STATE_L3_NEXT_CLASS     = 5'd21;
    localparam STATE_DONE              = 5'd22;

    // Parameters
    localparam NUM_PIXELS = 784;
    localparam L1_NEURONS = 16;
    localparam L2_NEURONS = 16;
    localparam NUM_CLASSES = 10;

    reg [4:0] state;
    reg [3:0] current_neuron;
    reg [3:0] current_class;
    reg [9:0] current_input;
    
    // Datapath Registers
    reg signed [31:0] accumulator;
    reg signed [31:0] current_bias;
    reg signed [7:0]  weight_reg;
    reg signed [31:0] input_reg;
    reg signed [39:0] product; // 32-bit * 8-bit = 40-bit result
    
    // Memories for intermediate results
    (* ram_style = "distributed" *) reg signed [31:0] L1_outputs [0:15];
    (* ram_style = "distributed" *) reg signed [31:0] L2_outputs [0:15];
    
    reg signed [31:0] class_scores [0:9];
    reg signed [31:0] max_score;
    reg [3:0] max_class;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            current_neuron <= 0;
            current_class <= 0;
            current_input <= 0;
            accumulator <= 0;
            current_bias <= 0;
            max_score <= 32'h80000000;
            max_class <= 0;
            predicted_digit <= 0;
            inference_done <= 0;
            busy <= 0;
            
            L1_weight_addr <= 0;
            L1_bias_addr <= 0;
            L2_weight_addr <= 0;
            L2_bias_addr <= 0;
            L3_weight_addr <= 0;
            L3_bias_addr <= 0;
            input_addr <= 0;
            
            weight_reg <= 0;
            input_reg <= 0;
            product <= 0;
            
            // Reset scores
            class_score_0 <= 0; class_score_1 <= 0; class_score_2 <= 0; class_score_3 <= 0;
            class_score_4 <= 0; class_score_5 <= 0; class_score_6 <= 0; class_score_7 <= 0;
            class_score_8 <= 0; class_score_9 <= 0;
            
        end else begin
            
            inference_done <= 0;
            
            case (state)
                STATE_IDLE: begin
                    busy <= 0;
                    if (start_inference && weights_ready) begin
                        state <= STATE_L1_WAIT_BIAS;
                        current_neuron <= 0;
                        current_input <= 0;
                        accumulator <= 0;
                        max_score <= 32'h80000000;
                        max_class <= 0;
                        busy <= 1;
                        L1_bias_addr <= 0;
                    end
                end
                
                // ============================================================
                // LAYER 1: 784 -> 16
                // ============================================================
                STATE_L1_WAIT_BIAS: state <= STATE_L1_LOAD_BIAS;
                
                STATE_L1_LOAD_BIAS: begin
                    current_bias <= $signed(L1_bias_data);
                    accumulator <= 0;
                    current_input <= 0;
                    // Reset pipeline
                    weight_reg <= 0;
                    input_reg <= 0;
                    product <= 0;
                    // Set initial address
                    L1_weight_addr <= current_neuron * NUM_PIXELS;
                    input_addr <= 0;
                    state <= STATE_L1_WAIT_DATA;
                end
                
                STATE_L1_WAIT_DATA: begin
                    // FIX: Pre-increment addresses here to align with BRAM latency
                    L1_weight_addr <= L1_weight_addr + 1;
                    input_addr <= input_addr + 1;
                    state <= STATE_L1_COMPUTE;
                end
                
                STATE_L1_COMPUTE: begin
                    // 1. Load Registers (from BRAM output of Previous Addr)
                    weight_reg <= $signed(L1_weight_data);
                    input_reg <= $signed(input_pixel); 
                    
                    // 2. Multiply (using registers from previous cycle)
                    product <= $signed(weight_reg) * $signed(input_reg[7:0]);
                    
                    // 3. Accumulate (using product from previous cycle)
                    // FIX: Cleaned up accumulation to use standard veriloag signed math
                    accumulator <= accumulator + product;
                    
                    if (current_input < NUM_PIXELS - 1) begin
                        current_input <= current_input + 1;
                        L1_weight_addr <= L1_weight_addr + 1;
                        input_addr <= input_addr + 1;
                    end else begin
                        state <= STATE_L1_FLUSH1;
                    end
                end
                
                STATE_L1_FLUSH1: begin
                    // Pipeline drain 1: Calc last product, Acc second-to-last
                    weight_reg <= $signed(L1_weight_data);
                    input_reg <= $signed(input_pixel);
                    product <= $signed(weight_reg) * $signed(input_reg[7:0]);
                    accumulator <= accumulator + product;
                    state <= STATE_L1_FLUSH2;
                end
                
                STATE_L1_FLUSH2: begin
                    // Pipeline drain 2: Acc last product
                    product <= $signed(weight_reg) * $signed(input_reg[7:0]);
                    accumulator <= accumulator + product;
                    state <= STATE_L1_NEXT_NEURON;
                end
                
                STATE_L1_NEXT_NEURON: begin : l1_next_block
                    reg signed [31:0] final_result;
                    reg signed [31:0] shifted_result;
                    
                    // Accumulator now holds the COMPLETE sum. Just add bias.
                    final_result = accumulator + current_bias;
                    
                    // Right shift by 7
                    shifted_result = final_result >>> 7;
                    
                    // ReLU
                    if (shifted_result[31] == 1'b1) L1_outputs[current_neuron] <= 32'd0;
                    else L1_outputs[current_neuron] <= shifted_result;
                    
                    if (current_neuron < L1_NEURONS - 1) begin
                        current_neuron <= current_neuron + 1;
                        L1_bias_addr <= current_neuron + 1;
                        state <= STATE_L1_WAIT_BIAS;
                    end else begin
                        current_neuron <= 0;
                        L2_bias_addr <= 0;
                        state <= STATE_L2_WAIT_BIAS;
                    end
                end
                
                // ============================================================
                // LAYER 2: 16 -> 16
                // ============================================================
                STATE_L2_WAIT_BIAS: state <= STATE_L2_LOAD_BIAS;
                
                STATE_L2_LOAD_BIAS: begin
                    current_bias <= $signed(L2_bias_data);
                    accumulator <= 0;
                    current_input <= 0;
                    weight_reg <= 0;
                    input_reg <= 0;
                    product <= 0;
                    L2_weight_addr <= (current_neuron << 4); 
                    state <= STATE_L2_WAIT_DATA;
                end
                
                STATE_L2_WAIT_DATA: begin
                    // FIX: Pre-increment address for alignment
                    L2_weight_addr <= L2_weight_addr + 1;
                    state <= STATE_L2_COMPUTE;
                end
                
                STATE_L2_COMPUTE: begin
                    weight_reg <= $signed(L2_weight_data);
                    input_reg <= L1_outputs[current_input];
                    
                    product <= $signed(weight_reg) * $signed(input_reg);
                    accumulator <= accumulator + product; // Use full product width
                    
                    if (current_input < L2_NEURONS - 1) begin
                        current_input <= current_input + 1;
                        L2_weight_addr <= L2_weight_addr + 1;
                    end else begin
                        state <= STATE_L2_FLUSH1;
                    end
                end
                
                STATE_L2_FLUSH1: begin
                    // Pipeline drain 1
                    weight_reg <= $signed(L2_weight_data);
                    input_reg <= L1_outputs[current_input];
                    product <= $signed(weight_reg) * $signed(input_reg);
                    accumulator <= accumulator + product;
                    state <= STATE_L2_FLUSH2;
                end

                STATE_L2_FLUSH2: begin
                    // Pipeline drain 2
                    accumulator <= accumulator + product;
                    state <= STATE_L2_NEXT_NEURON;
                end
                
                STATE_L2_NEXT_NEURON: begin : l2_next_block
                    reg signed [31:0] final_result;
                    reg signed [31:0] shifted_result;
                    
                    final_result = accumulator + current_bias;
                    
                    shifted_result = final_result >>> 7;
                    
                    if (shifted_result[31] == 1'b1) L2_outputs[current_neuron] <= 32'd0;
                    else L2_outputs[current_neuron] <= shifted_result;
                    
                    if (current_neuron < L2_NEURONS - 1) begin
                        current_neuron <= current_neuron + 1;
                        L2_bias_addr <= current_neuron + 1;
                        state <= STATE_L2_WAIT_BIAS;
                    end else begin
                        current_class <= 0;
                        L3_bias_addr <= 0;
                        state <= STATE_L3_WAIT_BIAS;
                    end
                end
                
                // ============================================================
                // LAYER 3: 16 -> 10
                // ============================================================
                STATE_L3_WAIT_BIAS: state <= STATE_L3_LOAD_BIAS;
                
                STATE_L3_LOAD_BIAS: begin
                    current_bias <= $signed(L3_bias_data);
                    accumulator <= 0;
                    current_input <= 0;
                    weight_reg <= 0;
                    input_reg <= 0;
                    product <= 0;
                    L3_weight_addr <= (current_class << 4);
                    state <= STATE_L3_WAIT_DATA;
                end
                
                STATE_L3_WAIT_DATA: begin
                    // FIX: Pre-increment address for alignment
                    L3_weight_addr <= L3_weight_addr + 1;
                    state <= STATE_L3_COMPUTE;
                end
                
                STATE_L3_COMPUTE: begin
                    weight_reg <= $signed(L3_weight_data);
                    input_reg <= L2_outputs[current_input];
                    
                    product <= $signed(weight_reg) * $signed(input_reg);
                    accumulator <= accumulator + product; // Use full product width
                    
                    if (current_input < L2_NEURONS - 1) begin
                        current_input <= current_input + 1;
                        L3_weight_addr <= L3_weight_addr + 1;
                    end else begin
                        state <= STATE_L3_FLUSH1;
                    end
                end
                
                STATE_L3_FLUSH1: begin
                    weight_reg <= $signed(L3_weight_data);
                    input_reg <= L2_outputs[current_input];
                    product <= $signed(weight_reg) * $signed(input_reg);
                    accumulator <= accumulator + product;
                    state <= STATE_L3_FLUSH2;
                end

                STATE_L3_FLUSH2: begin
                    accumulator <= accumulator + product;
                    state <= STATE_L3_NEXT_CLASS;
                end
                
                STATE_L3_NEXT_CLASS: begin : l3_next_block
                    reg signed [31:0] final_score;
                    
                    final_score = accumulator + current_bias;
                    class_scores[current_class] <= final_score;
                    
                    if (final_score > max_score) begin
                        max_score <= final_score;
                        max_class <= current_class;
                    end
                    
                    if (current_class < NUM_CLASSES - 1) begin
                        current_class <= current_class + 1;
                        L3_bias_addr <= current_class + 1;
                        state <= STATE_L3_WAIT_BIAS;
                    end else begin
                        state <= STATE_DONE;
                    end
                end
                
                STATE_DONE: begin
                    predicted_digit <= max_class;
                    inference_done <= 1;
                    busy <= 0;
                    
                    // Update output ports
                    class_score_0 <= class_scores[0];
                    class_score_1 <= class_scores[1];
                    class_score_2 <= class_scores[2];
                    class_score_3 <= class_scores[3];
                    class_score_4 <= class_scores[4];
                    class_score_5 <= class_scores[5];
                    class_score_6 <= class_scores[6];
                    class_score_7 <= class_scores[7];
                    class_score_8 <= class_scores[8];
                    class_score_9 <= class_scores[9];
                    
                    state <= STATE_IDLE;
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
