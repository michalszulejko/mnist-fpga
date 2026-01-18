module uart_router (
    input wire clk,
    input wire rst,
    input wire rx,
    input wire weights_loaded,
    output reg [7:0] weight_rx_data,
    output reg weight_rx_ready,
    output reg [7:0] image_rx_data,
    output reg image_rx_ready,
    output reg [7:0] cmd_rx_data,
    output reg cmd_rx_ready
);
    // Protocol Markers
    localparam WEIGHT_START1 = 8'hAA, WEIGHT_START2 = 8'h55;
    localparam WEIGHT_END1   = 8'h55, WEIGHT_END2   = 8'hAA;
    localparam IMAGE_START1  = 8'hBB, IMAGE_START2 = 8'h66;
    localparam IMAGE_END1    = 8'h66, IMAGE_END2   = 8'hBB;

    // LeNet-5 Weight Calculation:
    // Conv1 Weights: 6*1*5*5 = 150
    // Conv1 Biases:  6*4 = 24
    // Conv2 Weights: 16*6*5*5 = 2400
    // Conv2 Biases:  16*4 = 64
    // FC1 Weights:   120*400 = 48000
    // FC1 Biases:    120*4 = 480
    // FC2 Weights:   84*120 = 10080
    // FC2 Biases:    84*4 = 336
    // FC3 Weights:   10*84 = 840
    // FC3 Biases:    10*4 = 40
    // Tanh LUT:      256
    // TOTAL = 62,670 bytes
    localparam WEIGHT_SIZE = 62670;
    localparam IMAGE_SIZE = 784;

    wire [7:0] rx_data;
    wire rx_ready;
    uart_rx u_rx (.clk(clk), .rst(rst), .rx(rx), .data(rx_data), .ready(rx_ready));

    reg [3:0] state;
    reg [15:0] byte_count;
    reg [7:0] prev_byte;

    localparam IDLE = 0, WAIT_W2 = 1, RECV_W = 2, DONE_W = 3, WAIT_I2 = 4, RECV_I = 5, DONE_I = 6;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            byte_count <= 0;
            prev_byte <= 0;
            weight_rx_ready <= 0;
            image_rx_ready <= 0;
            cmd_rx_ready <= 0;
        end else begin
            weight_rx_ready <= 0;
            image_rx_ready <= 0;
            cmd_rx_ready <= 0;

            case (state)
                IDLE: if (rx_ready) begin
                    if (rx_data == WEIGHT_START1 && !weights_loaded) state <= WAIT_W2;
                    else if (rx_data == IMAGE_START1 && weights_loaded) state <= WAIT_I2;
                    // Command forwarding: 0xCC (digit), 0xCD (scores), 0xD0/0xD1/0xD2 (debug)
                    else if ((rx_data == 8'hCC || rx_data == 8'hCD || 
                              rx_data == 8'hD0 || rx_data == 8'hD1 || rx_data == 8'hD2) && weights_loaded) begin
                        cmd_rx_data <= rx_data;
                        cmd_rx_ready <= 1;
                    end
                end

                WAIT_W2: if (rx_ready) begin
                    if (rx_data == WEIGHT_START2) begin
                        state <= RECV_W;
                        byte_count <= 0;
                        // Do NOT forward the 0x55 marker byte to the weight loader
                    end else state <= IDLE;
                end

                RECV_W: if (rx_ready) begin
                    byte_count <= byte_count + 1;
                    prev_byte <= rx_data;
                    
                    // Only forward bytes that are part of the payload
                    if (byte_count < WEIGHT_SIZE) begin
                        weight_rx_data <= rx_data;
                        weight_rx_ready <= 1;
                    end
                    
                    // Check for end markers only after payload complete
                    if (byte_count >= WEIGHT_SIZE && prev_byte == WEIGHT_END1 && rx_data == WEIGHT_END2) begin
                        state <= DONE_W;
                    end
                end
                
                DONE_W: begin
                    // Stay here until reset or explicit command
                    // This prevents stray bytes from triggering image loading
                    state <= IDLE;
                end

                WAIT_I2: if (rx_ready) begin
                    if (rx_data == IMAGE_START2) begin
                        state <= RECV_I;
                        byte_count <= 0;
                        prev_byte <= 0;
                        // Do NOT forward the 0x66 marker byte to the image loader
                    end else state <= IDLE;
                end

                RECV_I: if (rx_ready) begin
                    byte_count <= byte_count + 1;
                    prev_byte <= rx_data;
                    
                    // --- FIX IS HERE ---
                    // Allow Payload (784) + 2 Marker bytes to pass to the loader
                    if (byte_count < IMAGE_SIZE + 2) begin
                        image_rx_data <= rx_data;
                        image_rx_ready <= 1;
                    end else begin
                        image_rx_ready <= 0;
                    end
                    
                    if (byte_count >= IMAGE_SIZE && prev_byte == IMAGE_END1 && rx_data == IMAGE_END2) begin
                        state <= DONE_I;
                    end
                end
                
                DONE_I: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
