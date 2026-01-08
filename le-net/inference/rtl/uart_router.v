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
    
    // NEW WEIGHT CALCULATION:
    // L1 Weights: 16 * 1 * 9 = 144
    // L1 Biases: 16 * 4 = 64
    // L2 Weights: 32 * 16 * 9 = 4608
    // L2 Biases: 32 * 4 = 128
    // Dense Weights: (32*5*5) * 10 = 8000
    // Dense Biases: 10 * 4 = 40
    // TOTAL = 12,984 bytes
    localparam WEIGHT_SIZE = 12984; 
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
                    else if ((rx_data == 8'hCC || rx_data == 8'hCD) && weights_loaded) begin
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
