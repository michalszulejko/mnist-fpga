/*
================================================================================
Debug Reader - Read back RAM values via UART for debugging
================================================================================
Commands:
  0xD0: Read 16 bytes from Tanh LUT (indices 120-135, around tanh(0) at 128)
  0xD1: Read 16 bytes from Conv1 weights (first 16 weights)
  0xD2: Read 16 bytes from Image RAM (first 16 pixels)
================================================================================
*/

module debug_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,
    input wire rx_ready,

    // Tanh LUT debug read interface
    output reg [7:0] tanh_dbg_addr,
    input wire [7:0] tanh_dbg_data,

    // Conv weights debug read interface
    output reg [11:0] conv_dbg_addr,
    input wire [7:0] conv_dbg_data,

    // Image RAM debug read interface
    output reg [9:0] img_dbg_addr,
    input wire [7:0] img_dbg_data,

    // UART TX interface
    output reg [7:0] tx_data,
    output reg tx_send,
    input wire tx_busy
);

    localparam CMD_READ_TANH = 8'hD0;
    localparam CMD_READ_CONV = 8'hD1;
    localparam CMD_READ_IMG  = 8'hD2;
    localparam NUM_BYTES = 16;

    localparam STATE_IDLE = 3'd0;
    localparam STATE_SETUP = 3'd1;
    localparam STATE_SEND = 3'd2;
    localparam STATE_WAIT = 3'd3;

    reg [2:0] state;
    reg [4:0] byte_counter;
    reg [1:0] cmd_type;  // 0=tanh, 1=conv, 2=image
    reg rx_ready_prev;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
            byte_counter <= 0;
            cmd_type <= 0;
            tanh_dbg_addr <= 0;
            conv_dbg_addr <= 0;
            img_dbg_addr <= 0;
        end else begin
            tx_send <= 0;
            rx_ready_prev <= rx_ready;

            case (state)
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        if (rx_data == CMD_READ_TANH) begin
                            cmd_type <= 2'd0;
                            byte_counter <= 0;
                            tanh_dbg_addr <= 120;  // Start at 120 (tanh(-8) to tanh(+7))
                            state <= STATE_SETUP;
                        end else if (rx_data == CMD_READ_CONV) begin
                            cmd_type <= 2'd1;
                            byte_counter <= 0;
                            conv_dbg_addr <= 0;  // First 16 conv1 weights
                            state <= STATE_SETUP;
                        end else if (rx_data == CMD_READ_IMG) begin
                            cmd_type <= 2'd2;
                            byte_counter <= 0;
                            img_dbg_addr <= 0;  // First 16 image pixels
                            state <= STATE_SETUP;
                        end
                    end
                end

                STATE_SETUP: begin
                    // Wait one cycle for combinational read to stabilize
                    state <= STATE_SEND;
                end

                STATE_SEND: begin
                    if (!tx_busy) begin
                        // Send the data based on command type
                        case (cmd_type)
                            2'd0: tx_data <= tanh_dbg_data;
                            2'd1: tx_data <= conv_dbg_data;
                            2'd2: tx_data <= img_dbg_data;
                            default: tx_data <= 8'h00;
                        endcase
                        tx_send <= 1;
                        state <= STATE_WAIT;
                    end
                end

                STATE_WAIT: begin
                    if (!tx_busy) begin
                        if (byte_counter < NUM_BYTES - 1) begin
                            byte_counter <= byte_counter + 1;
                            // Increment address based on command type
                            case (cmd_type)
                                2'd0: tanh_dbg_addr <= tanh_dbg_addr + 1;
                                2'd1: conv_dbg_addr <= conv_dbg_addr + 1;
                                2'd2: img_dbg_addr <= img_dbg_addr + 1;
                            endcase
                            state <= STATE_SETUP;
                        end else begin
                            state <= STATE_IDLE;
                        end
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
