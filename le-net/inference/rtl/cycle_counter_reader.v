/*
================================================================================
Cycle Counter Reader - Handles UART read requests for inference cycle count
================================================================================
Protocol:
  Request: Send byte 0xCE to request cycle count
  Response: FPGA sends 4 bytes containing the cycle count (little-endian)
================================================================================
*/

module cycle_counter_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,        // UART RX data
    input wire rx_ready,              // UART RX ready signal
    input wire [31:0] cycle_count,    // Cycle counter value from inference module
    output reg [7:0] tx_data,         // Data to send via UART TX
    output reg tx_send,               // Pulse to start UART TX transmission
    input wire tx_busy                // UART TX busy signal
);

    // Request byte constant
    localparam REQUEST_BYTE = 8'hCE;
    localparam NUM_BYTES = 4;  // 32-bit value = 4 bytes

    // States
    localparam STATE_IDLE = 2'd0;
    localparam STATE_READ_BYTE = 2'd1;
    localparam STATE_SEND_BYTE = 2'd2;
    localparam STATE_WAIT_TX = 2'd3;

    reg [1:0] state;
    reg [1:0] byte_counter;  // 0 to 3
    reg rx_ready_prev;
    reg [31:0] counter_latch;  // Latched cycle count value

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
            byte_counter <= 0;
            counter_latch <= 0;
        end else begin
            // Default: tx_send is pulse
            tx_send <= 0;

            // Detect rising edge of rx_ready
            rx_ready_prev <= rx_ready;

            case (state)
                // ----------------------------------------
                // IDLE: Wait for read request (0xCE)
                // ----------------------------------------
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        // New byte received
                        if (rx_data == REQUEST_BYTE) begin
                            // Valid request - latch the cycle count and start sending
                            counter_latch <= cycle_count;
                            byte_counter <= 0;
                            state <= STATE_READ_BYTE;
                        end
                    end
                end

                // ----------------------------------------
                // READ_BYTE: Select the appropriate byte to send
                // ----------------------------------------
                STATE_READ_BYTE: begin
                    // Mux the appropriate byte (little-endian: LSB first)
                    case (byte_counter)
                        2'd0: tx_data <= counter_latch[7:0];
                        2'd1: tx_data <= counter_latch[15:8];
                        2'd2: tx_data <= counter_latch[23:16];
                        2'd3: tx_data <= counter_latch[31:24];
                    endcase
                    state <= STATE_SEND_BYTE;
                end

                // ----------------------------------------
                // SEND_BYTE: Send the byte via UART TX
                // ----------------------------------------
                STATE_SEND_BYTE: begin
                    if (!tx_busy) begin
                        // UART TX is idle, send the data
                        tx_send <= 1;
                        state <= STATE_WAIT_TX;
                    end
                    // If TX is busy, wait in this state
                end

                // ----------------------------------------
                // WAIT_TX: Wait for transmission to complete
                // ----------------------------------------
                STATE_WAIT_TX: begin
                    if (!tx_busy) begin
                        // Transmission complete
                        if (byte_counter < NUM_BYTES - 1) begin
                            // More bytes to send
                            byte_counter <= byte_counter + 1;
                            state <= STATE_READ_BYTE;
                        end else begin
                            // All 4 bytes sent
                            state <= STATE_IDLE;
                        end
                    end
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule
