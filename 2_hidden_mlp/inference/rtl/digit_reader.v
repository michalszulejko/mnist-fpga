/*
================================================================================
Digit Reader - Handles UART read requests for predicted digit
================================================================================
Protocol:
  Request: Send byte 0xCC to request predicted digit
  Response: FPGA sends 1 byte containing the predicted digit (0-9) in lower 4 bits
================================================================================
*/

module digit_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,        // UART RX data
    input wire rx_ready,              // UART RX ready signal
    input wire [7:0] digit_data,      // Read data from predicted_digit_ram
    output reg [7:0] tx_data,         // Data to send via UART TX
    output reg tx_send,               // Pulse to start UART TX transmission
    input wire tx_busy                // UART TX busy signal
);

    // Request byte constant
    localparam REQUEST_BYTE = 8'hCC;
    
    // States
    localparam STATE_IDLE = 2'd0;
    localparam STATE_SEND_RESPONSE = 2'd1;
    
    reg [1:0] state;
    reg rx_ready_prev;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
        end else begin
            // Default: tx_send is pulse
            tx_send <= 0;
            
            // Detect rising edge of rx_ready
            rx_ready_prev <= rx_ready;
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for read request (0xCC)
                // ----------------------------------------
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        // New byte received
                        if (rx_data == REQUEST_BYTE) begin
                            // Valid request - prepare response
                            tx_data <= digit_data;
                            state <= STATE_SEND_RESPONSE;
                        end
                    end
                end
                
                // ----------------------------------------
                // SEND_RESPONSE: Send the digit via UART TX
                // ----------------------------------------
                STATE_SEND_RESPONSE: begin
                    if (!tx_busy) begin
                        // UART TX is idle, send the data
                        tx_send <= 1;
                        state <= STATE_IDLE;
                    end
                end
                
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule

