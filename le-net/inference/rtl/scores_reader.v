/*
================================================================================
Scores Reader - Handles UART read requests for all class scores
================================================================================
Protocol:
  Request: Send byte 0xCD to request all class scores
  Response: FPGA sends 40 bytes containing all 10 scores (little-endian)
================================================================================
*/

module scores_reader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,        // UART RX data
    input wire rx_ready,              // UART RX ready signal
    input wire [7:0] scores_data,     // Read data from scores_ram
    output reg [5:0] scores_addr,     // Address to read from scores_ram (0-39)
    output reg [7:0] tx_data,         // Data to send via UART TX
    output reg tx_send,               // Pulse to start UART TX transmission
    input wire tx_busy                // UART TX busy signal
);

    // Request byte constant
    localparam REQUEST_BYTE = 8'hCD;
    localparam NUM_BYTES = 40;  // 10 scores × 4 bytes each
    
    // States
    localparam STATE_IDLE = 2'd0;
    localparam STATE_READ_BYTE = 2'd1;
    localparam STATE_SEND_BYTE = 2'd2;
    localparam STATE_WAIT_TX = 2'd3;
    
    reg [1:0] state;
    reg [5:0] byte_counter;  // 0 to 39
    reg rx_ready_prev;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            tx_data <= 0;
            tx_send <= 0;
            rx_ready_prev <= 0;
            byte_counter <= 0;
            scores_addr <= 0;
        end else begin
            // Default: tx_send is pulse
            tx_send <= 0;
            
            // Detect rising edge of rx_ready
            rx_ready_prev <= rx_ready;
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for read request (0xCD)
                // ----------------------------------------
                STATE_IDLE: begin
                    if (rx_ready && !rx_ready_prev) begin
                        // New byte received
                        if (rx_data == REQUEST_BYTE) begin
                            // Valid request - start reading scores
                            byte_counter <= 0;
                            scores_addr <= 0;
                            state <= STATE_READ_BYTE;
                        end
                    end
                end
                
                // ----------------------------------------
                // READ_BYTE: Request byte from scores_ram
                // ----------------------------------------
                STATE_READ_BYTE: begin
                    // Wait 1 cycle for BRAM read (synchronous read)
                    state <= STATE_SEND_BYTE;
                end
                
                // ----------------------------------------
                // SEND_BYTE: Send the byte via UART TX
                // ----------------------------------------
                STATE_SEND_BYTE: begin
                    // Data from scores_ram is now available
                    tx_data <= scores_data;
                    
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
                            scores_addr <= byte_counter + 1;
                            state <= STATE_READ_BYTE;
                        end else begin
                            // All 40 bytes sent
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



