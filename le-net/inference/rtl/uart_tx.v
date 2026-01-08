/*
================================================================================
UART Transmitter Module - Handshake Fix
================================================================================
Update: "busy" signal is now Combinational logic.
This prevents a race condition where the consumer (scores_reader) would check
"busy" one cycle before the UART actually raised it, causing data overwrite
during burst transmissions.
================================================================================
*/

module uart_tx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input wire clk,
    input wire rst,
    input wire [7:0] data,       // Byte to transmit
    input wire send,             // Pulse to start transmission
    output reg tx,               // UART TX output line
    output wire busy             // HIGH while transmitting (Combinational)
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    // State definitions
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    reg [1:0] state;
    reg [15:0] clk_cnt;           // Clock counter
    reg [2:0] bit_cnt;            // Bit counter (0-7)
    reg [7:0] tx_shift;           // Shift register for outgoing bits
    reg tx_reg;                   // Registered TX value

    // === FIX: Combinational Busy Signal ===
    // Busy if we are not in IDLE state OR if we are commanded to send
    assign busy = (state != STATE_IDLE) || send;

    // UART transmit state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            tx <= 1'b1;  // Idle state is HIGH
            tx_shift <= 0;
        end else begin
            case (state)
                // ----------------------------------------
                // IDLE: Wait for send signal
                // ----------------------------------------
                STATE_IDLE: begin
                    tx <= 1'b1;
                    if (send) begin
                        state <= STATE_START;
                        tx_shift <= data;
                        clk_cnt <= 0;
                    end
                end
                
                // ----------------------------------------
                // START: Send start bit (LOW)
                // ----------------------------------------
                STATE_START: begin
                    tx <= 1'b0;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        bit_cnt <= 0;
                        state <= STATE_DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // DATA: Send 8 data bits (LSB first)
                // ----------------------------------------
                STATE_DATA: begin
                    tx <= tx_shift[bit_cnt];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        if (bit_cnt == 7) begin
                            bit_cnt <= 0;
                            state <= STATE_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // STOP: Send stop bit (HIGH)
                // ----------------------------------------
                STATE_STOP: begin
                    tx <= 1'b1;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        state <= STATE_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
