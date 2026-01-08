/*
================================================================================
Image RAM (784 bytes) - Distributed/Asynchronous
================================================================================
*/

module image_ram (
    input wire clk,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [9:0] rd_addr,
    output wire [7:0] rd_data // Changed to wire
);

    (* ram_style = "distributed" *) reg [7:0] ram [0:783];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Asynchronous read
    assign rd_data = ram[rd_addr];

endmodule



