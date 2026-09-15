VERILATOR = verilator
TOP       = accel_top
RTL       = ../rtl/pe.sv ../rtl/pe_array.sv ../rtl/accel_top.sv
TB        = tb_accel.cpp

.PHONY: all sim clean

all: sim

obj_dir/V$(TOP): $(RTL) $(TB)
	$(VERILATOR) -Wall --trace --assert --cc --exe --build \
		--top-module $(TOP) -Mdir obj_dir \
		$(RTL) $(TB)

sim: obj_dir/V$(TOP)
	./obj_dir/V$(TOP)

clean:
	rm -rf obj_dir waveform.vcd
