# Compiler
NVCC = nvcc

# The directory where the binaries will be stored
OUT_DIR = ./build

# Source files
SRC = $(wildcard *.cu)

# Output files
OBJ = $(basename $(SRC))

# Targets
all: directories $(OUT_DIR)/dotprod $(OUT_DIR)/dotprod_no_fma $(OUT_DIR)/dotprod_all_fma

directories:
	mkdir -p $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)

$(OUT_DIR)/dotprod: dotprod.cu
	$(NVCC) -o $@ $<

$(OUT_DIR)/dotprod_no_fma: dotprod.cu
	$(NVCC) -fmad=false -o $@ $<

$(OUT_DIR)/dotprod_all_fma: dotprod.cu
	$(NVCC) -fmad=true -Xcompiler '-mfma -ffp-contract=fast -O3' -o $@ $<
