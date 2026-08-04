NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -lineinfo -Xcompiler -O3

all: bin/persistent

bin/persistent: src/persistent.cu
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Small config under compute-sanitizer; run once before trusting any numbers.
sanitize: bin/persistent
	compute-sanitizer --tool memcheck ./bin/persistent --tiles-m 8 --tiles-n 8 --passes 1 --csv results/raw/sanitize.csv

clean:
	rm -rf bin

.PHONY: all sanitize clean
