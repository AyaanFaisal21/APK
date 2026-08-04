NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -lineinfo -Xcompiler -O3

all: bin/persistent bin/urgent_baseline

bin/persistent: src/persistent.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bin/urgent_baseline: src/urgent_baseline.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Small configs under compute-sanitizer; run once before trusting any numbers.
sanitize: all
	compute-sanitizer --tool memcheck ./bin/persistent --tiles-m 8 --tiles-n 8 --passes 1 --csv results/raw/sanitize.csv
	compute-sanitizer --tool memcheck ./bin/urgent_baseline --tiles-m 8 --tiles-n 8 --bg-tasks 500 --trials 3

clean:
	rm -rf bin

.PHONY: all sanitize clean
