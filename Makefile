# Three binaries, one per measurement: persistent (Phase 1 tile calibration),
# urgent_baseline (Phase 2 drain baseline), yield (Phases 3-4 instrument).
# Everything targets the A10 (sm_86); -lineinfo so Nsight maps back to source.
# `make sanitize` runs small configs under compute-sanitizer — do it after any
# kernel edit, before trusting numbers.
NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -lineinfo -Xcompiler -O3

all: bin/persistent bin/urgent_baseline bin/yield bin/midyield

bin/midyield: src/midyield.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bin/persistent: src/persistent.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bin/urgent_baseline: src/urgent_baseline.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bin/yield: src/yield.cu src/tile.cuh
	@mkdir -p bin results/raw
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Small configs under compute-sanitizer; run once before trusting any numbers.
sanitize: all
	compute-sanitizer --tool memcheck ./bin/persistent --tiles-m 8 --tiles-n 8 --passes 1 --csv results/raw/sanitize.csv
	compute-sanitizer --tool memcheck ./bin/urgent_baseline --tiles-m 8 --tiles-n 8 --bg-tasks 500 --trials 3
	compute-sanitizer --tool memcheck ./bin/yield --mode verify --tiles-m 8 --tiles-n 8 --tasks 3000
	compute-sanitizer --tool memcheck ./bin/midyield --mode verify --discipline drain --tiles-m 8 --tiles-n 8 --tasks 3000

clean:
	rm -rf bin

.PHONY: all sanitize clean
