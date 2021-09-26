#include <iostream>

const unsigned max_block_size = 1024;

__global__ void kernel_all_threads_malloc(void** ptr /*For side effect*/) {
	const auto tid = threadIdx.x;

	const auto t0 = clock64();
	ptr[tid] = malloc(sizeof(float));
	const auto t1 = clock64();

	const auto t2 = clock64();
	free(ptr[tid]);
	const auto t3 = clock64();


	if (threadIdx.x == 0) {
		printf("%u,all,%llu,%llu\n", blockDim.x, t1 - t0, t3 - t2);
	}
}

__global__ void kernel_one_thread_malloc(void** ptr /*For side effect*/) {
	const auto tid = threadIdx.x;

	if (threadIdx.x == 0) {
		const auto t0 = clock64();
		ptr[tid] = malloc(sizeof(float) * blockDim.x);
		const auto t1 = clock64();

		const auto t2 = clock64();
		free(ptr[tid]);
		const auto t3 = clock64();

		printf("%u,one,%llu,%llu\n", blockDim.x, t1 - t0, t3 - t2);
	}
}

int main() {
	std::printf("block_size,malloc_mode,malloc_clock,free_clock\n");
	float **ptr;
	for (unsigned b = 1; b < max_block_size; b++) {
		cudaMalloc(&ptr, sizeof(float*) * max_block_size);
		kernel_all_threads_malloc<<<1, b>>>((void**)ptr);
		cudaFree(ptr);
	}
	for (unsigned b = 1; b <= max_block_size; b++) {
		cudaMalloc(&ptr, sizeof(float*) * max_block_size);
		kernel_one_thread_malloc<<<1, b>>>((void**)ptr);
		cudaFree(ptr);
	}
	cudaDeviceSynchronize();
}
