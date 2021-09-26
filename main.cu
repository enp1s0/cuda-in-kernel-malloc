#include <iostream>
#include <chrono>

const unsigned max_block_size = 1024;

__global__ void kernel_all_threads_malloc(void** ptr /*For side effect*/,
		unsigned long long* total_malloc_clock,
		unsigned long long* total_free_clock
		) {
	const auto tid = threadIdx.x;
	void* p;

	const auto t0 = clock64();
	p = malloc(sizeof(float));
	const auto t1 = clock64();
	ptr[tid] = p;

	const auto t2 = clock64();
	free(p);
	const auto t3 = clock64();

	atomicAdd(total_malloc_clock, t1 - t0);
	atomicAdd(total_free_clock, t3 - t2);
}

__global__ void kernel_one_thread_malloc(void** ptr /*For side effect*/,
		unsigned long long* total_malloc_clock,
		unsigned long long* total_free_clock
		) {
	const auto tid = threadIdx.x;
	void* p;

	if (tid == 0) {
		const auto t0 = clock64();
		p = malloc(sizeof(float) * blockDim.x);
		const auto t1 = clock64();
		ptr[tid] = p;

		const auto t2 = clock64();
		free(p);
		const auto t3 = clock64();

		atomicAdd(total_malloc_clock, t1 - t0);
		atomicAdd(total_free_clock, t3 - t2);
	}
}

void in_block_test() {
	std::printf("block_size,mode,malloc_clock,free_clock,kernel_time\n");
	float **ptr;
	cudaMalloc(&ptr, sizeof(float*) * max_block_size);
	unsigned long long *total_malloc_clock, *total_free_clock;
	cudaMallocHost(&total_malloc_clock, sizeof(unsigned long long));
	cudaMallocHost(&total_free_clock, sizeof(unsigned long long));

	for (unsigned b = 1; b <= max_block_size; b++) {
		*total_malloc_clock = 0llu;
		*total_free_clock = 0llu;
		cudaDeviceSynchronize();
		const auto t0 = std::chrono::system_clock::now();
		kernel_all_threads_malloc<<<1, b>>>((void**)ptr, total_malloc_clock, total_free_clock);
		const auto t1 = std::chrono::system_clock::now();
		cudaDeviceSynchronize();
		std::printf("%u,all,%e,%e,%lu\n", b,
				static_cast<double>(*total_malloc_clock) / b,
				static_cast<double>(*total_free_clock) / b,
				std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
	}
	for (unsigned b = 1; b <= max_block_size; b++) {
		*total_malloc_clock = 0llu;
		*total_free_clock = 0llu;
		cudaDeviceSynchronize();
		const auto t0 = std::chrono::system_clock::now();
		kernel_one_thread_malloc<<<1, b>>>((void**)ptr, total_malloc_clock, total_free_clock);
		const auto t1 = std::chrono::system_clock::now();
		cudaDeviceSynchronize();
		std::printf("%u,one,%e,%e,%lu\n", b,
				static_cast<double>(*total_malloc_clock) / b,
				static_cast<double>(*total_free_clock) / b,
				std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
	}
	cudaFreeHost(total_malloc_clock);
	cudaFreeHost(total_free_clock);
	cudaFree(ptr);
}

int main() {
	in_block_test();
}
