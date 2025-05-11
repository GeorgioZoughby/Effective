#include "VectorCuda.cuh"

template <typename T>
__global__ void copyKernel(const T* from, T* to, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        to[i] = from[i];
    }
}


template <typename Op, typename T>
__global__ void scalarOpKernel(T* elts, const T scal, int size, Op op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        op(elts[i], scal);
    }
}


template <typename T>
__global__ void compareEqKernel(const T* elts1, const T* elts2, bool* equality, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        if (elts1[i] != elts2[i]) {
            *equality = false;
        }
    }
}


template <typename T>
__global__ void dotKernel(const T* elts1, const T* elts2, T* result, unsigned int size)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    T sum = 0;

    // Each thread accumulates its own partial sum
    for (int i = index; i < size; i += stride) {
        sum += elts1[i] * elts2[i];
    }

    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result atomically to global result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


template <typename T>
__global__ void clearKernel(T* elts, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        elts[i] = T();  // reset to default
    }
}

template <typename Op, typename T>
__global__ void vectorOpKernel(T* elts1, const T* elts2, unsigned int size, Op op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        op(elts1[index], elts2[index]);
    }
}





using namespace expr;
//#include <chrono>

//int main() {
//    Vector<int> v;
//
//    // Start measuring time
//    auto start = std::chrono::high_resolution_clock::now();
//
//    std::cout << "Pushing 3 elements...\n";
//    v.push_back(10);
//    v.push_back(20);
//    v.push_back(30);
//
//    std::cout << "Front: " << v.front() << "\n";
//    std::cout << "Back: " << v.back() << "\n";
//    std::cout << "At(1): " << v.at(1) << "\n";
//
//    std::cout << "Using operator[]: ";
//    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
//    std::cout << "\n";
//
//    std::cout << "Size: " << v.size() << ", Capacity: " << v.capacity() << "\n";
//
//    std::cout << "Pop back...\n";
//    v.pop_back();
//    std::cout << "Size: " << v.size() << ", Back: " << v.back() << "\n";
//
//    std::cout << "Resizing to 5 with default value 99...\n";
//    v.resize(5, 99);
//    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
//    std::cout << "\n";
//
//    std::cout << "Clearing vector...\n";
//    v.clear();
//    std::cout << "Size after clear: " << v.size() << ", Empty? " << v.empty() << "\n";
//
//    std::cout << "Testing iterators (push then iterate):\n";
//    for (int i = 1; i <= 5; ++i) v.push_back(i * 10);
//    for (auto it = v.begin(); it != v.end(); ++it)
//        std::cout << *it << " ";
//    std::cout << "\n";
//
//    Vector<int> v1;
//    for (int i = 0; i < 5; ++i) v1.push_back(i + 1);
//
//    // Arithmetic expressions
//    Vector<int> sum = v + v1;
//    Vector<int> diff = v - v1;
//    Vector<int> prod = v * v1;
//    Vector<int> scaled = v * 2;
//    Vector<int> shifted = v + 5;
//
//    std::cout << "v + v1: ";
//    for (int i = 0; i < sum.size(); ++i) std::cout << sum[i] << " ";
//    std::cout << "\nv - v1: ";
//    for (int i = 0; i < diff.size(); ++i) std::cout << diff[i] << " ";
//    std::cout << "\nv * v1: ";
//    for (int i = 0; i < prod.size(); ++i) std::cout << prod[i] << " ";
//    std::cout << "\nv * 2: ";
//    for (int i = 0; i < scaled.size(); ++i) std::cout << scaled[i] << " ";
//    std::cout << "\nv + 5: ";
//    for (int i = 0; i < shifted.size(); ++i) std::cout << shifted[i] << " ";
//    std::cout << "\n";
//    std::cout << "v + 5: size = " << shifted.size() << "\n";
//
//    std::cout << "Dot product v . v1: " << v.dot(v1) << "\n";
//
//    std::cout << "Equality test: v == v1? " << (v == v1 ? "true" : "false") << "\n";
//    std::cout << "Inequality test: v != v1? " << (v != v1 ? "true" : "false") << "\n";
//
//    // End measuring time
//    auto end = std::chrono::high_resolution_clock::now();
//
//    // Calculate the duration
//    std::chrono::duration<double> duration = end - start;
//    std::cout << "Time taken: " << duration.count() << " seconds\n";
//
//    return 0;
//}
//Time measure before pinned memory transfer: 1.33s
//Time measure after pinned memory transfer: 0.27s



// =======================
// Kernel Wrapper Functions with Streams
// =======================

// Each wrapper now uses a cudaStream_t for asynchronous operations

// Helper macro to define stream and clean up
#define CREATE_CUDA_STREAM(stream) \
    cudaStream_t stream; \
    cudaStreamCreate(&stream);

#define DESTROY_CUDA_STREAM(stream) \
    cudaStreamSynchronize(stream); \
    cudaStreamDestroy(stream);

template <typename T>
cudaError_t cuda_copy(const T* from, T* to, unsigned int size) {
    T* dev_from = nullptr;
    T* dev_to = nullptr;
    T* pinned_from = nullptr;
    T* pinned_to = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_from, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_to, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy data into pinned buffer
    memcpy(pinned_from, from, size * sizeof(T));

    // Allocate device memory
    cudaStatus = cudaMalloc((void**)&dev_from, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_to, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from host to device
    cudaStatus = cudaMemcpyAsync(dev_from, pinned_from, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Kernel launch
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    copyKernel<<<gridSize, blockSize, 0, stream>>>(dev_from, dev_to, size);

    // Async transfer from device to pinned host
    cudaStatus = cudaMemcpyAsync(pinned_to, dev_to, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Synchronize stream and copy to user buffer
    cudaStreamSynchronize(stream);
    memcpy(to, pinned_to, size * sizeof(T));

Error:
    // Clean up
    DESTROY_CUDA_STREAM(stream);
    if (dev_from) cudaFree(dev_from);
    if (dev_to) cudaFree(dev_to);
    if (pinned_from) cudaFreeHost(pinned_from);
    if (pinned_to) cudaFreeHost(pinned_to);

    return cudaStatus;
}

template <typename Op, typename T>
cudaError_t cuda_vector_op(T* elts1, const T* elts2, unsigned int size, Op op) {
    T* dev_elts1 = nullptr;
    T* dev_elts2 = nullptr;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    // Allocate device memory
    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from pinned memory to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    vectorOpKernel<<<numBlocks, blockSize, 0, stream>>>(dev_elts1, dev_elts2, size, op);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts1, dev_elts1, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Synchronize and copy from pinned memory to output
    cudaStreamSynchronize(stream);
    memcpy(elts1, pinned_elts1, size * sizeof(T));

Error:
    // Clean up
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}



template <typename Op, typename T>
cudaError_t cuda_scalar_op(T* elts, const T scal, unsigned int size, Op op) {
    T* dev_elts = 0;
    T* pinned_elts = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned host memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts, elts, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer from pinned memory to device
    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    scalarOpKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, scal, size, op);

    // Async transfer from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    Error:
        // Synchronize and copy from pinned memory to output
        cudaStreamSynchronize(stream);
    memcpy(elts, pinned_elts, size * sizeof(T));

    // Clean up
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts);
    cudaFreeHost(pinned_elts);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_compare_equality(const T* elts1, const T* elts2, unsigned int size, bool* equality) {
    T* dev_elts1 = 0;
    T* dev_elts2 = 0;
    bool* dev_equality = 0;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_equality, sizeof(bool));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemsetAsync(dev_equality, 1, sizeof(bool), stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    compareEqKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts1, dev_elts2, dev_equality, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(equality, dev_equality, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    // Synchronize and clean up
    cudaStreamSynchronize(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_equality);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_dot(const T* elts1, const T* elts2, unsigned int size, T* result) {
    T* dev_elts1 = 0;
    T* dev_elts2 = 0;
    T* dev_result = 0;
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts1, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaHostAlloc((void**)&pinned_elts2, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_result, sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemsetAsync(dev_result, 0, sizeof(T), stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(T);
    dotKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(dev_elts1, dev_elts2, dev_result, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(result, dev_result, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

Error:
    // Synchronize and clean up
    cudaStreamSynchronize(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_result);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}




template <typename T>
cudaError_t cuda_clear(T* elts, unsigned int size) {
    T* dev_elts = 0;
    T* pinned_elts = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaHostAlloc((void**)&pinned_elts, size * sizeof(T), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy input data into pinned memory
    memcpy(pinned_elts, elts, size * sizeof(T));

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Async transfer to device
    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    clearKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, size);

    // Async transfer back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    Error:
        // Synchronize and clean up
        cudaStreamSynchronize(stream);
        memcpy(elts, pinned_elts, size * sizeof(T));
        cudaFree(dev_elts);
        cudaFreeHost(pinned_elts);
        return cudaStatus;
}
