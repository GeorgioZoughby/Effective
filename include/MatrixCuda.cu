#include "MatrixCuda.cuh"


// CUDA Kernel Implementations
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
__global__ void dotKernel(const T* elts1, const T* elts2, T* result, unsigned int size) {
    extern __shared__ char smem[];
    T* sdata = reinterpret_cast<T*>(smem);  // Cast shared memory to correct type


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
__global__ void matrixOpKernel(T* elts1, const T* elts2, unsigned int size, Op op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        op(elts1[i], elts2[i]);
    }
}

template <typename T>
__global__ void matrixMultiplyKernel(const T* A, const T* B, T* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        T sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

template <typename T>
__global__ void transposeKernel(const T* input, T* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// CUDA Interface Implementation
template <typename T>
cudaError_t cuda_matrix_copy(const T* from, T* to, unsigned int size) {
    T* dev_from = nullptr;
    T* dev_to = nullptr;
    cudaError_t cudaStatus;

    // Allocate pinned memory
    T* pinned_from = nullptr;
    T* pinned_to = nullptr;
    cudaStatus = cudaMallocHost((void**)&pinned_from, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMallocHost((void**)&pinned_to, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_from, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_to, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy from host to pinned memory
    memcpy(pinned_from, from, size * sizeof(T));

    cudaStatus = cudaMemcpyAsync(dev_from, pinned_from, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    copyKernel<<<gridSize, blockSize, 0, stream>>>(dev_from, dev_to, size);

    cudaStatus = cudaMemcpyAsync(pinned_to, dev_to, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(to, pinned_to, size * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_to);
    cudaFree(dev_from);
    cudaFreeHost(pinned_from);
    cudaFreeHost(pinned_to);
    return cudaStatus;
}


template <typename Op, typename T>
cudaError_t cuda_matrix_op(T* elts1, const T* elts2, unsigned int size, Op op) {
    T* d_elts1 = nullptr;
    T* d_elts2 = nullptr;
    cudaError_t cudaStatus;

    // Allocate pinned memory
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    cudaStatus = cudaMallocHost((void**)&pinned_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMallocHost((void**)&pinned_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&d_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&d_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMemcpyAsync(d_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(d_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    matrixOpKernel<<<gridSize, blockSize, 0, stream>>>(d_elts1, d_elts2, size, op);

    cudaStatus = cudaMemcpyAsync(pinned_elts1, d_elts1, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(elts1, pinned_elts1, size * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(d_elts1);
    cudaFree(d_elts2);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    return cudaStatus;
}

template <typename Op, typename T>
cudaError_t cuda_matrix_scalar_op(T* elts, const T scalar, unsigned int size, Op op) {
    T* dev_elts = nullptr;
    cudaError_t cudaStatus;

    // Allocate pinned memory
    T* pinned_elts = nullptr;
    cudaStatus = cudaMallocHost((void**)&pinned_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(pinned_elts, elts, size * sizeof(T));

    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    scalarOpKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, scalar, size, op);

    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(elts, pinned_elts, size * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts);
    cudaFreeHost(pinned_elts);
    return cudaStatus;
}

template <typename T>
cudaError_t cuda_matrix_compare_equality(const T* elts1, const T* elts2, unsigned int size, bool* equality) {
    T* dev_elts1 = nullptr;
    T* dev_elts2 = nullptr;
    bool* dev_equality = nullptr;
    cudaError_t cudaStatus;

    // Allocate pinned memory
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    bool* pinned_equality = nullptr;
    cudaStatus = cudaMallocHost((void**)&pinned_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMallocHost((void**)&pinned_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMallocHost((void**)&pinned_equality, sizeof(bool));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    CREATE_CUDA_STREAM(stream);

    *pinned_equality = true;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_equality, sizeof(bool));
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));
    memcpy(pinned_equality, equality, sizeof(bool));

    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_equality, pinned_equality, sizeof(bool), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    compareEqKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts1, dev_elts2, dev_equality, size);

    cudaStatus = cudaMemcpyAsync(pinned_equality, dev_equality, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(equality, pinned_equality, sizeof(bool));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_equality);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    cudaFreeHost(pinned_equality);
    return cudaStatus;
}


template <typename T>
cudaError_t cuda_matrix_dot(const T* elts1, const T* elts2, unsigned int size, T* result) {
    T* dev_elts1 = nullptr;
    T* dev_elts2 = nullptr;
    T* dev_result = nullptr;
    cudaError_t cudaStatus;

    // Allocate pinned memory
    T* pinned_elts1 = nullptr;
    T* pinned_elts2 = nullptr;
    T* pinned_result = nullptr;
    cudaStatus = cudaMallocHost((void**)&pinned_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMallocHost((void**)&pinned_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaMallocHost((void**)&pinned_result, sizeof(T));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    CREATE_CUDA_STREAM(stream);

    *pinned_result = 0;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts1, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts2, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_result, sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(pinned_elts1, elts1, size * sizeof(T));
    memcpy(pinned_elts2, elts2, size * sizeof(T));

    cudaStatus = cudaMemcpyAsync(dev_elts1, pinned_elts1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_elts2, pinned_elts2, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_result, pinned_result, sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    dotKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts1, dev_elts2, dev_result, size);

    cudaStatus = cudaMemcpyAsync(pinned_result, dev_result, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(result, pinned_result, sizeof(T));
Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts1);
    cudaFree(dev_elts2);
    cudaFree(dev_result);
    cudaFreeHost(pinned_elts1);
    cudaFreeHost(pinned_elts2);
    cudaFreeHost(pinned_result);
    return cudaStatus;
}

template <typename T>
cudaError_t cuda_matrix_clear(T* elts, unsigned int size) {
    T* dev_elts = nullptr;
    T* pinned_elts = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaMallocHost((void**)&pinned_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_elts, size * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    memset(pinned_elts, 0, size * sizeof(T)); // Initialize pinned memory to zero

    // Copy to device
    cudaStatus = cudaMemcpyAsync(dev_elts, pinned_elts, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    clearKernel<<<gridSize, blockSize, 0, stream>>>(dev_elts, size);

    // Copy the result back to host
    cudaStatus = cudaMemcpyAsync(pinned_elts, dev_elts, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(elts, pinned_elts, size * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_elts);
    cudaFreeHost(pinned_elts);
    return cudaStatus;
}

template <typename T>
cudaError_t cuda_matrix_multiply(const T* A, const T* B, T* C, int rowsA, int colsA, int colsB) {
    T* dev_A = nullptr;
    T* dev_B = nullptr;
    T* dev_C = nullptr;
    T *pinned_A = nullptr, *pinned_B = nullptr, *pinned_C = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaMallocHost((void**)&pinned_A, rowsA * colsA * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMallocHost((void**)&pinned_B, colsA * colsB * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMallocHost((void**)&pinned_C, rowsA * colsB * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_A, rowsA * colsA * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_B, colsA * colsB * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_C, rowsA * colsB * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(pinned_A, A, rowsA * colsA * sizeof(T));
    memcpy(pinned_B, B, colsA * colsB * sizeof(T));

    // Copy to device
    cudaStatus = cudaMemcpyAsync(dev_A, pinned_A, rowsA * colsA * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpyAsync(dev_B, pinned_B, colsA * colsB * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x,
                  (rowsA + blockSize.y - 1) / blockSize.y);
    matrixMultiplyKernel<<<gridSize, blockSize, 0, stream>>>(dev_A, dev_B, dev_C, rowsA, colsA, colsB);

    // Copy the result back to host
    cudaStatus = cudaMemcpyAsync(pinned_C, dev_C, rowsA * colsB * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    memcpy(C, pinned_C, rowsA * colsB * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(pinned_A);
    cudaFreeHost(pinned_B);
    cudaFreeHost(pinned_C);
    return cudaStatus;
}


template <typename T>
cudaError_t cuda_matrix_transpose(const T* input, T* output, int rows, int cols) {
    T* dev_input = nullptr;
    T* dev_output = nullptr;
    T *pinned_input = nullptr, *pinned_output = nullptr;
    cudaError_t cudaStatus;
    CREATE_CUDA_STREAM(stream);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate pinned memory
    cudaStatus = cudaMallocHost((void**)&pinned_input, rows * cols * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMallocHost((void**)&pinned_output, rows * cols * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Allocate device memory
    cudaStatus = cudaMalloc((void**)&dev_input, rows * cols * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_output, rows * cols * sizeof(T));
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy the input matrix to pinned memory
    memcpy(pinned_input, input, rows * cols * sizeof(T));

    // Copy input matrix from pinned memory to device
    cudaStatus = cudaMemcpyAsync(dev_input, pinned_input, rows * cols * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Define block and grid sizes for kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch transpose kernel
    transposeKernel<<<gridSize, blockSize, 0, stream>>>(dev_input, dev_output, rows, cols);

    // Copy the transposed matrix back from device to pinned memory
    cudaStatus = cudaMemcpyAsync(pinned_output, dev_output, rows * cols * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto Error;

    // Copy the result back to the host
    memcpy(output, pinned_output, rows * cols * sizeof(T));

Error:
    DESTROY_CUDA_STREAM(stream);
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    return cudaStatus;
}