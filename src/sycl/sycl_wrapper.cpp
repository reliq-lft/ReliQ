/*
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/sycl/sycl_wrapper.cpp
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
  Native SYCL Backend with Multi-Type Support
  
  This implementation uses idiomatic SYCL with C++ templates for type-generic kernels.
  Supports: float32 (float), float64 (double), int32 (int), int64 (long long)
  
  Tensor Layout: AoSoA (Array of Structures of Arrays)
  - Sites are grouped into vectors of VectorWidth elements
  - Within each vector group, tensor elements are interleaved
  - Layout: [group0_elem0_lane0..7, group0_elem1_lane0..7, ..., group1_elem0_lane0..7, ...]
*/

#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdint>

// Device type constants matching Nim enum
constexpr int SYCL_DEVICE_DEFAULT = 0;
constexpr int SYCL_DEVICE_CPU = 1;
constexpr int SYCL_DEVICE_GPU = 2;
constexpr int SYCL_DEVICE_ACCELERATOR = 3;
constexpr int SYCL_DEVICE_ALL = 4;

// Global state
static std::mutex g_mutex;
static std::vector<sycl::device> g_devices_cpu;
static std::vector<sycl::device> g_devices_gpu;
static std::vector<sycl::device> g_devices_acc;
static bool g_initialized = false;

// Wrapper types
struct SyclQueueWrapper {
    sycl::queue queue;
    sycl::context context;
    sycl::device device;
};

struct SyclBufferWrapper {
    void* ptr;
    size_t size;
    sycl::queue* queue;
};

static void init_devices() {
    if (g_initialized) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initialized) return;
    
    for (auto& platform : sycl::platform::get_platforms()) {
        for (auto& device : platform.get_devices()) {
            if (device.is_cpu()) {
                g_devices_cpu.push_back(device);
            } else if (device.is_gpu()) {
                g_devices_gpu.push_back(device);
            } else if (device.is_accelerator()) {
                g_devices_acc.push_back(device);
            }
        }
    }
    g_initialized = true;
}

static std::vector<sycl::device>& get_device_list(int dtype) {
    init_devices();
    switch (dtype) {
        case SYCL_DEVICE_CPU: return g_devices_cpu;
        case SYCL_DEVICE_GPU: return g_devices_gpu;
        case SYCL_DEVICE_ACCELERATOR: return g_devices_acc;
        default:
            // Default: prefer GPU, then CPU
            if (!g_devices_gpu.empty()) return g_devices_gpu;
            return g_devices_cpu;
    }
}

// ============================================================================
// Template Kernel Implementations
// ============================================================================

template<typename T>
void kernel_copy_impl(sycl::queue& q, T* a, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = a[idx];
    });
}

template<typename T>
void kernel_add_impl(sycl::queue& q, T* a, T* b, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = a[idx] + b[idx];
    });
}

template<typename T>
void kernel_sub_impl(sycl::queue& q, T* a, T* b, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = a[idx] - b[idx];
    });
}

template<typename T>
void kernel_mul_impl(sycl::queue& q, T* a, T* b, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = a[idx] * b[idx];
    });
}

template<typename T>
void kernel_scalar_mul_impl(sycl::queue& q, T* a, T scalar, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = scalar * a[idx];
    });
}

template<typename T>
void kernel_scalar_add_impl(sycl::queue& q, T* a, T scalar, T* c, size_t numElements) {
    q.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
        c[idx] = scalar + a[idx];
    });
}

template<typename T>
void kernel_matmul_impl(sycl::queue& q, T* a, T* b, T* c,
                        size_t numSites, int rows, int cols, int inner,
                        int vectorWidth) {
    int elemsA = rows * inner;
    int elemsB = inner * cols;
    int elemsC = rows * cols;
    int strideA = elemsA * vectorWidth;
    int strideB = elemsB * vectorWidth;
    int strideC = elemsC * vectorWidth;
    
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        
        size_t baseA = group * strideA;
        size_t baseB = group * strideB;
        size_t baseC = group * strideC;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T sum = T(0);
                for (int k = 0; k < inner; k++) {
                    size_t idxA = baseA + (i * inner + k) * vectorWidth + lane;
                    size_t idxB = baseB + (k * cols + j) * vectorWidth + lane;
                    sum += a[idxA] * b[idxB];
                }
                size_t idxC = baseC + (i * cols + j) * vectorWidth + lane;
                c[idxC] = sum;
            }
        }
    });
}

template<typename T>
void kernel_matvec_impl(sycl::queue& q, T* mat, T* x, T* y,
                        size_t numSites, int rows, int cols, int vectorWidth) {
    int strideMat = rows * cols * vectorWidth;
    int strideVec = cols * vectorWidth;
    int strideOut = rows * vectorWidth;
    
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        
        size_t baseMat = group * strideMat;
        size_t baseVec = group * strideVec;
        size_t baseOut = group * strideOut;
        
        for (int i = 0; i < rows; i++) {
            T sum = T(0);
            for (int j = 0; j < cols; j++) {
                size_t idxMat = baseMat + (i * cols + j) * vectorWidth + lane;
                size_t idxVec = baseVec + j * vectorWidth + lane;
                sum += mat[idxMat] * x[idxVec];
            }
            size_t idxOut = baseOut + i * vectorWidth + lane;
            y[idxOut] = sum;
        }
    });
}

template<typename T>
void kernel_set_element_impl(sycl::queue& q, T* c, int elementIdx, T value,
                             size_t numSites, int elemsPerSite, int vectorWidth) {
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        size_t aosoaIdx = group * (elemsPerSite * vectorWidth) + elementIdx * vectorWidth + lane;
        c[aosoaIdx] = value;
    });
}

template<typename T>
void kernel_set_elements_impl(sycl::queue& q, T* c,
                              const int* elementIndices, const T* values, int numWrites,
                              size_t numSites, int elemsPerSite, int vectorWidth) {
    // Copy indices and values to device
    int* d_indices = sycl::malloc_device<int>(numWrites, q);
    T* d_values = sycl::malloc_device<T>(numWrites, q);
    q.memcpy(d_indices, elementIndices, numWrites * sizeof(int)).wait();
    q.memcpy(d_values, values, numWrites * sizeof(T)).wait();
    
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        size_t base = group * (elemsPerSite * vectorWidth);
        
        for (int w = 0; w < numWrites; w++) {
            int elemIdx = d_indices[w];
            T val = d_values[w];
            size_t aosoaIdx = base + elemIdx * vectorWidth + lane;
            c[aosoaIdx] = val;
        }
    }).wait();
    
    sycl::free(d_indices, q);
    sycl::free(d_values, q);
}

// Complex kernels (for floating point types only)
template<typename T>
void kernel_complex_add_impl(sycl::queue& q, T* a, T* b, T* c, size_t numComplexElements) {
    size_t numReals = numComplexElements * 2;
    q.parallel_for(sycl::range<1>(numReals), [=](sycl::id<1> idx) {
        c[idx] = a[idx] + b[idx];
    });
}

template<typename T>
void kernel_complex_scalar_mul_impl(sycl::queue& q, T* a, T scalar_re, T scalar_im, 
                                    T* c, size_t numComplexElements) {
    q.parallel_for(sycl::range<1>(numComplexElements), [=](sycl::id<1> idx) {
        size_t i = idx * 2;
        T a_re = a[i];
        T a_im = a[i + 1];
        c[i]     = scalar_re * a_re - scalar_im * a_im;
        c[i + 1] = scalar_re * a_im + scalar_im * a_re;
    });
}

template<typename T>
void kernel_complex_matmul_impl(sycl::queue& q, T* a, T* b, T* c,
                                size_t numSites, int rows, int cols, int inner,
                                int vectorWidth) {
    int elemsA = rows * inner * 2;
    int elemsB = inner * cols * 2;
    int elemsC = rows * cols * 2;
    int strideA = elemsA * vectorWidth;
    int strideB = elemsB * vectorWidth;
    int strideC = elemsC * vectorWidth;
    
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        
        size_t baseA = group * strideA;
        size_t baseB = group * strideB;
        size_t baseC = group * strideC;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T sum_re = T(0);
                T sum_im = T(0);
                for (int k = 0; k < inner; k++) {
                    size_t idxA_re = baseA + ((i * inner + k) * 2) * vectorWidth + lane;
                    size_t idxA_im = baseA + ((i * inner + k) * 2 + 1) * vectorWidth + lane;
                    size_t idxB_re = baseB + ((k * cols + j) * 2) * vectorWidth + lane;
                    size_t idxB_im = baseB + ((k * cols + j) * 2 + 1) * vectorWidth + lane;
                    
                    T a_re = a[idxA_re], a_im = a[idxA_im];
                    T b_re = b[idxB_re], b_im = b[idxB_im];
                    
                    sum_re += a_re * b_re - a_im * b_im;
                    sum_im += a_re * b_im + a_im * b_re;
                }
                size_t idxC_re = baseC + ((i * cols + j) * 2) * vectorWidth + lane;
                size_t idxC_im = baseC + ((i * cols + j) * 2 + 1) * vectorWidth + lane;
                c[idxC_re] = sum_re;
                c[idxC_im] = sum_im;
            }
        }
    });
}

template<typename T>
void kernel_complex_matvec_impl(sycl::queue& q, T* mat, T* x, T* y,
                                size_t numSites, int rows, int cols, int vectorWidth) {
    int strideMat = rows * cols * 2 * vectorWidth;
    int strideVec = cols * 2 * vectorWidth;
    int strideOut = rows * 2 * vectorWidth;
    
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        size_t group = site / vectorWidth;
        size_t lane = site % vectorWidth;
        
        size_t baseMat = group * strideMat;
        size_t baseVec = group * strideVec;
        size_t baseOut = group * strideOut;
        
        for (int i = 0; i < rows; i++) {
            T sum_re = T(0);
            T sum_im = T(0);
            for (int j = 0; j < cols; j++) {
                size_t idxMat_re = baseMat + ((i * cols + j) * 2) * vectorWidth + lane;
                size_t idxMat_im = baseMat + ((i * cols + j) * 2 + 1) * vectorWidth + lane;
                size_t idxVec_re = baseVec + (j * 2) * vectorWidth + lane;
                size_t idxVec_im = baseVec + (j * 2 + 1) * vectorWidth + lane;
                
                T m_re = mat[idxMat_re], m_im = mat[idxMat_im];
                T v_re = x[idxVec_re], v_im = x[idxVec_im];
                
                sum_re += m_re * v_re - m_im * v_im;
                sum_im += m_re * v_im + m_im * v_re;
            }
            size_t idxOut_re = baseOut + (i * 2) * vectorWidth + lane;
            size_t idxOut_im = baseOut + (i * 2 + 1) * vectorWidth + lane;
            y[idxOut_re] = sum_re;
            y[idxOut_im] = sum_im;
        }
    });
}

// ============================================================================
// Stencil Gather Kernel - copies from neighbor sites using offset table
// ============================================================================

template<typename T>
void kernel_stencil_copy_impl(sycl::queue& q, T* src, T* dst,
                              const int32_t* offsets, int pointIdx, int nPoints,
                              size_t numSites, int elemsPerSite, int vectorWidth) {
    // offsets layout: offsets[site * nPoints + pointIdx] = neighbor site index
    // AoSoA layout: group = site / VW, lane = site % VW
    // element i at site s is at: (s/VW) * (elemsPerSite * VW) + i * VW + (s % VW)
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        int nbrSite = offsets[site * nPoints + pointIdx];
        
        size_t dstGroup = site / vectorWidth;
        size_t dstLane  = site % vectorWidth;
        size_t nbrGroup = nbrSite / vectorWidth;
        size_t nbrLane  = nbrSite % vectorWidth;
        
        size_t dstBase = dstGroup * (elemsPerSite * vectorWidth);
        size_t srcBase = nbrGroup * (elemsPerSite * vectorWidth);
        
        for (int e = 0; e < elemsPerSite; e++) {
            dst[dstBase + e * vectorWidth + dstLane] = 
                src[srcBase + e * vectorWidth + nbrLane];
        }
    });
}

// Stencil scalar multiply: dst[n] = scalar * src[neighbor(n)]
template<typename T>
void kernel_stencil_scalar_mul_impl(sycl::queue& q, T* src, T scalar, T* dst,
                                    const int32_t* offsets, int pointIdx, int nPoints,
                                    size_t numSites, int elemsPerSite, int vectorWidth) {
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        int nbrSite = offsets[site * nPoints + pointIdx];
        
        size_t dstGroup = site / vectorWidth;
        size_t dstLane  = site % vectorWidth;
        size_t nbrGroup = nbrSite / vectorWidth;
        size_t nbrLane  = nbrSite % vectorWidth;
        
        size_t dstBase = dstGroup * (elemsPerSite * vectorWidth);
        size_t srcBase = nbrGroup * (elemsPerSite * vectorWidth);
        
        for (int e = 0; e < elemsPerSite; e++) {
            dst[dstBase + e * vectorWidth + dstLane] = 
                scalar * src[srcBase + e * vectorWidth + nbrLane];
        }
    });
}

// Stencil add: dst[n] = srcA[n] + srcB[neighbor(n)]
template<typename T>
void kernel_stencil_add_impl(sycl::queue& q, T* srcA, T* srcB, T* dst,
                             const int32_t* offsets, int pointIdx, int nPoints,
                             size_t numSites, int elemsPerSite, int vectorWidth) {
    q.parallel_for(sycl::range<1>(numSites), [=](sycl::id<1> idx) {
        size_t site = idx;
        int nbrSite = offsets[site * nPoints + pointIdx];
        
        size_t dstGroup = site / vectorWidth;
        size_t dstLane  = site % vectorWidth;
        size_t nbrGroup = nbrSite / vectorWidth;
        size_t nbrLane  = nbrSite % vectorWidth;
        
        size_t dstBase = dstGroup * (elemsPerSite * vectorWidth);
        size_t srcABase = dstGroup * (elemsPerSite * vectorWidth);  // srcA uses same site
        size_t srcBBase = nbrGroup * (elemsPerSite * vectorWidth);
        
        for (int e = 0; e < elemsPerSite; e++) {
            dst[dstBase + e * vectorWidth + dstLane] = 
                srcA[srcABase + e * vectorWidth + dstLane] +
                srcB[srcBBase + e * vectorWidth + nbrLane];
        }
    });
}

// ============================================================================
// Extern "C" Entry Points
// ============================================================================

extern "C" {

// ============================================================================
// Device and Queue Management
// ============================================================================

int sycl_get_device_count(int dtype) {
    return static_cast<int>(get_device_list(dtype).size());
}

void* sycl_create_queue(int dtype, int deviceIdx) {
    try {
        auto& devices = get_device_list(dtype);
        if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) {
            return nullptr;
        }
        
        auto wrapper = new SyclQueueWrapper();
        wrapper->device = devices[deviceIdx];
        wrapper->context = sycl::context(wrapper->device);
        wrapper->queue = sycl::queue(wrapper->context, wrapper->device, 
            sycl::property::queue::in_order{});
        return wrapper;
    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL queue creation failed: %s\n", e.what());
        return nullptr;
    }
}

void sycl_destroy_queue(void* queue) {
    if (queue) {
        auto wrapper = static_cast<SyclQueueWrapper*>(queue);
        wrapper->queue.wait();
        delete wrapper;
    }
}

const char* sycl_get_device_name(int dtype, int deviceIdx) {
    static thread_local std::string name;
    try {
        auto& devices = get_device_list(dtype);
        if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) {
            name = "Unknown device";
            return name.c_str();
        }
        name = devices[deviceIdx].get_info<sycl::info::device::name>();
        return name.c_str();
    } catch (...) {
        name = "Error getting device name";
        return name.c_str();
    }
}

int sycl_device_is_cpu(int dtype, int deviceIdx) {
    auto& devices = get_device_list(dtype);
    if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) return 0;
    return devices[deviceIdx].is_cpu() ? 1 : 0;
}

int sycl_device_is_gpu(int dtype, int deviceIdx) {
    auto& devices = get_device_list(dtype);
    if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) return 0;
    return devices[deviceIdx].is_gpu() ? 1 : 0;
}

// ============================================================================
// Memory Management
// ============================================================================

void* sycl_allocate(void* queue, size_t size) {
    if (!queue || size == 0) return nullptr;
    try {
        auto q = static_cast<SyclQueueWrapper*>(queue);
        auto buf = new SyclBufferWrapper();
        buf->ptr = sycl::malloc_device(size, q->queue);
        buf->size = size;
        buf->queue = &q->queue;
        return buf;
    } catch (...) {
        return nullptr;
    }
}

void sycl_deallocate(void* queue, void* buf) {
    if (queue && buf) {
        auto q = static_cast<SyclQueueWrapper*>(queue);
        auto b = static_cast<SyclBufferWrapper*>(buf);
        sycl::free(b->ptr, q->queue);
        delete b;
    }
}

void sycl_write(void* queue, void* buf, const void* src, size_t size) {
    if (!queue || !buf || !src || size == 0) return;
    auto q = static_cast<SyclQueueWrapper*>(queue);
    auto b = static_cast<SyclBufferWrapper*>(buf);
    q->queue.memcpy(b->ptr, src, size).wait();
}

void sycl_read(void* queue, void* dest, void* buf, size_t size) {
    if (!queue || !buf || !dest || size == 0) return;
    auto q = static_cast<SyclQueueWrapper*>(queue);
    auto b = static_cast<SyclBufferWrapper*>(buf);
    q->queue.memcpy(dest, b->ptr, size).wait();
}

void sycl_wait(void* queue) {
    if (queue) {
        auto q = static_cast<SyclQueueWrapper*>(queue);
        q->queue.wait();
    }
}

void* sycl_get_buffer_ptr(void* buffer) {
    if (!buffer) return nullptr;
    auto b = static_cast<SyclBufferWrapper*>(buffer);
    return b->ptr;
}

// ============================================================================
// Macro-Generated Type-Specific Kernels
// ============================================================================

#define DEFINE_BASIC_KERNELS(SUFFIX, TYPE) \
void sycl_kernel_copy_##SUFFIX(void* queue, void* bufA, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_copy_impl(q->queue, a, c, numElements); \
} \
\
void sycl_kernel_add_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufB || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_add_impl(q->queue, a, b, c, numElements); \
} \
\
void sycl_kernel_sub_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufB || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_sub_impl(q->queue, a, b, c, numElements); \
} \
\
void sycl_kernel_mul_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufB || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_mul_impl(q->queue, a, b, c, numElements); \
} \
\
void sycl_kernel_scalar_mul_##SUFFIX(void* queue, void* bufA, TYPE scalar, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_scalar_mul_impl(q->queue, a, scalar, c, numElements); \
} \
\
void sycl_kernel_scalar_add_##SUFFIX(void* queue, void* bufA, TYPE scalar, void* bufC, size_t numElements) { \
    if (!queue || !bufA || !bufC || numElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_scalar_add_impl(q->queue, a, scalar, c, numElements); \
}

#define DEFINE_MATRIX_KERNELS(SUFFIX, TYPE) \
void sycl_kernel_matmul_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                  size_t numSites, int rows, int cols, int inner, \
                                  int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufA || !bufB || !bufC || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_matmul_impl(q->queue, a, b, c, numSites, rows, cols, inner, vectorWidth); \
} \
\
void sycl_kernel_matvec_##SUFFIX(void* queue, void* bufA, void* bufX, void* bufY, \
                                  size_t numSites, int rows, int cols, \
                                  int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufA || !bufX || !bufY || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* mat = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* x = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufX)->ptr); \
    auto* y = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufY)->ptr); \
    kernel_matvec_impl(q->queue, mat, x, y, numSites, rows, cols, vectorWidth); \
} \
\
void sycl_kernel_matadd_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                  size_t numSites, int rows, int cols, \
                                  int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * rows * cols * vectorWidth; \
    sycl_kernel_add_##SUFFIX(queue, bufA, bufB, bufC, numElements); \
} \
\
void sycl_kernel_vecadd_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                  size_t numSites, int vecLen, \
                                  int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * vecLen * vectorWidth; \
    sycl_kernel_add_##SUFFIX(queue, bufA, bufB, bufC, numElements); \
} \
\
void sycl_kernel_tensor_scalar_mul_##SUFFIX(void* queue, void* bufA, TYPE scalar, void* bufC, \
                                             size_t numSites, int elemsPerSite, \
                                             int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * elemsPerSite * vectorWidth; \
    sycl_kernel_scalar_mul_##SUFFIX(queue, bufA, scalar, bufC, numElements); \
} \
\
void sycl_kernel_tensor_scalar_add_##SUFFIX(void* queue, void* bufA, TYPE scalar, void* bufC, \
                                             size_t numSites, int elemsPerSite, \
                                             int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * elemsPerSite * vectorWidth; \
    sycl_kernel_scalar_add_##SUFFIX(queue, bufA, scalar, bufC, numElements); \
}

#define DEFINE_ELEMENT_WRITE_KERNELS(SUFFIX, TYPE) \
void sycl_kernel_set_element_##SUFFIX(void* queue, void* bufC, \
                                       int elementIdx, TYPE value, \
                                       size_t numSites, int elemsPerSite, \
                                       int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufC || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_set_element_impl(q->queue, c, elementIdx, value, numSites, elemsPerSite, vectorWidth); \
} \
\
void sycl_kernel_set_elements_##SUFFIX(void* queue, void* bufC, \
                                        const int* elementIndices, const TYPE* values, int numWrites, \
                                        size_t numSites, int elemsPerSite, \
                                        int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufC || numSites == 0 || numWrites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_set_elements_impl(q->queue, c, elementIndices, values, numWrites, numSites, elemsPerSite, vectorWidth); \
}

#define DEFINE_COMPLEX_KERNELS(SUFFIX, TYPE) \
void sycl_kernel_complex_add_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, size_t numComplexElements) { \
    if (!queue || !bufA || !bufB || !bufC || numComplexElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_complex_add_impl(q->queue, a, b, c, numComplexElements); \
} \
\
void sycl_kernel_complex_scalar_mul_##SUFFIX(void* queue, void* bufA, TYPE scalar_re, TYPE scalar_im, \
                                              void* bufC, size_t numComplexElements) { \
    if (!queue || !bufA || !bufC || numComplexElements == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_complex_scalar_mul_impl(q->queue, a, scalar_re, scalar_im, c, numComplexElements); \
} \
\
void sycl_kernel_complex_matmul_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                          size_t numSites, int rows, int cols, int inner, \
                                          int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufA || !bufB || !bufC || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* a = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* b = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufB)->ptr); \
    auto* c = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufC)->ptr); \
    kernel_complex_matmul_impl(q->queue, a, b, c, numSites, rows, cols, inner, vectorWidth); \
} \
\
void sycl_kernel_complex_matvec_##SUFFIX(void* queue, void* bufA, void* bufX, void* bufY, \
                                          size_t numSites, int rows, int cols, \
                                          int vectorWidth, size_t numVectorGroups) { \
    if (!queue || !bufA || !bufX || !bufY || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* mat = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufA)->ptr); \
    auto* x = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufX)->ptr); \
    auto* y = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufY)->ptr); \
    kernel_complex_matvec_impl(q->queue, mat, x, y, numSites, rows, cols, vectorWidth); \
} \
\
void sycl_kernel_complex_matadd_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                          size_t numSites, int rows, int cols, \
                                          int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * rows * cols * 2 * vectorWidth; \
    sycl_kernel_add_##SUFFIX(queue, bufA, bufB, bufC, numElements); \
} \
\
void sycl_kernel_complex_vecadd_##SUFFIX(void* queue, void* bufA, void* bufB, void* bufC, \
                                          size_t numSites, int vecLen, \
                                          int vectorWidth, size_t numVectorGroups) { \
    size_t numElements = numVectorGroups * vecLen * 2 * vectorWidth; \
    sycl_kernel_add_##SUFFIX(queue, bufA, bufB, bufC, numElements); \
} \
\
void sycl_kernel_complex_tensor_scalar_mul_##SUFFIX(void* queue, void* bufA, \
                                                     TYPE scalar_re, TYPE scalar_im, \
                                                     void* bufC, size_t numSites, int elemsPerSite, \
                                                     int vectorWidth, size_t numVectorGroups) { \
    size_t numComplexElements = numVectorGroups * elemsPerSite * vectorWidth; \
    sycl_kernel_complex_scalar_mul_##SUFFIX(queue, bufA, scalar_re, scalar_im, bufC, numComplexElements); \
}

// Generate all kernel variants for each type
// float32 (f32)
DEFINE_BASIC_KERNELS(f32, float)
DEFINE_MATRIX_KERNELS(f32, float)
DEFINE_ELEMENT_WRITE_KERNELS(f32, float)
DEFINE_COMPLEX_KERNELS(f32, float)

// float64 (f64)
DEFINE_BASIC_KERNELS(f64, double)
DEFINE_MATRIX_KERNELS(f64, double)
DEFINE_ELEMENT_WRITE_KERNELS(f64, double)
DEFINE_COMPLEX_KERNELS(f64, double)

// int32 (i32)
DEFINE_BASIC_KERNELS(i32, int32_t)
DEFINE_MATRIX_KERNELS(i32, int32_t)
DEFINE_ELEMENT_WRITE_KERNELS(i32, int32_t)
// No complex kernels for integers

// int64 (i64)
DEFINE_BASIC_KERNELS(i64, int64_t)
DEFINE_MATRIX_KERNELS(i64, int64_t)
DEFINE_ELEMENT_WRITE_KERNELS(i64, int64_t)
// No complex kernels for integers

// ============================================================================
// Stencil Gather Kernels
// ============================================================================

#define DEFINE_STENCIL_KERNELS(SUFFIX, TYPE) \
void sycl_kernel_stencil_copy_##SUFFIX(void* queue, void* bufSrc, void* bufDst, \
                                        void* bufOffsets, int pointIdx, int nPoints, \
                                        size_t numSites, int elemsPerSite, int vectorWidth) { \
    if (!queue || !bufSrc || !bufDst || !bufOffsets || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* src = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufSrc)->ptr); \
    auto* dst = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufDst)->ptr); \
    auto* offsets = static_cast<int32_t*>(static_cast<SyclBufferWrapper*>(bufOffsets)->ptr); \
    kernel_stencil_copy_impl(q->queue, src, dst, offsets, pointIdx, nPoints, \
                             numSites, elemsPerSite, vectorWidth); \
} \
\
void sycl_kernel_stencil_scalar_mul_##SUFFIX(void* queue, void* bufSrc, TYPE scalar, void* bufDst, \
                                              void* bufOffsets, int pointIdx, int nPoints, \
                                              size_t numSites, int elemsPerSite, int vectorWidth) { \
    if (!queue || !bufSrc || !bufDst || !bufOffsets || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* src = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufSrc)->ptr); \
    auto* dst = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufDst)->ptr); \
    auto* offsets = static_cast<int32_t*>(static_cast<SyclBufferWrapper*>(bufOffsets)->ptr); \
    kernel_stencil_scalar_mul_impl(q->queue, src, scalar, dst, offsets, pointIdx, nPoints, \
                                   numSites, elemsPerSite, vectorWidth); \
} \
\
void sycl_kernel_stencil_add_##SUFFIX(void* queue, void* bufSrcA, void* bufSrcB, void* bufDst, \
                                       void* bufOffsets, int pointIdx, int nPoints, \
                                       size_t numSites, int elemsPerSite, int vectorWidth) { \
    if (!queue || !bufSrcA || !bufSrcB || !bufDst || !bufOffsets || numSites == 0) return; \
    auto q = static_cast<SyclQueueWrapper*>(queue); \
    auto* srcA = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufSrcA)->ptr); \
    auto* srcB = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufSrcB)->ptr); \
    auto* dst = static_cast<TYPE*>(static_cast<SyclBufferWrapper*>(bufDst)->ptr); \
    auto* offsets = static_cast<int32_t*>(static_cast<SyclBufferWrapper*>(bufOffsets)->ptr); \
    kernel_stencil_add_impl(q->queue, srcA, srcB, dst, offsets, pointIdx, nPoints, \
                            numSites, elemsPerSite, vectorWidth); \
}

DEFINE_STENCIL_KERNELS(f32, float)
DEFINE_STENCIL_KERNELS(f64, double)
DEFINE_STENCIL_KERNELS(i32, int32_t)
DEFINE_STENCIL_KERNELS(i64, int64_t)

} // extern "C"
