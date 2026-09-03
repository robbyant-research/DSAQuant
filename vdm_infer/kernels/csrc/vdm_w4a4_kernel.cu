#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vdm_w4a4 {

static constexpr int BLOCK_M = 256;
static constexpr int BLOCK_N = 128;
static constexpr int WARP_SIZE = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int INSN_M = 16;
static constexpr int INSN_N = 16;
static constexpr int INSN_K = 64;
static constexpr int WARP_M = BLOCK_M / NUM_WARPS; // 32
static constexpr int WARP_N = BLOCK_N;
static constexpr int WARP_K = INSN_K;
static constexpr int WARP_M_TILES = WARP_M / INSN_M; // 2
static constexpr int WARP_N_TILES = WARP_N / INSN_N; // 8
static constexpr int PACK_SIZE_INT4 = INSN_K / 8;    // eight int4 values per 32-bit pack
static constexpr int NUM_PACKS_PER_ROW = INSN_K / PACK_SIZE_INT4; // 8
static constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE_INT4 * WARP_SIZE / INSN_K; // 4
static constexpr int NUM_PACKWARPS = INSN_M / NUM_ROWS_PER_PACKWARP; // 4
static constexpr int WSCALES_PACK_SIZE = 4;
static constexpr int WSCALES_NUM_PACKS = 1;
static constexpr int WSCALES_VALID_LANES = 32;
static constexpr int ASCALES_PACK_SIZE = 2;
static constexpr int ASCALES_NUM_PACKS = 1;
static constexpr int ASCALES_VALID_LANES = 16;

struct __align__(4) packed_ascale_t {
    half2 data[ASCALES_PACK_SIZE / 2];
};

struct __align__(8) packed_wscale_t {
    half2 data[WSCALES_PACK_SIZE / 2];
};

struct __align__(32) packed_psum_t {
    int data[8];
};

struct __align__(16) packed_fpsum_t {
    half2 data[4];
};

using act_warp_t = std::array<uint4, WARP_M_TILES>;
using wgt_warp_t = std::array<uint4, WARP_N_TILES>;
using ascale_warp_t = std::array<packed_ascale_t, ASCALES_NUM_PACKS>;
using wscale_warp_t = std::array<packed_wscale_t, WSCALES_NUM_PACKS>;
using fpsum_warp_t = std::array<packed_fpsum_t, WARP_M_TILES * WARP_N_TILES>;

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ uint32_t smem_u32_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4(const void *ptr, uint4 &out) {
    uint32_t addr = smem_u32_addr(ptr);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "r"(addr));
}

__device__ __forceinline__ half2 shfl_half2(half2 value, int src_lane) {
    uint32_t raw = *reinterpret_cast<uint32_t *>(&value);
    raw = __shfl_sync(FULL_MASK, raw, src_lane);
    return *reinterpret_cast<half2 *>(&raw);
}

template <typename T>
__device__ __forceinline__ T load_pred(const T *ptr, bool pred) {
    T value{};
    if (pred) {
        value = *ptr;
    }
    return value;
}

template <typename T>
__device__ __forceinline__ void store_pred(T *ptr, const T &value, bool pred) {
    if (pred) {
        *ptr = value;
    }
}

__device__ __forceinline__ uint32_t quantize_float2_s4(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.s4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

__device__ __forceinline__ half2 int2half2(int x, int y) {
    return __halves2half2(__int2half_rn(x), __int2half_rn(y));
}

__device__ __forceinline__ half2 scaled_int2_to_half2(int x, int y, float2 scale) {
    float vx = __int2float_rn(x) * scale.x;
    float vy = __int2float_rn(y) * scale.y;
    return __halves2half2(__float2half_rn(vx), __float2half_rn(vy));
}

__device__ __forceinline__ uint4 mma_m16n8k64_s4s4(uint4 a, uint2 b, uint4 c) {
    uint4 d{0, 0, 0, 0};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                 "{%0, %1, %2, %3},"
                 "{%4, %5, %6, %7},"
                 "{%8, %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
#else
    asm volatile("trap;\n");
#endif
    return d;
}

__device__ __forceinline__ packed_psum_t mma_int4(uint4 act, uint4 wgt) {
    packed_psum_t psum;
    uint4 out1 = mma_m16n8k64_s4s4(act, uint2{wgt.x, wgt.y}, uint4{0, 0, 0, 0});
    uint4 out2 = mma_m16n8k64_s4s4(act, uint2{wgt.z, wgt.w}, uint4{0, 0, 0, 0});
    psum.data[0] = static_cast<int>(out1.x);
    psum.data[1] = static_cast<int>(out1.y);
    psum.data[2] = static_cast<int>(out1.z);
    psum.data[3] = static_cast<int>(out1.w);
    psum.data[4] = static_cast<int>(out2.x);
    psum.data[5] = static_cast<int>(out2.y);
    psum.data[6] = static_cast<int>(out2.z);
    psum.data[7] = static_cast<int>(out2.w);
    return psum;
}

__device__ __forceinline__ half2 broadcast_wscale(const wscale_warp_t &block, int k, int lane_id) {
    int src_lane = 4 * (k / WSCALES_PACK_SIZE) + lane_id % 4;
    int element_idx = (k % WSCALES_PACK_SIZE) / 2;
    return shfl_half2(block[0].data[element_idx], src_lane);
}

__device__ __forceinline__ half2 broadcast_ascale(const ascale_warp_t &block, int k, int lane_id) {
    int src_lane = 8 * (k / ASCALES_PACK_SIZE) + lane_id / 4;
    int element_idx = (k % ASCALES_PACK_SIZE) / 2;
    return shfl_half2(block[0].data[element_idx], src_lane);
}

__device__ __forceinline__ void load_act_tile(const uint4 *act, int k_group, int K, act_warp_t &out, bool pred) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        out[i] = load_pred(&act[((k_group * NUM_WARPS + warp_id) * WARP_M_TILES + i) * WARP_SIZE + lane_id], pred);
    }
}

__device__ __forceinline__ void load_wgt_tile(const uint4 *wgt, int k_group, int K, wgt_warp_t &out, bool pred) {
    int lane_id = threadIdx.x % WARP_SIZE;
    const uint4 *ptr = &wgt[k_group * WARP_N_TILES * WARP_SIZE + lane_id];
#pragma unroll
    for (int i = 0; i < WARP_N_TILES; ++i) {
        out[i] = load_pred(&ptr[i * WARP_SIZE], pred);
    }
}

__device__ __forceinline__ void load_ascale_tile(const packed_ascale_t *ascales, int k_group, int M, ascale_warp_t &out, bool pred) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < ASCALES_NUM_PACKS; ++i) {
        out[i] = load_pred(&ascales[(k_group * NUM_WARPS + warp_id) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES + i * ASCALES_VALID_LANES + lane_id],
                           pred && lane_id < ASCALES_VALID_LANES);
    }
}

__device__ __forceinline__ void load_wscale_tile(const packed_wscale_t *wscales, int k_group, int N, wscale_warp_t &out, bool pred) {
    int lane_id = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int i = 0; i < WSCALES_NUM_PACKS; ++i) {
        out[i] = load_pred(&wscales[(k_group * WSCALES_NUM_PACKS + i) * WSCALES_VALID_LANES + lane_id],
                           pred && lane_id < WSCALES_VALID_LANES);
    }
}

__device__ __forceinline__ void apply_scales(const act_warp_t &A,
                                             const wgt_warp_t &W,
                                             const ascale_warp_t &ascale,
                                             const wscale_warp_t &wscale,
                                             fpsum_warp_t &fpsum) {
    int lane_id = threadIdx.x % WARP_SIZE;

    half2 asx[WARP_M_TILES];
    half2 asy[WARP_M_TILES];
#pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        half2 as = broadcast_ascale(ascale, i * 2, lane_id);
        half lo = __low2half(as);
        half hi = __high2half(as);
        asx[i] = __halves2half2(lo, lo);
        asy[i] = __halves2half2(hi, hi);
    }

#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j) {
        half2 ws1 = broadcast_wscale(wscale, j * 4, lane_id);
        half2 ws2 = broadcast_wscale(wscale, j * 4 + 2, lane_id);
#pragma unroll
        for (int i = 0; i < WARP_M_TILES; ++i) {
            packed_psum_t psum = mma_int4(A[i], W[j]);
            packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
            fsum.data[0] = __hfma2(int2half2(psum.data[0], psum.data[1]), __hmul2(asx[i], ws1), fsum.data[0]);
            fsum.data[1] = __hfma2(int2half2(psum.data[2], psum.data[3]), __hmul2(asy[i], ws1), fsum.data[1]);
            fsum.data[2] = __hfma2(int2half2(psum.data[4], psum.data[5]), __hmul2(asx[i], ws2), fsum.data[2]);
            fsum.data[3] = __hfma2(int2half2(psum.data[6], psum.data[7]), __hmul2(asy[i], ws2), fsum.data[3]);
        }
    }
}

__device__ __forceinline__ void accumulate_int4(const act_warp_t &A, const wgt_warp_t &W, std::array<packed_psum_t, WARP_M_TILES * WARP_N_TILES> &ipsum) {
#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j) {
#pragma unroll
        for (int i = 0; i < WARP_M_TILES; ++i) {
            packed_psum_t delta = mma_int4(A[i], W[j]);
            packed_psum_t &acc = ipsum[i * WARP_N_TILES + j];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc.data[k] += delta.data[k];
            }
        }
    }
}

__device__ __forceinline__ void finalize_int_accum(const std::array<packed_psum_t, WARP_M_TILES * WARP_N_TILES> &ipsum,
                                                   const ascale_warp_t &ascale,
                                                   const wscale_warp_t &wscale,
                                                   fpsum_warp_t &fpsum) {
    int lane_id = threadIdx.x % WARP_SIZE;

    float2 asx[WARP_M_TILES];
    float2 asy[WARP_M_TILES];
#pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        half2 as = broadcast_ascale(ascale, i * 2, lane_id);
        float lo = __half2float(__low2half(as));
        float hi = __half2float(__high2half(as));
        asx[i] = make_float2(lo, lo);
        asy[i] = make_float2(hi, hi);
    }

#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j) {
        float2 ws1 = __half22float2(broadcast_wscale(wscale, j * 4, lane_id));
        float2 ws2 = __half22float2(broadcast_wscale(wscale, j * 4 + 2, lane_id));
#pragma unroll
        for (int i = 0; i < WARP_M_TILES; ++i) {
            const packed_psum_t &psum = ipsum[i * WARP_N_TILES + j];
            packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
            fsum.data[0] = scaled_int2_to_half2(psum.data[0], psum.data[1], make_float2(asx[i].x * ws1.x, asx[i].y * ws1.y));
            fsum.data[1] = scaled_int2_to_half2(psum.data[2], psum.data[3], make_float2(asy[i].x * ws1.x, asy[i].y * ws1.y));
            fsum.data[2] = scaled_int2_to_half2(psum.data[4], psum.data[5], make_float2(asx[i].x * ws2.x, asx[i].y * ws2.y));
            fsum.data[3] = scaled_int2_to_half2(psum.data[6], psum.data[7], make_float2(asy[i].x * ws2.x, asy[i].y * ws2.y));
        }
    }
}

__device__ __forceinline__ void pack_ascales(const half *input, packed_ascale_t *output) {
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id < ASCALES_VALID_LANES) {
        packed_ascale_t tmp;
        tmp.data[0] = __halves2half2(input[lane_id / 8 * 8 * ASCALES_PACK_SIZE + lane_id % 8],
                                     input[lane_id / 8 * 8 * ASCALES_PACK_SIZE + lane_id % 8 + 8]);
        output[lane_id] = tmp;
    }
}

__device__ __forceinline__ void pack_wscales(const half *input, packed_wscale_t *output) {
    int lane_id = threadIdx.x % WARP_SIZE;
    packed_wscale_t tmp;
#pragma unroll
    for (int i = 0; i < WSCALES_PACK_SIZE; i += 2) {
        tmp.data[i / 2] = *reinterpret_cast<const half2 *>(&input[lane_id / 4 * 4 * WSCALES_PACK_SIZE + lane_id % 4 * 2 + i * 4]);
    }
    output[lane_id] = tmp;
}

__device__ __forceinline__ void quantize_warp_static_scale(const half *input,
                                                            int stride,
                                                            const half *row_scales,
                                                            uint4 &output,
                                                            half *output_scale,
                                                            void *shmem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    using packed_input_t = std::array<half, PACK_SIZE_INT4>;
    packed_input_t packs[NUM_PACKWARPS];

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        int col_id = lane_id % NUM_PACKS_PER_ROW * PACK_SIZE_INT4;
        packs[i] = *reinterpret_cast<const packed_input_t *>(input + row_id * stride + col_id);
    }

    using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
    matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        float scale_f = __half2float(row_scales[row_id]);
        float rscale = 1.0f / scale_f;
        if (lane_id % NUM_PACKS_PER_ROW == 0) {
            output_scale[row_id] = __float2half_rn(scale_f);
        }
        uint32_t qpack = 0;
#pragma unroll
        for (int j = 0; j < PACK_SIZE_INT4; j += 2) {
            float2 fval = make_float2(__half2float(packs[i][j]) * rscale, __half2float(packs[i][j + 1]) * rscale);
            qpack |= quantize_float2_s4(fval) << (j * 4);
        }
        mat[row_id][lane_id % NUM_PACKS_PER_ROW] = qpack;
    }
    __syncwarp();

    int row = lane_id % 16;
    int col = lane_id / 16 * 4;
    ldmatrix_x4(&mat[row][col], output);
    __syncwarp();
}


__device__ __forceinline__ void quantize_warp_static_scale_guarded(const half *input,
                                                                    int stride,
                                                                    const half *row_scales,
                                                                    int base_row,
                                                                    int M,
                                                                    uint4 &output,
                                                                    half *output_scale,
                                                                    void *shmem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    using packed_input_t = std::array<half, PACK_SIZE_INT4>;
    packed_input_t packs[NUM_PACKWARPS];

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        int col_id = lane_id % NUM_PACKS_PER_ROW * PACK_SIZE_INT4;
        if (base_row + row_id < M) {
            packs[i] = *reinterpret_cast<const packed_input_t *>(input + row_id * stride + col_id);
        } else {
#pragma unroll
            for (int j = 0; j < PACK_SIZE_INT4; ++j) {
                packs[i][j] = __float2half_rn(0.0f);
            }
        }
    }

    using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
    matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        bool valid = base_row + row_id < M;
        float scale_f = valid ? __half2float(row_scales[base_row + row_id]) : 1.0f;
        float rscale = 1.0f / scale_f;
        if (lane_id % NUM_PACKS_PER_ROW == 0) {
            output_scale[row_id] = __float2half_rn(scale_f);
        }
        uint32_t qpack = 0;
#pragma unroll
        for (int j = 0; j < PACK_SIZE_INT4; j += 2) {
            float2 fval = make_float2(__half2float(packs[i][j]) * rscale, __half2float(packs[i][j + 1]) * rscale);
            qpack |= quantize_float2_s4(fval) << (j * 4);
        }
        mat[row_id][lane_id % NUM_PACKS_PER_ROW] = qpack;
    }
    __syncwarp();

    int row = lane_id % 16;
    int col = lane_id / 16 * 4;
    ldmatrix_x4(&mat[row][col], output);
    __syncwarp();
}

__device__ __forceinline__ void quantize_warp_compute_scale(const half *input,
                                                             int stride,
                                                             uint4 &output,
                                                             half *output_scale,
                                                             void *shmem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    using packed_input_t = std::array<half, PACK_SIZE_INT4>;
    packed_input_t packs[NUM_PACKWARPS];

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        int col_id = lane_id % NUM_PACKS_PER_ROW * PACK_SIZE_INT4;
        packs[i] = *reinterpret_cast<const packed_input_t *>(input + row_id * stride + col_id);
    }

    float max_value[NUM_PACKWARPS];
#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        max_value[i] = fabsf(__half2float(packs[i][0]));
#pragma unroll
        for (int j = 1; j < PACK_SIZE_INT4; ++j) {
            max_value[i] = fmaxf(max_value[i], fabsf(__half2float(packs[i][j])));
        }
    }
#pragma unroll
    for (int mask = NUM_PACKS_PER_ROW / 2; mask > 0; mask /= 2) {
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; ++i) {
            max_value[i] = fmaxf(max_value[i], __shfl_xor_sync(FULL_MASK, max_value[i], mask));
        }
    }
#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        max_value[i] = __shfl_sync(FULL_MASK, max_value[i], lane_id / NUM_PACKS_PER_ROW * NUM_PACKS_PER_ROW);
    }

    using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
    matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

#pragma unroll
    for (int i = 0; i < NUM_PACKWARPS; ++i) {
        int row_id = i * NUM_ROWS_PER_PACKWARP + lane_id / NUM_PACKS_PER_ROW;
        float scale = max_value[i] > 0.0f ? max_value[i] / 7.0f : 1.0f;
        float rscale = 1.0f / scale;
        if (lane_id % NUM_PACKS_PER_ROW == 0) {
            output_scale[row_id] = __float2half_rn(scale);
        }
        uint32_t qpack = 0;
#pragma unroll
        for (int j = 0; j < PACK_SIZE_INT4; j += 2) {
            float2 fval = make_float2(__half2float(packs[i][j]) * rscale, __half2float(packs[i][j + 1]) * rscale);
            qpack |= quantize_float2_s4(fval) << (j * 4);
        }
        mat[row_id][lane_id % NUM_PACKS_PER_ROW] = qpack;
    }
    __syncwarp();

    int row = lane_id % 16;
    int col = lane_id / 16 * 4;
    ldmatrix_x4(&mat[row][col], output);
    __syncwarp();
}

__global__ void token_scale_kernel(const half *input, half *scales, int M, int K, float clip_ratio) {
    int row = blockIdx.x;
    float local_max = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(input[row * K + k])));
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float scale = smem[0] > 0.0f ? (smem[0] / 7.0f) * clip_ratio : 1.0f;
        scales[row] = __float2half_rn(scale);
    }
}

__global__ void act_quantize_per_token_kernel(const half *input,
                                               const half *token_scales,
                                               uint4 *output,
                                               packed_ascale_t *oscales,
                                               int M,
                                               int K) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int bm = blockIdx.x / (BLOCK_M / WARP_M);
    int bk = blockIdx.y;
    int warp_id = blockIdx.x % (BLOCK_M / WARP_M);
    int row = blockIdx.x * WARP_M;
    int col = blockIdx.y * WARP_K;

    __shared__ __align__(128) half oscale_shmem[WARP_M];
    __shared__ __align__(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

#pragma unroll
    for (int tile_id = 0; tile_id < WARP_M_TILES; ++tile_id) {
        uint4 tmpout;
        int tile_row = row + tile_id * INSN_M;
        quantize_warp_static_scale_guarded(input + tile_row * K + col,
                                           K,
                                           token_scales,
                                           tile_row,
                                           M,
                                           tmpout,
                                           oscale_shmem + tile_id * INSN_M,
                                           tmp_shmem);
        output[(((bm * K / WARP_K + bk) * NUM_WARPS + warp_id) * WARP_M_TILES + tile_id) * WARP_SIZE + lane_id] = tmpout;
    }

    pack_ascales(oscale_shmem,
                 &oscales[((bm * K / WARP_K + bk) * NUM_WARPS + warp_id) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
}

__global__ void weight_quantize_per_channel_kernel(const half *input,
                                                    const half *channel_scales,
                                                    uint4 *output,
                                                    packed_wscale_t *packed_scales,
                                                    int N,
                                                    int K) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int bn = blockIdx.x;
    int bk = blockIdx.y;
    int col = blockIdx.x * WARP_N;
    int row = blockIdx.y * WARP_K;

    __shared__ __align__(128) half oscale_shmem[WARP_N];
    __shared__ __align__(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

#pragma unroll
    for (int tile_id = 0; tile_id < WARP_N_TILES; ++tile_id) {
        uint4 tmpout;
        quantize_warp_static_scale(input + (col + tile_id * INSN_N) * K + row,
                                   K,
                                   channel_scales + col + tile_id * INSN_N,
                                   tmpout,
                                   oscale_shmem + tile_id * INSN_N,
                                   tmp_shmem);
        uint32_t swap_tmp = tmpout.y;
        tmpout.y = tmpout.z;
        tmpout.z = swap_tmp;
        output[((bn * K / WARP_K + bk) * WARP_N_TILES + tile_id) * WARP_SIZE + lane_id] = tmpout;
    }

    // The GEMM path uses nunchaku's per-K-tile packed scale layout. For per-channel
    // quantization we repeat the same channel scale for every K/64 tile.
    pack_wscales(oscale_shmem, &packed_scales[(bn * K / WARP_K + bk) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES]);
}

struct __align__(8) half4_pack {
    half data[4];
};

__device__ __forceinline__ void add_bias_to_pack(half4_pack &pack, const half *bias, int col) {
    if (bias == nullptr) {
        return;
    }
    half2 p0 = *reinterpret_cast<half2 *>(&pack.data[0]);
    half2 p1 = *reinterpret_cast<half2 *>(&pack.data[2]);
    half2 b0 = *reinterpret_cast<const half2 *>(&bias[col]);
    half2 b1 = *reinterpret_cast<const half2 *>(&bias[col + 2]);
    p0 = __hadd2(p0, b0);
    p1 = __hadd2(p1, b1);
    *reinterpret_cast<half2 *>(&pack.data[0]) = p0;
    *reinterpret_cast<half2 *>(&pack.data[2]) = p1;
}

__device__ __forceinline__ void store_fpsum(const fpsum_warp_t &fpsum, half *output, const half *bias, int M, int N, int bm, int bn) {
    using matrix_t = half[8][WARP_N + 8];
    __shared__ __align__(128) uint8_t shmem[NUM_WARPS][((sizeof(matrix_t) + 127) / 128) * 128];

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem[warp_id]);

#pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            const packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
            int row = lane_id / 4;
            int col = lane_id % 4 * 2 + j * INSN_N;
            *reinterpret_cast<half2 *>(&mat[row][col]) = fsum.data[0];
            *reinterpret_cast<half2 *>(&mat[row][col + 8]) = fsum.data[2];
        }
        __syncwarp();

#pragma unroll
        for (int row = 0; row < 8; ++row) {
            int global_m = bm * BLOCK_M + warp_id * WARP_M + i * INSN_M + row;
            int global_n = bn * BLOCK_N + lane_id * 4;
            if (global_m < M && global_n + 3 < N) {
                half4_pack pack = *reinterpret_cast<half4_pack *>(&mat[row][lane_id * 4]);
                add_bias_to_pack(pack, bias, global_n);
                *reinterpret_cast<half4_pack *>(&output[global_m * N + global_n]) = pack;
            }
        }
        __syncwarp();

#pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            const packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
            int row = lane_id / 4;
            int col = lane_id % 4 * 2 + j * INSN_N;
            *reinterpret_cast<half2 *>(&mat[row][col]) = fsum.data[1];
            *reinterpret_cast<half2 *>(&mat[row][col + 8]) = fsum.data[3];
        }
        __syncwarp();

#pragma unroll
        for (int row = 0; row < 8; ++row) {
            int global_m = bm * BLOCK_M + warp_id * WARP_M + i * INSN_M + 8 + row;
            int global_n = bn * BLOCK_N + lane_id * 4;
            if (global_m < M && global_n + 3 < N) {
                half4_pack pack = *reinterpret_cast<half4_pack *>(&mat[row][lane_id * 4]);
                add_bias_to_pack(pack, bias, global_n);
                *reinterpret_cast<half4_pack *>(&output[global_m * N + global_n]) = pack;
            }
        }
        __syncwarp();
    }
}

__global__ void linear_w4a4_kernel(const uint4 *act,
                                    const uint4 *wgt,
                                    const packed_ascale_t *ascales,
                                    const packed_wscale_t *wscales,
                                    half *out,
                                    const half *bias,
                                    int M,
                                    int N,
                                    int K) {
    int bm = blockIdx.x;
    int bn = blockIdx.y;
    int k_groups = K / WARP_K;

    const uint4 *act_block = act + bm * k_groups * NUM_WARPS * WARP_M_TILES * WARP_SIZE;
    const uint4 *wgt_block = wgt + bn * k_groups * WARP_N_TILES * WARP_SIZE;
    const packed_ascale_t *ascale_block = ascales + bm * k_groups * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES;
    const packed_wscale_t *wscale_block = wscales + bn * k_groups * WSCALES_NUM_PACKS * WSCALES_VALID_LANES;

    constexpr int NUM_STAGES = 2;
    act_warp_t A[NUM_STAGES];
    wgt_warp_t W[NUM_STAGES];
    ascale_warp_t ascale0;
    wscale_warp_t wscale0;
    std::array<packed_psum_t, WARP_M_TILES * WARP_N_TILES> ipsum;
    fpsum_warp_t fpsum;

#pragma unroll
    for (int idx = 0; idx < WARP_M_TILES * WARP_N_TILES; ++idx) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ipsum[idx].data[j] = 0;
        }
    }

    // Per-token activation scale and per-channel weight scale are K-invariant, so load only group 0.
    load_ascale_tile(ascale_block, 0, M, ascale0, true);
    load_wscale_tile(wscale_block, 0, N, wscale0, true);

#pragma unroll
    for (int k = 0; k < NUM_STAGES - 1; ++k) {
        load_act_tile(act_block, k, K, A[k], k < k_groups);
        load_wgt_tile(wgt_block, k, K, W[k], k < k_groups);
    }

    // Register-level two-stage K pipeline, but accumulate int32 across all K and dequantize once at the end.
    for (int k1 = 0; k1 < k_groups; k1 += NUM_STAGES) {
#pragma unroll
        for (int k2 = 0; k2 < NUM_STAGES; ++k2) {
            int next_k = k1 + k2 + NUM_STAGES - 1;
            int load_idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
            bool pred = next_k < k_groups;
            load_act_tile(act_block, next_k, K, A[load_idx], pred);
            load_wgt_tile(wgt_block, next_k, K, W[load_idx], pred);
            accumulate_int4(A[k2], W[k2], ipsum);
        }
    }

    finalize_int_accum(ipsum, ascale0, wscale0, fpsum);
    store_fpsum(fpsum, out, bias, M, N, bm, bn);
}

static void check_common(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor ceil_empty_like_layout_bytes(int64_t rows, int64_t packed_cols, const torch::TensorOptions &options) {
    return torch::empty({rows, packed_cols}, options.dtype(torch::kUInt8));
}

std::vector<torch::Tensor> pack_weight_with_scale(torch::Tensor weight, torch::Tensor channel_scales) {
    check_common(weight, "weight");
    check_common(channel_scales, "channel_scales");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be float16");
    TORCH_CHECK(channel_scales.scalar_type() == torch::kFloat16, "channel_scales must be float16");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(channel_scales.dim() == 1 || (channel_scales.dim() == 2 && channel_scales.size(1) == 1),
                "channel_scales must be [N] or [N, 1]");
    int64_t N64 = weight.size(0);
    int64_t K64 = weight.size(1);
    TORCH_CHECK(channel_scales.numel() == N64, "channel_scales length must equal N");
    TORCH_CHECK(N64 % BLOCK_N == 0, "N must be a multiple of ", BLOCK_N);
    TORCH_CHECK(K64 % WARP_K == 0, "K must be a multiple of ", WARP_K);
    TORCH_CHECK(N64 <= INT_MAX && K64 <= INT_MAX, "shape too large for int indexing");
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);

    auto qweight = torch::empty({N64, K64 / 2}, weight.options().dtype(torch::kUInt8));
    auto packed_scales = torch::empty({N64, K64 / WARP_K}, weight.options());
    auto scale_flat = channel_scales.contiguous().view({N64});

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N / WARP_N, K / WARP_K);
    weight_quantize_per_channel_kernel<<<grid, WARP_SIZE, 0, stream>>>(reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
                                                                       reinterpret_cast<const half *>(scale_flat.data_ptr<at::Half>()),
                                                                       reinterpret_cast<uint4 *>(qweight.data_ptr<uint8_t>()),
                                                                       reinterpret_cast<packed_wscale_t *>(packed_scales.data_ptr<at::Half>()),
                                                                       N,
                                                                       K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qweight, packed_scales, scale_flat};
}

std::vector<torch::Tensor> quantize_weight_per_channel(torch::Tensor weight) {
    check_common(weight, "weight");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be float16");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    int64_t N64 = weight.size(0);
    int64_t K64 = weight.size(1);
    TORCH_CHECK(N64 % BLOCK_N == 0, "N must be a multiple of ", BLOCK_N);
    TORCH_CHECK(K64 % WARP_K == 0, "K must be a multiple of ", WARP_K);
    TORCH_CHECK(N64 <= INT_MAX && K64 <= INT_MAX, "shape too large for int indexing");
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);

    auto qweight = torch::empty({N64, K64 / 2}, weight.options().dtype(torch::kUInt8));
    auto channel_scales = torch::empty({N64}, weight.options());
    auto packed_scales = torch::empty({N64, K64 / WARP_K}, weight.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    token_scale_kernel<<<N, 256, 0, stream>>>(reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
                                              reinterpret_cast<half *>(channel_scales.data_ptr<at::Half>()),
                                              N,
                                              K,
                                              1.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid(N / WARP_N, K / WARP_K);
    weight_quantize_per_channel_kernel<<<grid, WARP_SIZE, 0, stream>>>(reinterpret_cast<const half *>(weight.data_ptr<at::Half>()),
                                                                       reinterpret_cast<const half *>(channel_scales.data_ptr<at::Half>()),
                                                                       reinterpret_cast<uint4 *>(qweight.data_ptr<uint8_t>()),
                                                                       reinterpret_cast<packed_wscale_t *>(packed_scales.data_ptr<at::Half>()),
                                                                       N,
                                                                       K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qweight, packed_scales, channel_scales};
}

torch::Tensor linear_dynamic_impl(torch::Tensor input, torch::Tensor qweight, torch::Tensor wscales, const torch::Tensor *bias, double clip_ratio) {
    check_common(input, "input");
    check_common(qweight, "qweight");
    check_common(wscales, "wscales");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, "qweight must be uint8");
    TORCH_CHECK(wscales.scalar_type() == torch::kFloat16, "wscales must be float16");
    if (bias != nullptr) {
        check_common(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat16, "bias must be float16");
        TORCH_CHECK(bias->dim() == 1, "bias must be [N]");
    }
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be [N, K / 2]");
    TORCH_CHECK(wscales.dim() == 2, "wscales must be packed as [N, K / 64] with repeated per-channel scales");

    int64_t M64 = input.size(0);
    int64_t K64 = input.size(1);
    int64_t N64 = qweight.size(0);
    TORCH_CHECK(qweight.size(1) == K64 / 2, "qweight second dim must be K / 2");
    TORCH_CHECK(wscales.size(0) == N64 && wscales.size(1) == K64 / WARP_K, "bad packed wscales shape");
    TORCH_CHECK(N64 % BLOCK_N == 0, "N must be a multiple of ", BLOCK_N);
    TORCH_CHECK(K64 % WARP_K == 0, "K must be a multiple of ", WARP_K);
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "shape too large for int indexing");
    TORCH_CHECK(bias == nullptr || bias->numel() == N64, "bias length must equal N");
    int64_t paddedM64 = ((M64 + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    int M = static_cast<int>(M64);
    int paddedM = static_cast<int>(paddedM64);
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);

    auto act_scales = torch::empty({M64}, input.options());
    auto qact = torch::empty({paddedM64, K64 / 2}, input.options().dtype(torch::kUInt8));
    auto ascales = torch::empty({paddedM64, K64 / WARP_K}, input.options());
    auto out = torch::empty({M64, N64}, input.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    token_scale_kernel<<<M, 256, 0, stream>>>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
                                              reinterpret_cast<half *>(act_scales.data_ptr<at::Half>()),
                                              M,
                                              K,
                                              static_cast<float>(clip_ratio));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 qgrid(paddedM / WARP_M, K / WARP_K);
    act_quantize_per_token_kernel<<<qgrid, WARP_SIZE, 0, stream>>>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
                                                                   reinterpret_cast<const half *>(act_scales.data_ptr<at::Half>()),
                                                                   reinterpret_cast<uint4 *>(qact.data_ptr<uint8_t>()),
                                                                   reinterpret_cast<packed_ascale_t *>(ascales.data_ptr<at::Half>()),
                                                                   M,
                                                                   K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 ggrid(paddedM / BLOCK_M, N / BLOCK_N);
    linear_w4a4_kernel<<<ggrid, WARP_SIZE * NUM_WARPS, 0, stream>>>(reinterpret_cast<const uint4 *>(qact.data_ptr<uint8_t>()),
                                                                    reinterpret_cast<const uint4 *>(qweight.data_ptr<uint8_t>()),
                                                                    reinterpret_cast<const packed_ascale_t *>(ascales.data_ptr<at::Half>()),
                                                                    reinterpret_cast<const packed_wscale_t *>(wscales.data_ptr<at::Half>()),
                                                                    reinterpret_cast<half *>(out.data_ptr<at::Half>()),
                                                                    bias == nullptr ? nullptr : reinterpret_cast<const half *>(bias->data_ptr<at::Half>()),
                                                                    M,
                                                                    N,
                                                                    K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor linear_dynamic(torch::Tensor input, torch::Tensor qweight, torch::Tensor wscales, double clip_ratio) {
    return linear_dynamic_impl(input, qweight, wscales, nullptr, clip_ratio);
}

torch::Tensor linear_dynamic_bias(torch::Tensor input, torch::Tensor qweight, torch::Tensor wscales, torch::Tensor bias, double clip_ratio) {
    return linear_dynamic_impl(input, qweight, wscales, &bias, clip_ratio);
}

std::vector<torch::Tensor> quantize_activation_per_token(torch::Tensor input, double clip_ratio) {
    check_common(input, "input");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    int64_t M64 = input.size(0);
    int64_t K64 = input.size(1);
    TORCH_CHECK(K64 % WARP_K == 0, "K must be a multiple of ", WARP_K);
    TORCH_CHECK(M64 <= INT_MAX && K64 <= INT_MAX, "shape too large for int indexing");
    int64_t paddedM64 = ((M64 + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    int M = static_cast<int>(M64);
    int paddedM = static_cast<int>(paddedM64);
    int K = static_cast<int>(K64);

    auto act_scales = torch::empty({M64}, input.options());
    auto qact = torch::empty({paddedM64, K64 / 2}, input.options().dtype(torch::kUInt8));
    auto ascales = torch::empty({paddedM64, K64 / WARP_K}, input.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    token_scale_kernel<<<M, 256, 0, stream>>>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
                                              reinterpret_cast<half *>(act_scales.data_ptr<at::Half>()),
                                              M,
                                              K,
                                              static_cast<float>(clip_ratio));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 qgrid(paddedM / WARP_M, K / WARP_K);
    act_quantize_per_token_kernel<<<qgrid, WARP_SIZE, 0, stream>>>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
                                                                   reinterpret_cast<const half *>(act_scales.data_ptr<at::Half>()),
                                                                   reinterpret_cast<uint4 *>(qact.data_ptr<uint8_t>()),
                                                                   reinterpret_cast<packed_ascale_t *>(ascales.data_ptr<at::Half>()),
                                                                   M,
                                                                   K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qact, ascales, act_scales};
}

torch::Tensor linear_prequantized(torch::Tensor qact, torch::Tensor ascales, torch::Tensor qweight, torch::Tensor wscales) {
    check_common(qact, "qact");
    check_common(ascales, "ascales");
    check_common(qweight, "qweight");
    check_common(wscales, "wscales");
    TORCH_CHECK(qact.scalar_type() == torch::kUInt8, "qact must be uint8");
    TORCH_CHECK(ascales.scalar_type() == torch::kFloat16, "ascales must be float16");
    TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, "qweight must be uint8");
    TORCH_CHECK(wscales.scalar_type() == torch::kFloat16, "wscales must be float16");
    TORCH_CHECK(qact.dim() == 2, "qact must be [M, K / 2]");
    TORCH_CHECK(ascales.dim() == 2, "ascales must be [M, K / 64]");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be [N, K / 2]");
    TORCH_CHECK(wscales.dim() == 2, "wscales must be [N, K / 64]");

    int64_t M64 = qact.size(0);
    int64_t K64 = qact.size(1) * 2;
    int64_t N64 = qweight.size(0);
    TORCH_CHECK(qweight.size(1) == K64 / 2, "qweight second dim must be K / 2");
    TORCH_CHECK(ascales.size(0) == M64 && ascales.size(1) == K64 / WARP_K, "bad ascales shape");
    TORCH_CHECK(wscales.size(0) == N64 && wscales.size(1) == K64 / WARP_K, "bad wscales shape");
    TORCH_CHECK(M64 % BLOCK_M == 0, "M must be a multiple of ", BLOCK_M);
    TORCH_CHECK(N64 % BLOCK_N == 0, "N must be a multiple of ", BLOCK_N);
    TORCH_CHECK(K64 % WARP_K == 0, "K must be a multiple of ", WARP_K);
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "shape too large for int indexing");
    int M = static_cast<int>(M64);
    int N = static_cast<int>(N64);
    int K = static_cast<int>(K64);

    auto out = torch::empty({M64, N64}, ascales.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 ggrid(M / BLOCK_M, N / BLOCK_N);
    linear_w4a4_kernel<<<ggrid, WARP_SIZE * NUM_WARPS, 0, stream>>>(reinterpret_cast<const uint4 *>(qact.data_ptr<uint8_t>()),
                                                                    reinterpret_cast<const uint4 *>(qweight.data_ptr<uint8_t>()),
                                                                    reinterpret_cast<const packed_ascale_t *>(ascales.data_ptr<at::Half>()),
                                                                    reinterpret_cast<const packed_wscale_t *>(wscales.data_ptr<at::Half>()),
                                                                    reinterpret_cast<half *>(out.data_ptr<at::Half>()),
                                                                    nullptr,
                                                                    M,
                                                                    N,
                                                                    K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace vdm_w4a4

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weight_per_channel", &vdm_w4a4::quantize_weight_per_channel,
          "Pack fp16 weights into signed W4 per-channel layout and repeated packed scale layout for GEMM");
    m.def("pack_weight_with_scale", &vdm_w4a4::pack_weight_with_scale,
          "Pack fp16 weights into signed W4 layout using caller-provided per-channel scales");
    m.def("linear_dynamic", &vdm_w4a4::linear_dynamic,
          py::arg("input"), py::arg("qweight"), py::arg("wscales"), py::arg("clip_ratio") = 1.0,
          "W4A4 linear with signed W4 per-channel weights and signed A4 per-token dynamic activation quantization");
    m.def("linear_dynamic_bias", &vdm_w4a4::linear_dynamic_bias,
          py::arg("input"), py::arg("qweight"), py::arg("wscales"), py::arg("bias"), py::arg("clip_ratio") = 1.0,
          "W4A4 linear with fused bias epilogue");
    m.def("quantize_activation_per_token", &vdm_w4a4::quantize_activation_per_token,
          py::arg("input"), py::arg("clip_ratio") = 1.0,
          "Quantize fp16 activations into signed A4 per-token layout");
    m.def("linear_prequantized", &vdm_w4a4::linear_prequantized,
          "Run only the W4A4 GEMM on pre-quantized activation and weight tensors");
}
