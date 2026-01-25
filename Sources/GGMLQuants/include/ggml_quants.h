/*
 * GGML Dequantization - Standalone Implementation
 * Extracted from GGML (https://github.com/ggml-org/ggml)
 *
 * This is a standalone version of GGML dequantization functions
 * for use in external projects.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Platform-specific macros
// ============================================================================

#ifndef GGML_RESTRICT
#define GGML_RESTRICT restrict
#endif

#ifndef GGML_API
#ifdef __cplusplus
#define GGML_API extern "C"
#else
#define GGML_API
#endif
#endif

// ============================================================================
// Type definitions
// ============================================================================

typedef uint16_t ggml_fp16_t;
typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

// ============================================================================
// FP16 conversion functions
// ============================================================================

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

// E8M0 to FP32 conversion for MXFP4
static inline float ggml_compute_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = ((uint32_t)(x - 1)) << 23;
    }
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = bits;
    return fp32.as_value;
}

#define GGML_E8M0_TO_FP32_HALF(x) ggml_compute_e8m0_to_fp32_half(x)

// ============================================================================
// Block size constants
// ============================================================================

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32
#define QK4_NL 32
#define QK_MXFP4 32
#define QK_K 256
#define K_SCALE_SIZE 12
#define IQ3S_N_SCALE (QK_K/64)
#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

// ============================================================================
// Block structure definitions
// ============================================================================

// 4-bit quantization (4.5 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d;           // delta (scale factor)
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

// 4-bit quantization with min (5.0 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d; // delta (scale)
    ggml_half m; // min
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;

// 5-bit quantization (5.5 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;

// 5-bit quantization with min (6.0 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d;           // delta
    ggml_half m;           // min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;

// 8-bit quantization (8.5 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// 8-bit quantization with sum (9.0 bits per weight)
// 32 weights per block
typedef struct {
    ggml_half d;       // delta
    ggml_half s;       // d * sum(qs[i])
    int8_t qs[QK8_1];  // quants
} block_q8_1;

// IQ4_NL quantization (4.25 bits per weight with non-linear values)
// 32 weights per block
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;

// ============================================================================
// K-quantization structures (super-block based, 256 weights per block)
// ============================================================================

// 2-bit quantization (2.625 bits per weight)
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_half d;             // super-block scale for quantized scales
    ggml_half dmin;          // super-block scale for quantized mins
} block_q2_K;

// 3-bit quantization (3.4375 bits per weight)
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

// 4-bit quantization (4.5 bits per weight)
typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

// 5-bit quantization (5.5 bits per weight)
typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

// 6-bit quantization (6.5625 bits per weight)
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

// 8-bit quantization (used for temporary quantization)
typedef struct {
    ggml_half d;         // delta
    int8_t  qs[QK_K];    // quants
    int16_t bsums[QK_K/16]; // sum of quants in blocks of 16
} block_q8_K;

// ============================================================================
// Function declarations - Dequantization
// ============================================================================

GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq4_nl(const block_iq4_nl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
