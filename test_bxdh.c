#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <pthread.h>

#define N 512
#define Q 18433
#define ROOT 5
#define LOGN 9
#define ETA 4
#define BITS_PER_COEFF 14
#define COMPRESSED_POLY_SIZE ((N * BITS_PER_COEFF + 7) / 8)
#define BARRETT_SHIFT 48
#define bits_needed (2 * ETA * N)
#define byte_len ((bits_needed + 7) / 8)

// Error code definitions
typedef enum {
    SUCCESS = 0,
    ERROR_RANDOM_FAILED = -1,
    ERROR_MEMORY_ALLOCATION = -2,
    ERROR_CRYPTO_OPERATION = -3
} result_t;

// Function declarations
void precompute_roots();
int64_t mod_q(int64_t x);
int64_t pow_mod(int64_t base, int64_t exp);
int bitrev(int x, int logn);
void ntt_512(int64_t *a);
void intt_512(int64_t *a);
void poly_mul_512(const int64_t *a, const int64_t *b, int64_t *c);
void generate_random_poly(int64_t *poly);

// CBD2 related functions
int bit_count_u32(uint32_t x);
void cbd_eta(const uint8_t *input_bytes, int64_t *coeffs);
void prg_gaussian(const uint8_t *seed, int64_t *result);
void poly_add_inplace(int64_t *a, const int64_t *b);
void poly_sub_inplace(int64_t *a, const int64_t *b);

// Precomputation struct for 512-point NTT
typedef struct {
    int64_t a_ntt[512];  // Precomputed NTT form of polynomial a
    int64_t a_original[N];
    int64_t inv_N_512;
} precomputed_a_t;

// Precomputed twiddle factors for 512-point NTT
static int64_t roots_512_layered[LOGN][256];
static int64_t inv_roots_512_layered[LOGN][256];
static int64_t Q_2_512_POW_MOD;

// Batch random number generation
#define BATCH_SIZE 65536
static uint8_t rand_batch[BATCH_SIZE];
static size_t batch_offset = BATCH_SIZE;
static pthread_mutex_t batch_mutex = PTHREAD_MUTEX_INITIALIZER;

// ==================== Optimization utility functions ====================
// Precompute Barrett reduction parameters
static const int64_t BARRETT_M = ((1LL << BARRETT_SHIFT) + Q - 1) / Q;

// Barrett reduction - fast modular operation
int64_t mod_q(int64_t x) {
    int64_t q = 0, r = 0;
    q = (x * BARRETT_M) >> BARRETT_SHIFT;
    r = x - q * Q - Q;
    r += (r >> 63) & Q;
    return r;
}

// Fast modular exponentiation
int64_t pow_mod(int64_t base, int64_t exp) {
    int64_t result = 1;
    base = mod_q(base);
    
    // Handle special case of exp=0 (constant time)
    int64_t is_zero_exp = (exp == 0) - 1;
    result = (result & ~is_zero_exp) | (1 & is_zero_exp);
    
    while (exp > 0) {
        int should_multiply = (exp & 1);
        int64_t temp = mod_q((__int128)result * base);
        
        result = (temp & -should_multiply) | (result & ~(-should_multiply));
        base = mod_q((__int128)base * base);
        exp >>= 1;
    }
    return result;
}

int64_t add_mod_q(int64_t x, int64_t y) {
    int64_t r = x + y - Q;
    r += (r >> 63) & Q;
    return r;
}

int64_t sub_mod_q(int64_t x, int64_t y) {
    int64_t r = x - y;
    r += (r >> 63) & Q;
    return r;
}

// Batch secure random number generation
int batched_secure_random(uint8_t *buf, size_t len) {
    if (len > BATCH_SIZE / 2) {
        return RAND_bytes(buf, len) == 1 ? SUCCESS : ERROR_RANDOM_FAILED;
    }
    
    pthread_mutex_lock(&batch_mutex);
    
    if (batch_offset + len > BATCH_SIZE) {
        if (RAND_bytes(rand_batch, BATCH_SIZE) != 1) {
            pthread_mutex_unlock(&batch_mutex);
            return ERROR_RANDOM_FAILED;
        }
        batch_offset = 0;
    }
    
    memcpy(buf, rand_batch + batch_offset, len);
    batch_offset += len;
    
    pthread_mutex_unlock(&batch_mutex);
    return SUCCESS;
}

// ==================== NTT related functions ====================

void butterfly(int64_t *a, int64_t *b, int64_t w) {
    int64_t u = *a;
    int64_t v = mod_q((__int128)(*b) * w);
    *a = add_mod_q(u, v);
    *b = sub_mod_q(u, v);
}

// Bit reversal function
int bitrev(int x, int logn) {
    int r = 0;
    for (int i = 0; i < logn; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

// Bit-reverse permutation
void bitrev_permute(int64_t *a, int n, int logn) {
    for (int i = 0; i < n; ++i) {
        int j = bitrev(i, logn);
        if (j > i) {
            int64_t t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}

void precompute_roots() {
    int64_t g = ROOT;

    /* 512-point NTT roots */
    int64_t w512 = pow_mod(g, (Q - 1) / 512);
    int layer_512 = 0;
    
    for (int len = 2; len <= 512; len <<= 1, layer_512++) {
        int step = 512 / len;
        int64_t w_len = pow_mod(w512, step);
        int64_t w_current = 1;
        
        for (int j = 0; j < len / 2; j++) {
            roots_512_layered[layer_512][j] = w_current;
            inv_roots_512_layered[layer_512][j] = pow_mod(w_current, Q - 2);
            w_current = mod_q((__int128)w_current * w_len);
        }
    }

    Q_2_512_POW_MOD = pow_mod(512, Q - 2);
}

void ntt_512(int64_t *a) {
    // Permute to bit-reversed order
    bitrev_permute(a, 512, LOGN);

    int layer = 0;
    for (int len = 2; len <= 512; len <<= 1, layer++) {
        int half = len >> 1;
        for (int i = 0; i < 512; i += len) {
            const int64_t *w_ptr = roots_512_layered[layer];
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                butterfly(a_ptr + j, b_ptr + j, w_ptr[j]);
            }
        }
    }
}

void intt_512(int64_t *a) {
    int layer = LOGN - 1;
    for (int len = 512; len > 1; len >>= 1, layer--) {
        int half = len >> 1;
        for (int i = 0; i < 512; i += len) {
            const int64_t *w_ptr = inv_roots_512_layered[layer];
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                int64_t u = a_ptr[j];
                int64_t v = b_ptr[j];
                a_ptr[j] = mod_q(u + v);
                b_ptr[j] = mod_q((__int128)(u - v) * w_ptr[j]);
            }
        }
    }
    
    // Scale by 1/N
    for (int i = 0; i < 512; i++) {
        a[i] = mod_q((__int128)a[i] * Q_2_512_POW_MOD);
    }
    
    // Restore natural order
    bitrev_permute(a, 512, LOGN);
}

// ==================== Polynomial operations ====================

// Optimized 512-point polynomial multiplication using NTT
void poly_mul_512(const int64_t *a, const int64_t *b, int64_t *c) {
    int64_t A[512], B[512];
    
    // Copy and compute NTT
    memcpy(A, a, sizeof(int64_t) * 512);
    memcpy(B, b, sizeof(int64_t) * 512);
    
    ntt_512(A);
    ntt_512(B);
    
    // Pointwise multiplication
    for (int i = 0; i < 512; i++) {
        A[i] = mod_q((__int128)A[i] * B[i]);
    }
    
    // Inverse NTT
    intt_512(A);
    
    // Copy result
    memcpy(c, A, sizeof(int64_t) * 512);
}

// ==================== Precomputation optimization ====================

// Precomputation function for 512-point NTT
void precompute_a_ntt(int64_t *a, precomputed_a_t *precomputed) {
    // Copy original polynomial
    memcpy(precomputed->a_original, a, sizeof(int64_t) * N);
    
    // Pad to 512 points if necessary (for N=512, no padding needed)
    int64_t padded[512];
    memcpy(padded, a, sizeof(int64_t) * 512);
    
    // Compute NTT
    ntt_512(padded);
    
    // Store NTT result
    memcpy(precomputed->a_ntt, padded, sizeof(int64_t) * 512);
    
    // Save scaling factor
    precomputed->inv_N_512 = Q_2_512_POW_MOD;
}

// Optimized polynomial multiplication using precomputed NTT
void poly_mul_precomputed(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c) {
    int64_t B_ntt[512];
    
    // Copy and compute NTT of b
    memcpy(B_ntt, b, sizeof(int64_t) * 512);
    ntt_512(B_ntt);
    
    // Pointwise multiplication in frequency domain
    int64_t product[512];
    for (int i = 0; i < 512; i++) {
        product[i] = mod_q((__int128)precomputed->a_ntt[i] * B_ntt[i]);
    }
    
    // Inverse NTT back to time domain
    intt_512(product);
    
    // Copy result
    memcpy(c, product, sizeof(int64_t) * 512);
}

// Polynomial addition (in-place operation)
void poly_add_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i += 4) {
        a[i]   = mod_q(a[i]   + b[i]);
        a[i+1] = mod_q(a[i+1] + b[i+1]);
        a[i+2] = mod_q(a[i+2] + b[i+2]);
        a[i+3] = mod_q(a[i+3] + b[i+3]);
    }
}

// Polynomial subtraction (in-place operation)
void poly_sub_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i += 4) {
        a[i]   = mod_q(a[i]   - b[i]);
        a[i+1] = mod_q(a[i+1] - b[i+1]);
        a[i+2] = mod_q(a[i+2] - b[i+2]);
        a[i+3] = mod_q(a[i+3] - b[i+3]);
    }
}

// Polynomial addition (output to c)
void poly_add_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i += 4) {
        c[i]   = mod_q(a[i]   + b[i]);
        c[i+1] = mod_q(a[i+1] + b[i+1]);
        c[i+2] = mod_q(a[i+2] + b[i+2]);
        c[i+3] = mod_q(a[i+3] + b[i+3]);
    }
}

// Polynomial subtraction (output to c)
void poly_sub_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i += 4) {
        c[i]   = mod_q(a[i]   - b[i]);
        c[i+1] = mod_q(a[i+1] - b[i+1]);
        c[i+2] = mod_q(a[i+2] - b[i+2]);
        c[i+3] = mod_q(a[i+3] - b[i+3]);
    }
}

// ==================== Random polynomial generation ====================

void generate_random_poly(int64_t *poly) {
    uint8_t buf[N * 2];
    int filled = 0;

    while (filled < N) {
        if (batched_secure_random(buf, sizeof(buf)) != SUCCESS) {
            fprintf(stderr, "Error: RAND_bytes failed.\n");
            exit(1);
        }

        for (int i = 0; i + 1 < (int)sizeof(buf) && filled < N; i += 2) {
            uint16_t r = ((uint16_t)buf[i] << 8) | buf[i + 1];
            if (r < Q) {
                poly[filled++] = r;
            }
        }
    }
}

// Bit counting helper function
int bit_count_u32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    return (x * 0x01010101) >> 24;
}

// SHAKE256 extendable output function
int shake256(const uint8_t *input, size_t input_len, 
             uint8_t *output, size_t output_len) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (ctx == NULL) {
        return ERROR_CRYPTO_OPERATION;
    }

    const EVP_MD *md = EVP_shake256();
    if (md == NULL) {
        EVP_MD_CTX_free(ctx);
        return ERROR_CRYPTO_OPERATION;
    }

    if (EVP_DigestInit_ex(ctx, md, NULL) != 1 ||
        EVP_DigestUpdate(ctx, input, input_len) != 1 ||
        EVP_DigestFinalXOF(ctx, output, output_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return ERROR_CRYPTO_OPERATION;
    }

    EVP_MD_CTX_free(ctx);
    return SUCCESS;
}

// CBD implementation for eta=2
void cbd_eta2(const uint8_t *buf, int64_t *r) {
    for (int i = 0; i < N / 8; i++) {
        uint64_t t = 0;
        // Read 4 bytes (8 coefficients = 32 bits)
        for (int j = 0; j < 4; j++)
            t |= ((uint64_t)buf[4 * i + j]) << (8 * j);

        // Compute a,b popcount
        uint64_t d = t & 0x55555555ULL;       // Extract even bits
        d += (t >> 1) & 0x55555555ULL;        // Extract odd bits and add

        // Each 2 bits in d encode the difference a-b
        for (int j = 0; j < 8; j++) {
            uint64_t a = (d >> (4 * j)) & 0x3;      // bits 0-1
            uint64_t b = (d >> (4 * j + 2)) & 0x3;  // bits 2-3
            r[8 * i + j] = (int64_t)(a - b);
        }
    }
}

// CBD implementation for eta=4
void cbd_eta(const uint8_t *buf, int64_t *r) {
    for (int i = 0; i < N / 8; i++) {
        uint64_t t = 0;
        // read 8 bytes = 8 coffs
        for (int j = 0; j < 8; j++)
            t |= ((uint64_t)buf[8*i + j]) << (8*j);

        // Kyber-style bit slicing
        uint64_t d = t & 0x1111111111111111ULL;          // Let bit positions {0,4,8,...,60}
        d += (t >> 1) & 0x1111111111111111ULL;
        d += (t >> 2) & 0x1111111111111111ULL;
        d += (t >> 3) & 0x1111111111111111ULL;

        for (int j = 0; j < 8; j++) {
            uint64_t a = (d >> (8*j)) & 0xF;      // Low 4bit = a
            uint64_t b = (d >> (8*j + 4)) & 0xF;  // High 4bit = b
            r[8*i + j] = (int64_t)(a - b);
        }
    }
}

// Generate Gaussian distribution polynomial
void prg_gaussian(const uint8_t *seed, int64_t *result) {
    uint8_t *random_bytes = (uint8_t *)malloc(byte_len);
    if (!random_bytes) {
        fprintf(stderr, "malloc failed in prg_gaussian\n");
        exit(1);
    }
    shake256(seed, 33, random_bytes, byte_len);
    cbd_eta(random_bytes, result);
    free(random_bytes);
}

// Optimized s and b generation function using 512-point NTT
void generate_s_b_gaussian_optimized(const uint8_t seed32[32], const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed_s[33] = {0}, seed_e[33] = {0};
    memcpy(seed_s, seed32, 32);
    memcpy(seed_e, seed32, 32);
    seed_s[32] = 's';
    seed_e[32] = 'e';

    int64_t e[N] = {0};

    prg_gaussian(seed_s, s);
    prg_gaussian(seed_e, e);

    // Directly compute into b using precomputed NTT
    poly_mul_precomputed(precomputed, s, b);
    
    // Add error term in-place: b = b + 2e
    for (int i = 0; i < N; i++) {
        b[i] = mod_q(b[i] + 2 * e[i]);
    }
}

// ==================== Key generation ====================

result_t keygen(const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed[32] = {0};
    if (batched_secure_random(seed, 32) != SUCCESS) {
        return ERROR_RANDOM_FAILED;
    }
    generate_s_b_gaussian_optimized(seed, precomputed, s, b);
    return SUCCESS;
}

// ==================== Helper functions ====================

#include <inttypes.h>
#include <stdbool.h>

// canonical modular reduction to [0, Q-1]

// centered: map canonical [0, Q-1] -> symmetric interval
// tie_to_negative: when ux == Q/2 and Q is even, maps(true => to ux-Q)
static inline int64_t centered_from_u64(uint64_t ux, bool tie_to_negative) {
    uint64_t half = (uint64_t)(Q / 2); // floor(Q/2)
    if (ux > half) {
        return (int64_t)((int64_t)ux - (int64_t)Q); // if negative
    }
    if (ux == half && (Q % 2 == 0) && tie_to_negative) {
        // map bound to negative
        return (int64_t)((int64_t)ux - (int64_t)Q);
    }
    //otherwise
    return (int64_t)ux;
}

// wrapper: accepts possibly not-canonical x
static inline int64_t centered(int64_t x, bool tie_to_negative) {
    uint64_t ux = mod_q(x);
    return centered_from_u64(ux, tie_to_negative);
}

// sig_0: threshold based on q/4
// t = floor(Q/4).
// when |x| > t  return 1.
// when |x| < = t return 0 
static inline int64_t sig_0(int64_t x, bool tie_to_negative) {
    int64_t cx = centered(x, tie_to_negative); // in symmetric interval
    uint64_t t = (uint64_t)(Q / 4);            // floor
    int64_t mask11 = cx >> 63;                   // sign mask
    uint64_t absx = (uint64_t)((cx ^ mask11) - mask11);
    // return 1 iff absx > t
    if (absx > t) return 1;
    // tie or small -> 0
    return 0;
}

// sig_1: similar but threshold = floor(Q/4)+1
static inline int64_t sig_1(int64_t x, bool tie_to_negative) {
    int64_t cx = centered(x, tie_to_negative);
    uint64_t t = (uint64_t)(Q / 4) + 1;
    int64_t mask11 = cx >> 63;
    uint64_t absx = (uint64_t)((cx ^ mask11) - mask11);
    return (absx > t) ? 1 : 0;
}

// mod2: produce the recovered bit using w (hint in {0,1})
// Idea: apply same shift used in scheme, then take parity of centered representative.
// Implementation avoids UB: convert to canonical, then to centered int64_t, then parity via uint64_t.
static inline int64_t mod2(int64_t x, int64_t w, bool tie_to_negative) {
    // Typical scheme uses adding w * (q/2) (or (q-1)/2 in older buggy code).
    // Use exact q/2 (floor) here for consistency (both sides must use same).
    int64_t add = (int64_t)w * (int64_t)(Q / 2); // w is 0 or 1
    int64_t shifted = x + add;
    // compute centered representative
    int64_t c = centered(shifted, tie_to_negative); // c possibly negative
    uint64_t uc = (uint64_t)c; // two's complement representation; parity ok
    return (int64_t)(uc & 1u);
}

// ==================== RLDH protocol implementation ====================
const bool TIE_TO_NEGATIVE = true;
void astRLDH(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb) {
    uint8_t seed[33] = {0};
    if (batched_secure_random(seed, 33) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        exit(1);
    }

    int64_t e[N] = {0}, tmp[N] = {0};
    prg_gaussian(seed, e);

    // tmp = pk_a * sk_b (mod q) using 512-point NTT
    poly_mul_512(pk_a, sk_b, tmp);
    
    // tmp = tmp + 2e mod q
    for (int i = 0; i < N; i++) {
        tmp[i] = mod_q(tmp[i] + 2 * e[i]);
    }

    // Set kb to centered modulo (q)
    int64_t kb[N] = {0};


for (int i = 0; i < N; i++) {
    // ensure canonical representative
    tmp[i] = (int64_t)mod_q(tmp[i]);
    kb[i] = centered(tmp[i], TIE_TO_NEGATIVE);
}

for (int i = 0; i < N; i++) {
    int64_t sig = sig_0(kb[i], TIE_TO_NEGATIVE);
    htb[i] = sig;
    dhb[i] = mod2(kb[i], htb[i], TIE_TO_NEGATIVE);
}
}

void recRLDH(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha) {
    int64_t tmp[N] = {0};
    poly_mul_512(pk_b, sk_a, tmp);

    int64_t ka[N] = {0};
for (int i = 0; i < N; i++) {
    tmp[i] = (int64_t)mod_q(tmp[i]);
    ka[i] = centered(tmp[i], TIE_TO_NEGATIVE);
}

for (int i = 0; i < N; i++) {
    dha[i] = mod2(ka[i], htb[i], TIE_TO_NEGATIVE);
}
}

// ==================== Polynomial compression ====================
static const uint8_t BIT_MASKS[9] = {0, 1, 3, 7, 15, 31, 63, 127, 255};

void poly_to_compressed_bytes(const int64_t *poly, int n, uint8_t *bytes) {
    uint64_t buffer = 0;
    int bit_count = 0;
    int byte_pos = 0;
    
    for (int i = 0; i < n; i++) {
        uint16_t coeff = poly[i];
        
        buffer = (buffer << BITS_PER_COEFF) | coeff;
        bit_count += BITS_PER_COEFF;
        
        while (bit_count >= 8) {
            bytes[byte_pos++] = (buffer >> (bit_count - 8)) & 0xFF;
            bit_count -= 8;
        }
    }
    
    if (bit_count > 0) {
        bytes[byte_pos] = (buffer << (8 - bit_count)) & BIT_MASKS[bit_count];
    }
}

void compressed_bytes_to_poly(const uint8_t *bytes, int n, int64_t *poly) {
    uint64_t bit_buffer = 0;
    int bits_in_buffer = 0;
    int byte_index = 0;
    const int bits_per_coeff = BITS_PER_COEFF;
    
    for (int i = 0; i < n; i++) {
        while (bits_in_buffer < bits_per_coeff) {
            bit_buffer = (bit_buffer << 8) | bytes[byte_index++];
            bits_in_buffer += 8;
        }
        
        int shift = bits_in_buffer - bits_per_coeff;
        uint16_t coeff = (bit_buffer >> shift) & ((1 << bits_per_coeff) - 1);
        poly[i] = coeff;
        
        bits_in_buffer -= bits_per_coeff;
        bit_buffer &= (1ULL << bits_in_buffer) - 1;
    }
}

// ==================== KDF function ====================

int PRF_2(const int64_t dhb[N], const uint8_t *sid, size_t sid_len,
          uint8_t k[32], uint8_t k0[32]) {
    
    uint8_t dhb_compressed[COMPRESSED_POLY_SIZE];
    poly_to_compressed_bytes(dhb, N, dhb_compressed);
    
    size_t data_len = COMPRESSED_POLY_SIZE + sid_len;
    uint8_t data[data_len + 3];
    
    memcpy(data, dhb_compressed, COMPRESSED_POLY_SIZE);
    memcpy(data + COMPRESSED_POLY_SIZE, sid, sid_len);
    
    int success = 1;
    
    memcpy(data + data_len, "k", 1);
    if (shake256(data, data_len + 1, k, 32) != SUCCESS) {
        success = 0;
    }
    
    if (success) {
        memcpy(data + data_len, "k0", 2);
        if (shake256(data, data_len + 2, k0, 32) != SUCCESS) {
            success = 0;
        }
    }

    return success ? SUCCESS : ERROR_CRYPTO_OPERATION;
}

// ==================== Session ID construction ====================

size_t build_session_id(const int64_t lpk_a[N], const int64_t spk_a[N],
                       const int64_t epk_a[N], const int64_t sum_pk_b[N],
                       uint8_t *sid_output) {
    size_t offset = 0;
    
    memcpy(sid_output + offset, "A", 1);
    offset += 1;
    
    memcpy(sid_output + offset, "B", 1);
    offset += 1;
    
    poly_to_compressed_bytes(lpk_a, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    poly_to_compressed_bytes(spk_a, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    poly_to_compressed_bytes(epk_a, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    poly_to_compressed_bytes(sum_pk_b, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    return offset;
}

size_t build_session_id1(const int64_t sum_pk_a[N], const int64_t sum_pk_b[N],
                       uint8_t *sid_output) {
    size_t offset = 0;
    
    memcpy(sid_output + offset, "A", 1);
    offset += 1;
    
    memcpy(sid_output + offset, "B", 1);
    offset += 1;
    
    poly_to_compressed_bytes(sum_pk_a, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    poly_to_compressed_bytes(sum_pk_b, N, sid_output + offset);
    offset += COMPRESSED_POLY_SIZE;
    
    return offset;
}

// ==================== AES-GCM encryption/decryption ====================

int aes_gcm_encrypt(const uint8_t *key, size_t key_len,
                   const uint8_t *iv,
                   const uint8_t *aad, size_t aad_len,
                   const uint8_t *plaintext, size_t plaintext_len,
                   uint8_t *ciphertext) {
    
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len, ciphertext_len_out;
    
    if (key_len == 32) {
        EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL);
    } else {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    
    EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv);
    
    if (aad && aad_len > 0) {
        EVP_EncryptUpdate(ctx, NULL, &len, aad, aad_len);
    }
    
    if (!EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    ciphertext_len_out = len;
    
    if (!EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    ciphertext_len_out += len;
    
    if (!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, ciphertext + ciphertext_len_out)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    
    EVP_CIPHER_CTX_free(ctx);
    return 1;
}

int aes_gcm_decrypt(const uint8_t *key, size_t key_len,
                   const uint8_t *iv,
                   const uint8_t *aad, size_t aad_len,
                   uint8_t *ciphertext, size_t ciphertext_len,  
                   uint8_t *plaintext) {
    
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len, plaintext_len_out;
    
    if (key_len == 32) {
        EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL);
    } else {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    
    EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv);
    
    if (aad && aad_len > 0) {
        EVP_DecryptUpdate(ctx, NULL, &len, aad, aad_len);
    }
    
    if (!EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len - 16)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    plaintext_len_out = len;
    
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, ciphertext + ciphertext_len - 16);
    
    if (!EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    plaintext_len_out += len;
    
    EVP_CIPHER_CTX_free(ctx);
    return 1;
}

// ==================== Test functions ====================

#include <x86intrin.h>
uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}


void test_keygen() {
    int64_t a[N] = {0};
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt(a, &precomputed);

    int64_t s[N] = {0}, b[N] = {0};
    uint64_t total_cycles = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc();
        keygen(&precomputed, s, b);
        uint64_t end = rdtsc();
        total_cycles += (end - start);
    }
    
    printf("Average keygen cycles: %.2f\n", (double)total_cycles / 1000);
    printf("Average keygen cycles: %.3f Mcycles\n", (double)total_cycles / 1000 / 1e6);
}




void test_RLDH(){
    int64_t a[N] = {0};
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt(a, &precomputed);

    int64_t s[N] = {0}, b[N] = {0};
    int64_t s1[N] = {0}, b1[N] = {0}, htb[N] = {0}, dha[N] = {0}, dhb[N] = {0};
    keygen(&precomputed, s, b);
    keygen(&precomputed, s1, b1);
    uint64_t total_cycles = 0;
    uint64_t total_cycles1 = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc();
        astRLDH(b, s1, htb, dhb);
        uint64_t end = rdtsc();
        total_cycles += (end - start);
        start = rdtsc();
        recRLDH(b1, s, htb, dha);        
        end = rdtsc();
        total_cycles1 += (end - start);
        for(int j = 0; j < N; j++){
            if(dha[j] != dhb[j]){
                printf("Reconciliation failed\n");
                return;
            }
        }
    }
    printf("Average AstRLDH cycles: %.2f\n", (double)total_cycles / 1000);
    printf("Average AstRLDH cycles: %.3f Mcycles\n", (double)total_cycles / 1000 / 1e6);
    printf("Average RecRLDH cycles: %.2f\n", (double)total_cycles1 / 1000);
    printf("Average RecRLDH cycles: %.3f Mcycles\n", (double)total_cycles1 / 1000 / 1e6);
}

void test_dh_keygen() {
    
    int64_t a[N]={0};
    generate_random_poly(a);
    
    precomputed_a_t precomputed;  // Use stack memory
    precompute_a_ntt(a, &precomputed);
    
    int64_t lpk_a[N]={0}, lsk_b[N]={0}, lpk_b[N]={0}, lsk_a[N]={0};
    int64_t epk_a[N]={0}, esk_a[N]={0}, epk_b[N]={0}, esk_b[N]={0};
    int64_t spk_a[N]={0}, ssk_a[N]={0};
    int64_t htb[N]={0}, dhb[N]={0}, dha[N]={0};
    int64_t sum_sk_a[N]={0}, sum_pk_a[N]={0}, sum_sk_b[N]={0}, sum_pk_b[N]={0};
    
    // Generate various key pairs
    keygen(&precomputed, lsk_a, lpk_a);
    keygen(&precomputed, lsk_b, lpk_b);
    keygen(&precomputed, ssk_a, spk_a);
    
    uint8_t seed_a[32]={0}, seed_b[32]={0};
    if (batched_secure_random(seed_a, 32) != SUCCESS ||
        batched_secure_random(seed_b, 32) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        return;
    }

    generate_s_b_gaussian_optimized(seed_a, &precomputed, esk_a, epk_a);
    generate_s_b_gaussian_optimized(seed_b, &precomputed, esk_b, epk_b);
    
    // Calculate sum keys
    poly_add_q(lsk_a, ssk_a, sum_sk_a);
    poly_add_inplace(sum_sk_a, esk_a);
    poly_add_q(lpk_a, spk_a, sum_pk_a);
    poly_add_inplace(sum_pk_a, epk_a);
    poly_add_q(lsk_b, esk_b, sum_sk_b);
    poly_add_q(lpk_b, epk_b, sum_pk_b);
    
    // RLDH protocol
    astRLDH(sum_pk_a, sum_sk_b, htb, dhb);
    recRLDH(sum_pk_b, sum_sk_a, htb, dha);
    // SID 
    size_t sid_size = 2 + 4 * COMPRESSED_POLY_SIZE;
    uint8_t sid[sid_size];
    
    size_t sid_len = build_session_id(lpk_a, spk_a, epk_a, sum_pk_b, sid);
    
    // KDF
    uint8_t k[32]={0}, k0[32]={0};
    PRF_2(dhb, sid, sid_len, k, k0);
    
    // Verification process
    uint8_t sd_prime[32]={0};
    if (batched_secure_random(sd_prime, 32) != SUCCESS) {
        fprintf(stderr, "Random generation failed for sd_prime!\n");
        return;
    }
    
  // Step 1: compute h = SHAKE256(lpk_b + sd')
    uint8_t h[32]={0};
    EVP_MD_CTX *shake_ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(shake_ctx, EVP_shake256(), NULL);
    
    // Concatenate lpk_b and sd'
    uint8_t lpk_b_bytes[COMPRESSED_POLY_SIZE];
    poly_to_compressed_bytes(lpk_b, N, lpk_b_bytes);
    
    EVP_DigestUpdate(shake_ctx, lpk_b_bytes, COMPRESSED_POLY_SIZE);
    EVP_DigestUpdate(shake_ctx, sd_prime, 32);
    EVP_DigestFinalXOF(shake_ctx, h, 32);
    EVP_MD_CTX_free(shake_ctx);
    
    // Step 2: prepare data to encrypt: h || sd || sd'
    size_t plaintext_len = 32 + 32 + 32;  // h + sd + sd'
    uint8_t *plaintext = malloc(plaintext_len);
    if (!plaintext) {
        fprintf(stderr, "Memory allocation failed for plaintext\n");
        return;
    }
    
    memcpy(plaintext, h, 32);
    memcpy(plaintext + 32, seed_b, 32);  // sd
    memcpy(plaintext + 64, sd_prime, 32); // sd'
    
    // Step 3: Encrypt with AES-GCM
    size_t ciphertext_len = plaintext_len + 16;  // GCM tag
    uint8_t *ciphertext = malloc(ciphertext_len);
    uint8_t iv[12];  // GCM IV
    
    if (!ciphertext || batched_secure_random(iv, 12) != 0) {
        fprintf(stderr, "Memory allocation or IV generation failed\n");
        free(plaintext);
        free(ciphertext);
        return;
    }
    
    // Use k0 as AES key for encryption
    int encrypt_success = aes_gcm_encrypt(k0, 32, iv, 
                                         NULL, 0,  // No AAD
                                         plaintext, plaintext_len,
                                         ciphertext);
    
    if (!encrypt_success) {
        fprintf(stderr, "AES-GCM encryption failed\n");
        free(plaintext);
        free(ciphertext);
        return;
    }
    
    // Step 4: Decrypt with k0
    uint8_t k_prime[32]={0}, k0_prime[32]={0};
    PRF_2(dha, sid, sid_len, k_prime, k0_prime);
    uint8_t *decrypted = malloc(plaintext_len);
    if (!decrypted) {
        fprintf(stderr, "Memory allocation failed for decrypted data\n");
        free(plaintext);
        free(ciphertext);
        return;
    }

    // Note: ciphertext parameter const qualifier removed here
    int decrypt_success = aes_gcm_decrypt(k0_prime, 32, iv,
                                        NULL, 0,
                                        ciphertext, ciphertext_len,  
                                        decrypted);    
    if (!decrypt_success) {
        fprintf(stderr, "AES-GCM decryption failed\n");
        free(plaintext);
        free(ciphertext);
        free(decrypted);
        return;
    }
   // Step 5: separate sd and sd'
    uint8_t decrypted_h[32]={0}, decrypted_sd[32]={0}, decrypted_sd_prime[32]={0};
    memcpy(decrypted_h, decrypted, 32);
    memcpy(decrypted_sd, decrypted + 32, 32);
    memcpy(decrypted_sd_prime, decrypted + 64, 32);
    
    // Step 6: recover lpk_b' = sum_pk_b - epk_b
    int64_t s_b_recovered[N]={0}, b_b_recovered[N]={0};
    generate_s_b_gaussian_optimized(decrypted_sd, &precomputed, s_b_recovered, b_b_recovered);
    int64_t lpk_b_recovered[N]={0};
    poly_sub_q(sum_pk_b, b_b_recovered, lpk_b_recovered);
    
    // Step 7: verify h = SHAKE256(lpk_b' + sd')
    uint8_t h_verified[32]={0};
    EVP_MD_CTX *verify_ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(verify_ctx, EVP_shake256(), NULL);
    
    uint8_t lpk_b_recovered_bytes[COMPRESSED_POLY_SIZE];
    poly_to_compressed_bytes(lpk_b_recovered, N, lpk_b_recovered_bytes);   
    EVP_DigestUpdate(verify_ctx, lpk_b_recovered_bytes, COMPRESSED_POLY_SIZE);
    EVP_DigestUpdate(verify_ctx, decrypted_sd_prime, 32);
    EVP_DigestFinalXOF(verify_ctx, h_verified, 32);
    EVP_MD_CTX_free(verify_ctx);
    
    // Verification result
    int h_verification = memcmp(decrypted_h, h_verified, 32) == 0;
    if(h_verification!=1){
        printf("h verify: %s\n", h_verification ? "PASS" : "FAIL");
        return;
    }
    
    free(plaintext);
    free(ciphertext);
    free(decrypted);
}

void test_BXDH() {
    // srand(time(NULL));
    printf("Initializing B-XDH system...\n");
    uint64_t start, end;
    printf("Testing B-XDH with parameters (n=%d, q=%d, eta=%d)...\n", N, Q,2);
    double total_time = 0.0;  
    for (int i = 0; i < 1000; i++) {
        start = rdtsc();  
        test_dh_keygen();
        end = rdtsc();
        double ntt_time = ((double)(end - start));
        total_time += ntt_time;
    }
    double avg_cycles = total_time / 1000;                      // Average cycles
    double avg_mcycles = avg_cycles / 1e6;                      // Convert to Mcycles

    printf("Average B-XDH time: %.3f Mcycles (%.0f cycles)\n",
           avg_mcycles, avg_cycles);  
}

int main() {
    // precompute
    precompute_roots();
    // start test
    test_keygen();
    test_RLDH();
    test_BXDH();
    // test_BX3DH_optimized();
    // test_dh_keygen();
    printf("All tests passed successfully!\n");
    return 0;
}


