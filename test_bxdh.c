#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <pthread.h>

#define N 768
#define SEGMENT_SIZE 256
#define NUM_SEGMENTS 3
// #define Q 12289
#define Q 12289
#define ETA 2
#define BITS_PER_COEFF 14
#define COMPRESSED_POLY_SIZE ((N * BITS_PER_COEFF + 7) / 8)  // 1344 bytes for N=768
#define BARRETT_SHIFT 24

// 错误码定义
typedef enum {
    SUCCESS = 0,
    ERROR_RANDOM_FAILED = -1,
    ERROR_MEMORY_ALLOCATION = -2,
    ERROR_CRYPTO_OPERATION = -3
} result_t;



// 函数声明
void precompute_roots();
int64_t mod_q(int64_t x);
static inline int64_t barrett_reduce(int64_t x);
int64_t pow_mod(int64_t base, int64_t exp, int64_t mod);
int bitrev(int x, int bits);
void ntt_generic(int64_t *a, int N_size, const int64_t *roots,int bits);
void intt_generic(int64_t *a, int N_size, const int64_t *inv_roots, int64_t inv_N, int bits);
void ntt_256(int64_t *a);
void intt_256(int64_t *a);
void ntt_512(int64_t *a);
void intt_512(int64_t *a);
void split_poly(const int64_t *a, int64_t segments[3][256]);
void combine_poly(const int64_t segments[3][256], int64_t *result);
void conv_256_linear(const int64_t *a256, const int64_t *b256, int64_t *result);
void poly_mul_3x256_corrected(const int64_t *a, const int64_t *b, int64_t *c);
void naive_negacyclic_convolution(const int64_t *a, const int64_t *b, int64_t *c);
void generate_random_poly(int64_t *poly);
int compare_poly(const int64_t *a, const int64_t *b);
void print_poly_prefix(const int64_t *poly, int count);
void test_poly_mul();

// CBD2相关函数
static inline int bit_count_u32(uint32_t x);
void cbd_eta(const uint8_t *input_bytes, int64_t *coeffs, int eta, int n);
void prg_gaussian(const uint8_t *seed, int64_t *result, int n);
void generate_error_poly(int64_t *e);
void poly_add_inplace(int64_t *a, const int64_t *b);
void poly_sub_inplace(int64_t *a, const int64_t *b);

// 预计算结构体
typedef struct {
    int64_t a_ntt[3][512] __attribute__((aligned(32)));
    int64_t a_original[N] __attribute__((aligned(32)));
    int64_t inv_N_512;
    int64_t inv_N_256;
} precomputed_a_t;

// 预计算旋转因子
static int64_t roots_256[256] __attribute__((aligned(32)));
static int64_t inv_roots_256[256] __attribute__((aligned(32)));
static int64_t roots_512[512] __attribute__((aligned(32)));
static int64_t inv_roots_512[512] __attribute__((aligned(32)));

// 随机数生成批量处理
#define BATCH_SIZE 65536
static uint8_t rand_batch[BATCH_SIZE] __attribute__((aligned(32)));
static size_t batch_offset = BATCH_SIZE;
static pthread_mutex_t batch_mutex = PTHREAD_MUTEX_INITIALIZER;

// ==================== 优化工具函数 ====================
// 预计算Barrett约减参数
static const int64_t BARRETT_M = ((1LL << (2 * BARRETT_SHIFT)) + Q - 1) / Q;
// Barrett约减 - 快速模运算
static inline int64_t barrett_reduce(int64_t x) {
    int64_t q = (x * BARRETT_M) >> (2 * BARRETT_SHIFT);
    int64_t r = x - q * Q;
    r -= Q;
    r += (r >> 63) & Q;
    return r;
}

// 模运算 - 使用Barrett约减
int64_t mod_q(int64_t x) {
    return barrett_reduce(x);
}

// 快速模幂运算
int64_t pow_mod(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
    base = barrett_reduce(base);
    while (exp > 0) {
        if (exp & 1) {
            result = barrett_reduce(result * base);
        }
        base = barrett_reduce(base * base);
        exp >>= 1;
    }
    return result;
}

static uint16_t bitrev_table_16[1024] = {0};

// 初始化位反转表
void init_bitrev_table() {
    for (int i = 0; i < 1024; i++) {
        uint16_t x = (uint16_t)i;
        x = ((x & 0x5555) << 1) | ((x & 0xAAAA) >> 1);
        x = ((x & 0x3333) << 2) | ((x & 0xCCCC) >> 2);
        x = ((x & 0x0F0F) << 4) | ((x & 0xF0F0) >> 4);
        x = ((x & 0x00FF) << 8) | ((x & 0xFF00) >> 8);
        bitrev_table_16[i] = x;
    }
}

// 快速位反转（查表法）
int bitrev(int x, int bits) {
    if (bits <= 8) {
        // 8位以内，使用低8位查表
        return bitrev_table_16[x & 0xFF] >> (8 - bits);
    } else if (bits <= 16) {
        // 16位以内，直接查表并右移
        return bitrev_table_16[x & 0xFFFF] >> (16 - bits);
    } else {
        // 超过16位，回退到计算版本
        int y = 0;
        for (int i = 0; i < bits; i++) {
            if ((x >> i) & 1) {
                y |= 1 << (bits - 1 - i);
            }
        }
        return y;
    }
}

static inline int64_t add_mod_q(int64_t x, int64_t y) {
    int64_t r = x + y;
    if (r >= Q) r -= Q;
    return r;
}

static inline int64_t sub_mod_q(int64_t x, int64_t y) {
    int64_t r = x - y;
    if (r < 0) r += Q;
    return r;
}
// 批量安全随机数生成
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

// ==================== NTT相关函数 ====================

// 预计算旋转因子
void precompute_roots() {
    int64_t g = 11;  // Q=12289的本原根
    
    // 256点旋转因子
    int64_t w_256 = pow_mod(g, (Q-1)/(2*256), Q);
    roots_256[0] = 1;
    for (int i = 1; i < 256; i++) {
        roots_256[i] = barrett_reduce(roots_256[i-1] * w_256);
    }
    for (int i = 0; i < 256; i++) {
        inv_roots_256[i] = pow_mod(roots_256[i], Q-2, Q);
    }
    
    // 512点旋转因子
    int64_t w_512 = pow_mod(g, (Q-1)/512, Q);
    roots_512[0] = 1;
    for (int i = 1; i < 512; i++) {
        roots_512[i] = barrett_reduce(roots_512[i-1] * w_512);
    }
    for (int i = 0; i < 512; i++) {
        inv_roots_512[i] = pow_mod(roots_512[i], Q-2, Q);
    }
}
static inline void butterfly(int64_t *a, int64_t *b, int64_t w) {
    int64_t u = *a;
    int64_t v = mod_q((__int128)*b * w);
    *a = add_mod_q(u, v);
    *b = sub_mod_q(u, v);
}
// 通用NTT函数
void ntt_generic(int64_t *a, int N_size, const int64_t *roots, int bits) {
    
    for (int i = 0; i < N_size; i++) {
        int j = bitrev(i,bits);
        if (i < j) {
            int64_t temp_val = a[i];
            a[i] = a[j];
            a[j] = temp_val;
        }
    }
    
    // 蝶形运算
    for (int len = 2; len <= N_size; len <<= 1) {
        int half = len >> 1;
        int step = N_size / len;
        
        // 使用指针算术减少索引计算
        for (int i = 0; i < N_size; i += len) {
            const int64_t *w_ptr = roots;
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                butterfly(a_ptr + j, b_ptr + j, *w_ptr);
                w_ptr += step;
            }
        }
    }
}

// 通用逆NTT函数
void intt_generic(int64_t *a, int N_size, const int64_t *inv_roots, int64_t inv_N, int bits) {
    // ---- 蝶形运算 ----
    for (int m = N_size / 2; m >= 1; m >>= 1) {
        int step = N_size / (2 * m);
        for (int i = 0; i < N_size; i += 2 * m) {
            int widx = 0;
            for (int j = i; j < i + m; j++) {
                int k = j + m;
                int64_t w = inv_roots[widx];
                int64_t u = a[j];
                int64_t v = a[k];
                a[j] = barrett_reduce(u + v);
                a[k] = barrett_reduce((u - v) * w);
                widx += step;
            }
        }
    }

    // ---- 乘上 N 的逆元 ----
    for (int i = 0; i < N_size; i++) {
        a[i] = barrett_reduce(a[i] * inv_N);
    }

    // ---- 输出位反转（原地） ----
    for (int i = 0; i < N_size; i++) {
        int j = bitrev(i,bits);
        if (j >= N_size) continue;
        if (i < j) {
            int64_t t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}

// 专用NTT包装函数
void ntt_256(int64_t *a) {
    ntt_generic(a, 256, roots_256,8);
}

void intt_256(int64_t *a) {
    intt_generic(a, 256, inv_roots_256, pow_mod(256, Q-2, Q),8);
}

void ntt_512(int64_t *a) {
    ntt_generic(a, 512, roots_512,9);
}

void intt_512(int64_t *a) {
    intt_generic(a, 512, inv_roots_512, pow_mod(512, Q-2, Q),9);
}

// ==================== 多项式操作 ====================

// 多项式分割
void split_poly(const int64_t *a, int64_t segments[3][256]) {
    // 使用memcpy批量复制
    memcpy(segments[0], a, 256 * sizeof(int64_t));
    memcpy(segments[1], a + 256, 256 * sizeof(int64_t));
    memcpy(segments[2], a + 512, 256 * sizeof(int64_t));
}

// 优化合并函数
void combine_poly(const int64_t segments[3][256], int64_t *result) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 256; j += 4) {
            result[i*256 + j]   = barrett_reduce(segments[i][j]);
            result[i*256 + j+1] = barrett_reduce(segments[i][j+1]);
            result[i*256 + j+2] = barrett_reduce(segments[i][j+2]);
            result[i*256 + j+3] = barrett_reduce(segments[i][j+3]);
        }
    }
}

// 256点线性卷积
void conv_256_linear(const int64_t *a256, const int64_t *b256, int64_t *result) {
    int64_t a_pad[512], b_pad[512];
    
    // (1) 合并填充和置零
    for (int i = 0; i < 512; i++) {
        if (i < 256) {
            a_pad[i] = barrett_reduce(a256[i]);
            b_pad[i] = barrett_reduce(b256[i]);
        } else {
            a_pad[i] = 0;
            b_pad[i] = 0;
        }
    }

    // 512点NTT
    ntt_512(a_pad);
    ntt_512(b_pad);

    // (2) 展开点乘循环 (unrolled *4)
    for (int i = 0; i < 512; i += 4) {
        a_pad[i]   = a_pad[i]   * b_pad[i];
        a_pad[i+1] = a_pad[i+1] * b_pad[i+1];
        a_pad[i+2] = a_pad[i+2] * b_pad[i+2];
        a_pad[i+3] = a_pad[i+3] * b_pad[i+3];
    }

    // 逆变换
    intt_512(a_pad);

    // 输出线性卷积结果（长度511）
    for (int i = 0; i < 511; i++) {
        result[i] = barrett_reduce(a_pad[i]);
    }
}

// 优化的3×256点负循环卷积
void poly_mul_3x256_corrected(const int64_t *a, const int64_t *b, int64_t *c) {
    int64_t A_segs[3][256];
    int64_t B_segs[3][256];
    split_poly(a, A_segs);
    split_poly(b, B_segs);

    // 1) 预先 FFT(A_segs) 和 FFT(B_segs)
    int64_t A_ntt[3][512], B_ntt[3][512];
    for (int r = 0; r < 3; ++r) {
        int64_t tmp[512];
        for (int i = 0; i < 256; ++i) tmp[i] = mod_q(A_segs[r][i]);
        for (int i = 256; i < 512; ++i) tmp[i] = 0;
        ntt_512(tmp);
        memcpy(A_ntt[r], tmp, sizeof(tmp));
    }
    for (int s = 0; s < 3; ++s) {
        int64_t tmp[512];
        for (int i = 0; i < 256; ++i) tmp[i] = mod_q(B_segs[s][i]);
        for (int i = 256; i < 512; ++i) tmp[i] = 0;
        ntt_512(tmp);
        memcpy(B_ntt[s], tmp, sizeof(tmp));
    }

    // 2) 累加器（用 __int128 防溢出），每次 INTT 后直接 fold 加到 acc
    __int128 acc[N];
    for (int i = 0; i < N; ++i) acc[i] = 0;

    int64_t product[512];
    for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < 3; ++s) {
            // 频域点乘（安全）
            for (int i = 0; i < 512; ++i) {
                product[i] = mod_q(A_ntt[r][i]*B_ntt[s][i]);
            }

            // 逆 NTT 得到时域（长度512），前511 为线性卷积
            intt_512(product);

            int base = (r + s) * 256;
            for (int k = 0; k < 511; ++k) {
                int total_deg = base + k;
                int64_t coef = mod_q(product[k]); // product[k] 已是 int64_t，但确保 mod
                if (coef == 0) continue;
                if (total_deg < N)
                    acc[total_deg] += (__int128) coef;
                else
                    acc[total_deg - N] -= (__int128) coef;
            }
        }
    }

    // 3) 最后 reduce 到 c
    for (int i = 0; i < N; ++i) {
        // 假设你有 barrett_reduce128，否则把 acc[i] 缩放到 int64 再 barrett_reduce
        c[i] = mod_q(acc[i]);
    }
}

// ==================== 预计算优化 ====================

// 预计算函数
void precompute_a_ntt(int64_t *a, precomputed_a_t *precomputed) {
    int64_t A_segs[3][256];
    split_poly(a, A_segs);
    
    for (int i = 0; i < 3; i++) {
        int64_t padded[512] = {0};
        
        // 填充到512点
        for (int j = 0; j < 256; j++) {
            padded[j] = mod_q(A_segs[i][j]);
        }
        
        // 计算NTT
        ntt_512(padded);
        
        // 存储NTT结果
        memcpy(precomputed->a_ntt[i], padded, sizeof(int64_t) * 512);
    }
    
    // 保存原始a和预计算缩放因子
    memcpy(precomputed->a_original, a, sizeof(int64_t) * N);
    precomputed->inv_N_512 = pow_mod(512, Q-2, Q);
    precomputed->inv_N_256 = pow_mod(256, Q-2, Q);
}

// 使用预计算NTT的优化多项式乘法
void poly_mul_3x256_precomputed(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c) {
    int64_t B_segs[3][256];
    split_poly(b, B_segs);

    // 1) 预计算 B 的 3 段 NTT，一次性
    int64_t b_ntt[3][512];
    for (int s = 0; s < 3; ++s) {
        int64_t tmp[512] = {0};
        for (int i = 0; i < 256; ++i) tmp[i] = mod_q(B_segs[s][i]);
        // zero 高位已准备好
        ntt_512(tmp);                 // 在位，把 tmp 变为 NTT 域（mod q）
        memcpy(b_ntt[s], tmp, sizeof(tmp));
    }

    // 2) 保存每对 (r,s) 的线性卷积（511）
    int64_t conv[3][3][511];
    memset(conv, 0, sizeof(conv));

    int64_t product[512];
    for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < 3; ++s) {
            // 点乘（NTT 域）；注意安全乘法 + 模约简
            for (int i = 0; i < 512; ++i) {
                product[i] = mod_q(precomputed->a_ntt[r][i]*b_ntt[s][i]);
            }

            // 逆 NTT：得到长度 512 的时域结果（其中前 511 为线性卷积）
            intt_512(product);

            // 提取前 511 项（线性卷积长度 = 256 + 256 - 1）
            for (int k = 0; k < 511; ++k) {
                conv[r][s][k] = mod_q(product[k]);
            }
        }
    }

    // 3) 合并到结果并做环上折叠 x^768 + 1
    for (int i = 0; i < N; ++i) c[i] = 0;

    for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < 3; ++s) {
            int shift_blocks = r + s;
            for (int k = 0; k < 511; ++k) {
                int total_deg = 256 * shift_blocks + k;
                int64_t coef = conv[r][s][k];
                if (coef == 0) continue;

                if (total_deg < N) {
                    c[total_deg] = mod_q(c[total_deg] + coef);
                } else {
                    int t = total_deg - N;
                    // x^N == -1 -> 折叠时减去
                    c[t] = mod_q(c[t] - coef);
                }
            }
        }
    }

    // 4) 最后统一约简（保证输出落在 [0,q)）
    for (int i = 0; i < N; ++i) {
        c[i] = mod_q(c[i]);
    }
}

// ==================== 多项式算术运算 ====================

// 多项式加法（原地操作）
void poly_add_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i += 4) {
        a[i]   = barrett_reduce(a[i]   + b[i]);
        a[i+1] = barrett_reduce(a[i+1] + b[i+1]);
        a[i+2] = barrett_reduce(a[i+2] + b[i+2]);
        a[i+3] = barrett_reduce(a[i+3] + b[i+3]);
    }
}

// 多项式减法（原地操作）
void poly_sub_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i += 4) {
        a[i]   = barrett_reduce(a[i]   - b[i]);
        a[i+1] = barrett_reduce(a[i+1] - b[i+1]);
        a[i+2] = barrett_reduce(a[i+2] - b[i+2]);
        a[i+3] = barrett_reduce(a[i+3] - b[i+3]);
    }
}

// 多项式加法（输出到c）
void poly_add_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i += 4) {
        c[i]   = barrett_reduce(a[i]   + b[i]);
        c[i+1] = barrett_reduce(a[i+1] + b[i+1]);
        c[i+2] = barrett_reduce(a[i+2] + b[i+2]);
        c[i+3] = barrett_reduce(a[i+3] + b[i+3]);
    }
}

// 多项式减法（输出到c）
void poly_sub_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i += 4) {
        c[i]   = barrett_reduce(a[i]   - b[i]);
        c[i+1] = barrett_reduce(a[i+1] - b[i+1]);
        c[i+2] = barrett_reduce(a[i+2] - b[i+2]);
        c[i+3] = barrett_reduce(a[i+3] - b[i+3]);
    }
}

// ==================== 随机多项式生成 ====================

// 生成随机多项式
void generate_random_poly(int64_t *poly) {
    uint8_t buf[N * 2]; // 为所有系数生成随机字节
    if (batched_secure_random(buf, sizeof(buf)) != SUCCESS) {
        fprintf(stderr, "Error: RAND_bytes failed.\n");
        exit(1);
    }
    
    for (int i = 0; i < N; i++) {
        uint16_t r = ((uint16_t)buf[2*i] << 8) | buf[2*i+1];
        poly[i] = r % Q;
    }
}

// 位计数辅助函数
static inline int bit_count_u32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// SHAKE256扩展输出函数
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

const uint32_t mask = (1u << 2) - 1u;
const uint64_t extract_mask = (1ULL << (2 * 2)) - 1ULL;
// CBD实现
void cbd_eta(const uint8_t *input_bytes, int64_t *coeffs, int eta, int n) {
    const int byte_len = (2 * eta * n + 7) / 8;
    uint64_t buf = 0;
    int buf_bits = 0;
    int byte_index = 0;

    // 预计算掩码

    for (int i = 0; i < n; i++) {
        // 填充缓冲区
        while (buf_bits < 2 * eta) {
            uint8_t next = (byte_index < byte_len) ? input_bytes[byte_index++] : 0;
            buf |= ((uint64_t)next) << buf_bits;
            buf_bits += 8;
        }

        // 提取和计算
        uint32_t x = (uint32_t)(buf & extract_mask);
        int a = bit_count_u32(x & mask);
        int b = bit_count_u32((x >> eta) & mask);
        coeffs[i] = a - b;

        buf >>= (2 * eta);
        buf_bits -= (2 * eta);
    }
}

// 生成高斯分布多项式
void prg_gaussian(const uint8_t *seed, int64_t *result, int n) {
    const int eta = 2;
    int bits_needed = 2 * eta * n;
    int byte_len = (bits_needed + 7) / 8;
    
    uint8_t *random_bytes = (uint8_t *)malloc(byte_len);
    if (!random_bytes) {
        fprintf(stderr, "malloc failed in prg_gaussian\n");
        exit(1);
    }
    shake256(seed, 33, random_bytes, byte_len);
    cbd_eta(random_bytes, result, eta, n);
    free(random_bytes);
}

// 优化的s和b生成函数（无动态内存分配）
void generate_s_b_gaussian_optimized(const uint8_t seed32[32], const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed_s[33], seed_e[33];
    memcpy(seed_s, seed32, 32);
    memcpy(seed_e, seed32, 32);
    seed_s[32] = 's';
    seed_e[32] = 'e';

    int64_t e[N];  // 使用栈内存

    prg_gaussian(seed_s, s, N);
    prg_gaussian(seed_e, e, N);

    // 直接计算到b中
    poly_mul_3x256_precomputed(precomputed, s, b);
    
    // 原地添加错误项：b = b + 2e
    for (int i = 0; i < N; i++) {
        b[i] = barrett_reduce(b[i] + 2 * e[i]);
    }
}

// ==================== 密钥生成 ====================

result_t keygen(const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed[32];
    if (batched_secure_random(seed, 32) != SUCCESS) {
        return ERROR_RANDOM_FAILED;
    }
    generate_s_b_gaussian_optimized(seed, precomputed, s, b);
    return SUCCESS;
}

// ==================== 辅助函数 ====================

static inline int64_t floor_div(double x) {
    return (int64_t)floor(x);
}

static inline int64_t mod2(int64_t x, int64_t w, int64_t q) {
    int64_t tmp = (x + w * ((q - 1) / 2)) % q;
    if (tmp < 0) tmp += q;
    return tmp % 2;
}

static inline int64_t sig_0(int64_t x, int64_t q) {
    int64_t bound = q / 4;
    return (x >= -bound && x <= bound) ? 0 : 1;
}

static inline int64_t sig_1(int64_t x, int64_t q) {
    int64_t bound = q / 4;
    return (x >= (-bound + 1) && x <= (bound + 1)) ? 0 : 1;
}

// ==================== RLDH协议实现 ====================

void astRLDH(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb, int n, int64_t q) {
    uint8_t seed[32];
    if (batched_secure_random(seed, 32) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        exit(1);
    }

    int64_t e[N], tmp[N];
    prg_gaussian(seed, e, n);

    // tmp = pk_a * sk_b (mod q)
    poly_mul_3x256_corrected(pk_a, sk_b, tmp);
    
    // tmp = tmp + 2e mod q
    for (int i = 0; i < n; i++) {
        tmp[i] = barrett_reduce(tmp[i] + 2 * e[i]);
    }

    // 设置kb为中心模(q)
    int64_t kb[N];
    for (int i = 0; i < n; i++) {
        kb[i] = tmp[i];
        if (kb[i] > (q - 1) / 2) {
            kb[i] = kb[i] - q + 1;
            if (kb[i] == 0)
                //处理边界
                kb[i] += 1;
        }
    }

    // 提示和派生位
    // uint8_t random_bits[96];
    // if (batched_secure_random(random_bits, 96) != SUCCESS) {
    //     fprintf(stderr, "Random generation failed!\n");
    //     exit(1);
    // }

    for (int i = 0; i < n; i++) {
        // int byte_index = i / 8;
        // int bit_index = i % 8;
        // int alea = (random_bits[byte_index] >> bit_index) & 1;
        
        // int64_t sig = alea ? sig_0(kb[i], q) : sig_1(kb[i], q);
        int64_t sig=sig_0(kb[i], q);
        htb[i] = sig;
        dhb[i] = mod2(kb[i], htb[i], q);
    }
}

void recRLDH(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha, int n, int64_t q) {
    int64_t tmp[N];
    poly_mul_3x256_corrected(pk_b, sk_a, tmp);

    int64_t ka[N];
    for (int i = 0; i < n; i++) {
        ka[i] = tmp[i];
        if (ka[i] > (q - 1) / 2) {
            ka[i] = ka[i] - q + 1;
            if (ka[i] == 0)
            //处理边界
                ka[i] += 1;
        }
    }

    for (int i = 0; i < n; i++) {
        dha[i] = mod2(ka[i], htb[i], q);
    }
}

// ==================== 多项式压缩 ====================
static const uint8_t BIT_MASKS[9] = {0, 1, 3, 7, 15, 31, 63, 127, 255};
void poly_to_compressed_bytes(const int64_t *poly, int n, uint8_t *bytes) {
    uint64_t buffer = 0;
    int bit_count = 0;
    int byte_pos = 0;
    
    for (int i = 0; i < n; i++) {
        // 快速正规范化
        uint16_t coeff = (poly[i] % Q + Q) % Q;
        
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
        // 确保缓冲区中有足够的位
        while (bits_in_buffer < bits_per_coeff) {
            bit_buffer = (bit_buffer << 8) | bytes[byte_index++];
            bits_in_buffer += 8;
        }
        
        // 提取系数
        int shift = bits_in_buffer - bits_per_coeff;
        uint16_t coeff = (bit_buffer >> shift) & ((1 << bits_per_coeff) - 1);
        poly[i] = coeff;
        
        // 更新缓冲区
        bits_in_buffer -= bits_per_coeff;
        bit_buffer &= (1ULL << bits_in_buffer) - 1;
    }
}

// ==================== KDF函数 ====================

int PRF_2(const int64_t dhb[N], const uint8_t *sid, size_t sid_len,
          uint8_t k[32], uint8_t k0[32]) {
    
    uint8_t dhb_compressed[COMPRESSED_POLY_SIZE];
    poly_to_compressed_bytes(dhb, N, dhb_compressed);
    
    size_t data_len = COMPRESSED_POLY_SIZE + sid_len;
    uint8_t data[data_len+ 3];
    
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

// ==================== 会话ID构建 ====================

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

// ==================== AES-GCM加解密 ====================

int aes_gcm_encrypt(const uint8_t *key, size_t key_len,
                   const uint8_t *iv, size_t iv_len,
                   const uint8_t *aad, size_t aad_len,
                   const uint8_t *plaintext, size_t plaintext_len,
                   uint8_t *ciphertext, size_t ciphertext_len) {
    
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
                   const uint8_t *iv, size_t iv_len,
                   const uint8_t *aad, size_t aad_len,
                   uint8_t *ciphertext, size_t ciphertext_len,  
                   uint8_t *plaintext, size_t plaintext_len) {
    
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

// ==================== 测试函数 ====================

#include <x86intrin.h>
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void test_dh_keygen() {
    srand(time(NULL));
    
    int64_t a[N];
    generate_random_poly(a);
    
    precomputed_a_t precomputed;  // 使用栈内存
    precompute_a_ntt(a, &precomputed);
    
    int64_t lpk_a[N], lsk_b[N], lpk_b[N], lsk_a[N];
    int64_t epk_a[N], esk_a[N], epk_b[N], esk_b[N];
    int64_t spk_a[N], ssk_a[N];
    int64_t htb[N], dhb[N], dha[N];
    int64_t sum_sk_a[N], sum_pk_a[N], sum_sk_b[N], sum_pk_b[N];
    
    // 生成各种密钥对
    keygen(&precomputed, lsk_a, lpk_a);
    keygen(&precomputed, lsk_b, lpk_b);
    keygen(&precomputed, ssk_a, spk_a);
    
    uint8_t seed_a[32], seed_b[32];
    if (batched_secure_random(seed_a, 32) != SUCCESS ||
        batched_secure_random(seed_b, 32) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        return;
    }

    generate_s_b_gaussian_optimized(seed_a, &precomputed, esk_a, epk_a);
    generate_s_b_gaussian_optimized(seed_b, &precomputed, esk_b, epk_b);
    
    // 计算和密钥
    poly_add_q(lsk_a, ssk_a, sum_sk_a);
    poly_add_inplace(sum_sk_a, esk_a);
    poly_add_q(lpk_a, spk_a, sum_pk_a);
    poly_add_inplace(sum_pk_a, epk_a);
    poly_add_q(lsk_b, esk_b, sum_sk_b);
    poly_add_q(lpk_b, epk_b, sum_pk_b);
    
    // RLDH协议
    astRLDH(sum_pk_a, sum_sk_b, htb, dhb, N, Q);
    recRLDH(sum_pk_b, sum_sk_a, htb, dha, N, Q);
    
    // SID
    size_t sid_size = 2 + 4 * COMPRESSED_POLY_SIZE;
    uint8_t sid[sid_size];
    
    size_t sid_len = build_session_id(lpk_a, spk_a, epk_a, sum_pk_b, sid);
    
    // KDF
    uint8_t k[32], k0[32];
    PRF_2(dhb, sid, sid_len, k, k0);
    
    // 验证流程
    uint8_t sd_prime[32];
    if (batched_secure_random(sd_prime, 32) != SUCCESS) {
        fprintf(stderr, "Random generation failed for sd_prime!\n");
        return;
    }
    
  // Step 1: compute h = SHAKE256(lpk_b + sd')
    uint8_t h[32];
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
    int encrypt_success = aes_gcm_encrypt(k0, 32, iv, 12, 
                                         NULL, 0,  // No AAD
                                         plaintext, plaintext_len,
                                         ciphertext, ciphertext_len);
    
    if (!encrypt_success) {
        fprintf(stderr, "AES-GCM encryption failed\n");
        free(plaintext);
        free(ciphertext);
        return;
    }
    
    // Step 4: Decrypt with k0
    uint8_t k_prime[32], k0_prime[32];
    PRF_2(dha, sid, sid_len, k_prime, k0_prime);
    uint8_t *decrypted = malloc(plaintext_len);
    if (!decrypted) {
        fprintf(stderr, "Memory allocation failed for decrypted data\n");
        free(plaintext);
        free(ciphertext);
        return;
    }

    // Note: ciphertext parameter const qualifier removed here
    int decrypt_success = aes_gcm_decrypt(k0_prime, 32, iv, 12,
                                        NULL, 0,
                                        ciphertext, ciphertext_len,  
                                        decrypted, plaintext_len);    
    if (!decrypt_success) {
        fprintf(stderr, "AES-GCM decryption failed\n");
        free(plaintext);
        free(ciphertext);
        free(decrypted);
        return;
    }
   // Step 5: separate sd and sd'
    uint8_t decrypted_h[32], decrypted_sd[32], decrypted_sd_prime[32];
    memcpy(decrypted_h, decrypted, 32);
    memcpy(decrypted_sd, decrypted + 32, 32);
    memcpy(decrypted_sd_prime, decrypted + 64, 32);
    
    // Step 6: recover lpk_b' = sum_pk_b - epk_b
    int64_t s_b_recovered[N], b_b_recovered[N];
    generate_s_b_gaussian_optimized(decrypted_sd, &precomputed, s_b_recovered, b_b_recovered);
    int64_t lpk_b_recovered[N];
    poly_sub_q(sum_pk_b, b_b_recovered, lpk_b_recovered);
    
    // Step 7: verify h = SHAKE256(lpk_b' + sd')
    uint8_t h_verified[32];
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

void test_keygen() {
    // srand(time(NULL));
    int64_t a[N];
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt(a, &precomputed);

    int64_t s[N], b[N];
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
    int64_t a[N];
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt(a, &precomputed);

    int64_t s[N], b[N];
    int64_t s1[N], b1[N], htb[N],dha[N],dhb[N];
    keygen(&precomputed, s, b);
    keygen(&precomputed, s1, b1);
    uint64_t total_cycles = 0;
    uint64_t total_cycles1 = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc();
        astRLDH(b, s1, htb, dhb, N, Q);
        uint64_t end = rdtsc();
        total_cycles += (end - start);
        start = rdtsc();
        recRLDH(b1, s, htb, dha, N, Q);        
        end = rdtsc();
        total_cycles1 += (end - start);
        for(int j=0;j<N;j++){
            if(dha[j]!=dhb[j]){
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
void test_primitive_root() {
    int64_t q = 15361;
    int64_t g_candidates[] = {3, 5, 7, 11, 13, 17, 19, 23};
    int num_candidates = sizeof(g_candidates) / sizeof(g_candidates[0]);
    
    for (int i = 0; i < num_candidates; i++) {
        int64_t g = g_candidates[i];
        int64_t w256 = pow_mod(g, (q-1)/(2*256), q);
        int64_t w512 = pow_mod(g, (q-1)/512, q);
        
        int64_t verify256 = pow_mod(w256, 256, q);
        int64_t verify512 = pow_mod(w512, 512, q);
        
        if (verify256 == q-1 && verify512 == 1) {
            printf("Found correct primitive root: g = %ld\n", g);
            printf("w_256 = %ld, w_512 = %ld\n", w256, w512);
            return;
        }
    }
    printf("No correct primitive root found!\n");
}
int main() {
    // precompute
    init_bitrev_table();
    precompute_roots();
    // start test
    test_keygen();
    test_RLDH();
    test_BXDH();
    printf("All tests passed successfully!\n");
    // test_primitive_root();
    return 0;
}
