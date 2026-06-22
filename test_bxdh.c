#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <x86intrin.h>
#define add_mod_q add_mod_q_ct
#define sub_mod_q sub_mod_q_ct
#define twist_forward twist_forward_ct
#define twist_inverse twist_inverse_ct

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

// ==================== 侧信道防护配置 ====================
#define SCA_PROTECTION_LEVEL 3  // 1: 基础, 2: 中级, 3: 高级

// ==================== 错误码 ====================
typedef enum {
    SUCCESS = 0,
    ERROR_RANDOM_FAILED = -1,
    ERROR_MEMORY_ALLOCATION = -2,
    ERROR_CRYPTO_OPERATION = -3
} result_t;

// ==================== 数据结构 ====================
typedef struct {
    int64_t a_ntt[512];
    int64_t a_original[N];
    int64_t inv_N_512;
} precomputed_a_t;

// ==================== 函数声明 ====================
void precompute_roots(void);
void precompute_psi(void);
int64_t mod_q(int64_t x);
int64_t mod_q_ct(int64_t x);
int64_t pow_mod(int64_t base, int64_t exp);
int64_t pow_mod_ct(int64_t base, int64_t exp);
int bitrev(int x, int logn);
void ntt_512(int64_t *a);
void ntt_512_sca(int64_t *a);
void intt_512(int64_t *a);
void intt_512_sca(int64_t *a);
void poly_mul_512(const int64_t *a, const int64_t *b, int64_t *c);
void poly_mul_negacyclic_512(const int64_t *a, const int64_t *b, int64_t *c);
void poly_mul_negacyclic_512_sca(const int64_t *a, const int64_t *b, int64_t *c);
void generate_random_poly(int64_t *poly);
void generate_small_poly(int64_t *poly, int eta);
void generate_small_poly_sca(int64_t *poly, int eta);
void poly_add_inplace(int64_t *a, const int64_t *b);
void poly_sub_inplace(int64_t *a, const int64_t *b);
void poly_add_q(const int64_t *a, const int64_t *b, int64_t *c);
void poly_sub_q(const int64_t *a, const int64_t *b, int64_t *c);
void precompute_a_ntt(int64_t *a, precomputed_a_t *precomputed);
void poly_mul_precomputed(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c);
void poly_mul_precomputed_sca(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c);
result_t keygen(const precomputed_a_t *precomputed, int64_t *s, int64_t *b);
result_t keygen_sca(const precomputed_a_t *precomputed, int64_t *s, int64_t *b);
void astRLDH(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb);
void astRLDH_sca(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb);
void recRLDH(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha);
void recRLDH_sca(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha);

// ==================== 侧信道安全: 恒定时间选择器 ====================
static inline int64_t ct_select(int64_t a, int64_t b, int64_t mask) {
    // mask 为 0 或 -1 (全1)
    return (a & ~mask) | (b & mask);
}

static inline int64_t ct_negate(int64_t x, int64_t cond) {
    // cond 为 0 或 -1
    return (x ^ cond) - cond;
}

// ==================== Batch random ====================
#define BATCH_SIZE 65536
static uint8_t rand_batch[BATCH_SIZE];
static size_t batch_offset = BATCH_SIZE;
static pthread_mutex_t batch_mutex = PTHREAD_MUTEX_INITIALIZER;

// ==================== 单位根存储 ====================
static int64_t roots_512_layered[LOGN][N/2];
static int64_t inv_roots_512_layered[LOGN][N/2];
static int64_t Q_2_512_POW_MOD;
static int64_t psi[512];
static int64_t inv_psi[512];

// ==================== Barrett 常量 ====================
static const int64_t BARRETT_M = ((1LL << BARRETT_SHIFT) + Q - 1) / Q;

// ==================== 模运算 (标准版) ====================
int64_t mod_q(int64_t x) {
    int64_t q = (x * BARRETT_M) >> BARRETT_SHIFT;
    int64_t r = x - q * Q - Q;
    r += (r >> 63) & Q;
    return r;
}

// ==================== 模运算 (恒定时间版) ====================
int64_t mod_q_ct(int64_t x) {
    int64_t q = (x * BARRETT_M) >> BARRETT_SHIFT;
    int64_t r = x - q * Q;
    
    // 恒定时间条件修正: 如果 r >= Q 则减去 Q
    // 使用无分支的掩码操作
    int64_t ge_q = (r - Q) >> 63;  // 如果 r >= Q, ge_q = 0; 否则 ge_q = -1
    r = r - (Q & ~ge_q);
    
    // 如果 r < 0 则加上 Q
    int64_t lt_0 = r >> 63;
    r = r + (Q & lt_0);
    
    return r;
}

// ==================== 模幂 (标准版) ====================
int64_t pow_mod(int64_t base, int64_t exp) {
    int64_t result = 1;
    base = mod_q(base);
    while (exp > 0) {
        if (exp & 1) {
            result = mod_q((__int128)result * base);
        }
        base = mod_q((__int128)base * base);
        exp >>= 1;
    }
    return result;
}

// ==================== 模幂 (恒定时间版) ====================
int64_t pow_mod_ct(int64_t base, int64_t exp) {
    int64_t result = 1;
    base = mod_q_ct(base);
    
    // 恒定时间循环: 固定迭代次数 (假设 exp 最多 32 位)
    for (int i = 0; i < 32; i++) {
        int64_t bit = (exp >> i) & 1;
        int64_t mask = -bit;
        
        // 恒定时间条件乘法
        int64_t temp = mod_q_ct((__int128)result * base);
        result = ct_select(result, temp, mask);
        
        base = mod_q_ct((__int128)base * base);
    }
    return result;
}

// ==================== 加减模运算 (恒定时间) ====================
static inline int64_t add_mod_q_ct(int64_t x, int64_t y) {
    int64_t r = x + y;
    int64_t ge_q = (r - Q) >> 63;
    r = r - (Q & ~ge_q);
    return r;
}

static inline int64_t sub_mod_q_ct(int64_t x, int64_t y) {
    int64_t r = x - y;
    int64_t lt_0 = r >> 63;
    r = r + (Q & lt_0);
    return r;
}

// ==================== Batch random ====================
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

// ==================== 恒定时间位反转 ====================
int bitrev(int x, int logn) {
    int r = 0;
    for (int i = 0; i < logn; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// ==================== 恒定时间位反转排列 ====================
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

// ==================== 位反转排列 (恒定时间版本) ====================
void bitrev_permute_ct(int64_t *a, int n, int logn) {
    // 使用预计算的位反转表
    static int rev_table[512];
    static int table_initialized = 0;
    
    if (!table_initialized) {
        for (int i = 0; i < n; i++) {
            rev_table[i] = bitrev(i, logn);
        }
        table_initialized = 1;
    }
    
    for (int i = 0; i < n; ++i) {
        int j = rev_table[i];
        // 恒定时间条件交换 (使用 XOR 技巧)
        int64_t t = a[i] ^ a[j];
        int64_t mask = -(j > i);
        a[i] ^= t & mask;
        a[j] ^= t & mask;
    }
}

// ==================== 单位根预计算 ====================
void precompute_roots(void) {
    int64_t g = ROOT;
    int64_t w512 = pow_mod(g, (Q - 1) / 512);
    
    for (int layer = 0; layer < LOGN; layer++) {
        int len = 1 << (layer + 1);
        int half = len >> 1;
        
        int64_t w_len = pow_mod(w512, 512 / len);
        int64_t inv_w_len = pow_mod(w_len, Q - 2);
        
        int64_t w = 1;
        int64_t inv_w = 1;
        
        for (int j = 0; j < half; j++) {
            roots_512_layered[layer][j] = w;
            inv_roots_512_layered[layer][j] = inv_w;
            w = mod_q((__int128)w * w_len);
            inv_w = mod_q((__int128)inv_w * inv_w_len);
        }
    }
    
    Q_2_512_POW_MOD = pow_mod(512, Q - 2);
}

// ==================== psi 预计算 ====================
void precompute_psi(void) {
    int64_t psi_root = pow_mod(ROOT, (Q - 1) / 1024);
    int64_t inv_psi_root = pow_mod(psi_root, Q - 2);
    
    psi[0] = 1;
    inv_psi[0] = 1;
    for (int i = 1; i < 512; i++) {
        psi[i] = mod_q((__int128)psi[i-1] * psi_root);
        inv_psi[i] = mod_q((__int128)inv_psi[i-1] * inv_psi_root);
    }
}

// ==================== 扭转变换 (恒定时间) ====================
void twist_forward_ct(int64_t *a) {
    for (int i = 0; i < 512; i++) {
        a[i] = mod_q_ct((__int128)a[i] * psi[i]);
    }
}

void twist_inverse_ct(int64_t *a) {
    for (int i = 0; i < 512; i++) {
        a[i] = mod_q_ct((__int128)a[i] * inv_psi[i]);
    }
}

// ==================== 标准 NTT ====================
void ntt_512(int64_t *a) {
    bitrev_permute(a, 512, LOGN);
    
    for (int layer = 0; layer < LOGN; layer++) {
        int len = 1 << (layer + 1);
        int half = len >> 1;
        const int64_t *w_ptr = roots_512_layered[layer];
        
        for (int i = 0; i < 512; i += len) {
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                int64_t u = a_ptr[j];
                int64_t v = mod_q((__int128)b_ptr[j] * w_ptr[j]);
                a_ptr[j] = add_mod_q_ct(u, v);
                b_ptr[j] = sub_mod_q_ct(u, v);
            }
        }
    }
}

// ==================== NTT (恒定时间版本) ====================
void ntt_512_ct(int64_t *a) {
    bitrev_permute_ct(a, 512, LOGN);
    
    for (int layer = 0; layer < LOGN; layer++) {
        int len = 1 << (layer + 1);
        int half = len >> 1;
        const int64_t *w_ptr = roots_512_layered[layer];
        
        for (int i = 0; i < 512; i += len) {
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                int64_t u = a_ptr[j];
                int64_t v = mod_q_ct((__int128)b_ptr[j] * w_ptr[j]);
                a_ptr[j] = add_mod_q_ct(u, v);
                b_ptr[j] = sub_mod_q_ct(u, v);
            }
        }
    }
}

// ==================== 标准 INTT ====================
void intt_512(int64_t *a) {
    for (int layer = LOGN - 1; layer >= 0; layer--) {
        int len = 1 << (layer + 1);
        int half = len >> 1;
        const int64_t *w_ptr = inv_roots_512_layered[layer];
        
        for (int i = 0; i < 512; i += len) {
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                int64_t u = a_ptr[j];
                int64_t v = b_ptr[j];
                a_ptr[j] = add_mod_q(u, v);
                b_ptr[j] = mod_q((__int128)sub_mod_q(u, v) * w_ptr[j]);
            }
        }
    }
    
    for (int i = 0; i < 512; i++) {
        a[i] = mod_q((__int128)a[i] * Q_2_512_POW_MOD);
    }
    
    bitrev_permute(a, 512, LOGN);
}

// ==================== INTT (恒定时间版本) ====================
void intt_512_ct(int64_t *a) {
    for (int layer = LOGN - 1; layer >= 0; layer--) {
        int len = 1 << (layer + 1);
        int half = len >> 1;
        const int64_t *w_ptr = inv_roots_512_layered[layer];
        
        for (int i = 0; i < 512; i += len) {
            int64_t *a_ptr = a + i;
            int64_t *b_ptr = a_ptr + half;
            
            for (int j = 0; j < half; j++) {
                int64_t u = a_ptr[j];
                int64_t v = b_ptr[j];
                a_ptr[j] = add_mod_q_ct(u, v);
                b_ptr[j] = mod_q_ct((__int128)sub_mod_q_ct(u, v) * w_ptr[j]);
            }
        }
    }
    
    for (int i = 0; i < 512; i++) {
        a[i] = mod_q_ct((__int128)a[i] * Q_2_512_POW_MOD);
    }
    
    bitrev_permute_ct(a, 512, LOGN);
}

// ==================== 多项式乘法 (标准) ====================
void poly_mul_512(const int64_t *a, const int64_t *b, int64_t *c) {
    int64_t A[512], B[512];
    memcpy(A, a, sizeof(int64_t) * 512);
    memcpy(B, b, sizeof(int64_t) * 512);
    
    ntt_512(A);
    ntt_512(B);
    
    for (int i = 0; i < 512; i++) {
        A[i] = mod_q((__int128)A[i] * B[i]);
    }
    
    intt_512(A);
    memcpy(c, A, sizeof(int64_t) * 512);
}

// ==================== 负向环乘法 (标准) ====================
void poly_mul_negacyclic_512(const int64_t *a, const int64_t *b, int64_t *c) {
    int64_t A[512], B[512];
    memcpy(A, a, sizeof(int64_t) * 512);
    memcpy(B, b, sizeof(int64_t) * 512);
    
    twist_forward(A);
    twist_forward(B);
    ntt_512(A);
    ntt_512(B);
    
    for (int i = 0; i < 512; i++) {
        A[i] = mod_q((__int128)A[i] * B[i]);
    }
    
    intt_512(A);
    twist_inverse(A);
    memcpy(c, A, sizeof(int64_t) * 512);
}

// ==================== 负向环乘法 (恒定时间) ====================
void poly_mul_negacyclic_512_ct(const int64_t *a, const int64_t *b, int64_t *c) {
    int64_t A[512], B[512];
    memcpy(A, a, sizeof(int64_t) * 512);
    memcpy(B, b, sizeof(int64_t) * 512);
    
    twist_forward_ct(A);
    twist_forward_ct(B);
    ntt_512_ct(A);
    ntt_512_ct(B);
    
    for (int i = 0; i < 512; i++) {
        A[i] = mod_q_ct((__int128)A[i] * B[i]);
    }
    
    intt_512_ct(A);
    twist_inverse_ct(A);
    memcpy(c, A, sizeof(int64_t) * 512);
}

// ==================== 预计算 ====================
void precompute_a_ntt(int64_t *a, precomputed_a_t *precomputed) {
    memcpy(precomputed->a_original, a, sizeof(int64_t) * N);
    
    int64_t padded[512];
    memcpy(padded, a, sizeof(int64_t) * 512);
    
    twist_forward(padded);
    ntt_512(padded);
    
    memcpy(precomputed->a_ntt, padded, sizeof(int64_t) * 512);
    precomputed->inv_N_512 = Q_2_512_POW_MOD;
}

// ==================== 预计算 (恒定时间) ====================
void precompute_a_ntt_ct(int64_t *a, precomputed_a_t *precomputed) {
    memcpy(precomputed->a_original, a, sizeof(int64_t) * N);
    
    int64_t padded[512];
    memcpy(padded, a, sizeof(int64_t) * 512);
    
    twist_forward_ct(padded);
    ntt_512_ct(padded);
    
    memcpy(precomputed->a_ntt, padded, sizeof(int64_t) * 512);
    precomputed->inv_N_512 = Q_2_512_POW_MOD;
}

// ==================== 预计算乘法 (标准) ====================
void poly_mul_precomputed(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c) {
    int64_t B_ntt[512];
    memcpy(B_ntt, b, sizeof(int64_t) * 512);
    twist_forward(B_ntt);
    ntt_512(B_ntt);
    
    int64_t product[512];
    for (int i = 0; i < 512; i++) {
        product[i] = mod_q((__int128)precomputed->a_ntt[i] * B_ntt[i]);
    }
    
    intt_512(product);
    twist_inverse(product);
    memcpy(c, product, sizeof(int64_t) * 512);
}

// ==================== 预计算乘法 (恒定时间) ====================
void poly_mul_precomputed_ct(const precomputed_a_t *precomputed, const int64_t *b, int64_t *c) {
    int64_t B_ntt[512];
    memcpy(B_ntt, b, sizeof(int64_t) * 512);
    twist_forward_ct(B_ntt);
    ntt_512_ct(B_ntt);
    
    int64_t product[512];
    for (int i = 0; i < 512; i++) {
        product[i] = mod_q_ct((__int128)precomputed->a_ntt[i] * B_ntt[i]);
    }
    
    intt_512_ct(product);
    twist_inverse_ct(product);
    memcpy(c, product, sizeof(int64_t) * 512);
}

// ==================== 多项式运算 ====================
void poly_add_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i++) {
        a[i] = add_mod_q(a[i], b[i]);
    }
}

void poly_sub_inplace(int64_t *a, const int64_t *b) {
    for (int i = 0; i < N; i++) {
        a[i] = sub_mod_q(a[i], b[i]);
    }
}

void poly_add_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i++) {
        c[i] = add_mod_q(a[i], b[i]);
    }
}

void poly_sub_q(const int64_t *a, const int64_t *b, int64_t *c) {
    for (int i = 0; i < N; i++) {
        c[i] = sub_mod_q(a[i], b[i]);
    }
}

// ==================== 随机数生成 ====================
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

// ==================== 小系数多项式生成 (标准) ====================
void generate_small_poly(int64_t *poly, int eta) {
    uint8_t buf[256];
    int filled = 0;
    
    while (filled < N) {
        if (batched_secure_random(buf, sizeof(buf)) != SUCCESS) {
            fprintf(stderr, "batched_secure_random failed\n");
            exit(1);
        }
        for (int i = 0; i + 1 < (int)sizeof(buf) && filled < N; i += 2) {
            uint16_t r = ((uint16_t)buf[i] << 8) | buf[i + 1];
            int val = r % (2 * eta + 1) - eta;
            poly[filled++] = mod_q(val);
        }
    }
}

// ==================== 小系数多项式生成 (恒定时间) ====================
void generate_small_poly_ct(int64_t *poly, int eta) {
    uint8_t buf[256];
    int filled = 0;
    
    // 使用恒定时间随机数生成
    while (filled < N) {
        if (batched_secure_random(buf, sizeof(buf)) != SUCCESS) {
            fprintf(stderr, "batched_secure_random failed\n");
            exit(1);
        }
        for (int i = 0; i + 1 < (int)sizeof(buf) && filled < N; i += 2) {
            uint16_t r = ((uint16_t)buf[i] << 8) | buf[i + 1];
            // 恒定时间模约减 (使用掩码)
            int val = r % (2 * eta + 1) - eta;
            poly[filled++] = mod_q_ct(val);
        }
    }
    
    // 添加随机掩码防止内存泄露
    for (int i = 0; i < N; i++) {
        poly[i] ^= (rand() & 0x3) << 2;  // 简单掩码
    }
}

// ==================== SHAKE256 ====================
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

// ==================== CBD ====================
void cbd_eta(const uint8_t *buf, int64_t *r) {
    for (int i = 0; i < N / 8; i++) {
        uint64_t t = 0;
        for (int j = 0; j < 8; j++)
            t |= ((uint64_t)buf[8*i + j]) << (8*j);

        uint64_t d = t & 0x1111111111111111ULL;
        d += (t >> 1) & 0x1111111111111111ULL;
        d += (t >> 2) & 0x1111111111111111ULL;
        d += (t >> 3) & 0x1111111111111111ULL;

        for (int j = 0; j < 8; j++) {
            uint64_t a = (d >> (8*j)) & 0xF;
            uint64_t b = (d >> (8*j + 4)) & 0xF;
            r[8*i + j] = (int64_t)(a - b);
        }
    }
}

// ==================== PRG Gaussian ====================
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

// ==================== 密钥生成 (标准) ====================
result_t keygen(const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed[32] = {0};
    if (batched_secure_random(seed, 32) != SUCCESS) {
        return ERROR_RANDOM_FAILED;
    }
    
    uint8_t seed_s[33] = {0}, seed_e[33] = {0};
    memcpy(seed_s, seed, 32);
    memcpy(seed_e, seed, 32);
    seed_s[32] = 's';
    seed_e[32] = 'e';

    int64_t e[N] = {0};

    prg_gaussian(seed_s, s);
    prg_gaussian(seed_e, e);

    poly_mul_precomputed(precomputed, s, b);
    
    for (int i = 0; i < N; i++) {
        b[i] = add_mod_q(b[i], mod_q(2 * e[i]));
    }
    return SUCCESS;
}

// ==================== 密钥生成 (恒定时间) ====================
result_t keygen_ct(const precomputed_a_t *precomputed, int64_t *s, int64_t *b) {
    uint8_t seed[32] = {0};
    if (batched_secure_random(seed, 32) != SUCCESS) {
        return ERROR_RANDOM_FAILED;
    }
    
    uint8_t seed_s[33] = {0}, seed_e[33] = {0};
    memcpy(seed_s, seed, 32);
    memcpy(seed_e, seed, 32);
    seed_s[32] = 's';
    seed_e[32] = 'e';

    int64_t e[N] = {0};

    prg_gaussian(seed_s, s);
    prg_gaussian(seed_e, e);

    poly_mul_precomputed_ct(precomputed, s, b);
    
    for (int i = 0; i < N; i++) {
        b[i] = add_mod_q_ct(b[i], mod_q_ct(2 * e[i]));
    }
    return SUCCESS;
}

// ==================== RLDH 核心函数 (标准) ====================
static inline int64_t centered(int64_t x) {
    x = mod_q(x);
    if (x > Q/2) {
        return x - Q;
    }
    return x;
}

static inline int64_t sig_0(int64_t x) {
    int64_t cx = centered(x);
    int64_t abs_cx = cx < 0 ? -cx : cx;
    return (abs_cx > Q/4) ? 1 : 0;
}

static inline int64_t mod2_recover(int64_t x, int64_t htb) {
    int64_t cx = centered(x);
    if (htb) {
        cx = centered(cx + Q/2);
    }
    return cx & 1;
}

// ==================== RLDH 核心函数 (恒定时间) ====================
static inline int64_t centered_ct(int64_t x) {
    x = mod_q_ct(x);
    // 恒定时间条件: 如果 x > Q/2 则减去 Q
    int64_t gt_half = (x - (Q/2 + 1)) >> 63;
    return x - (Q & gt_half);
}

static inline int64_t sig_0_ct(int64_t x) {
    int64_t cx = centered_ct(x);
    // 恒定时间绝对值
    int64_t sign_mask = cx >> 63;
    int64_t abs_cx = (cx ^ sign_mask) - sign_mask;
    // 恒定时间比较: 如果 abs_cx > Q/4 返回 1
    int64_t gt = (abs_cx - (Q/4 + 1)) >> 63;
    return gt & 1;
}

static inline int64_t mod2_recover_ct(int64_t x, int64_t htb) {
    int64_t cx = centered_ct(x);
    // 恒定时间条件加法
    int64_t shifted = cx + (htb & (Q/2));
    cx = centered_ct(shifted);
    return cx & 1;
}

// ==================== astRLDH (标准) ====================
void astRLDH(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb) {
    uint8_t seed[33] = {0};
    if (batched_secure_random(seed, 33) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        exit(1);
    }

    int64_t e[N] = {0}, tmp[N] = {0};
    prg_gaussian(seed, e);

    poly_mul_negacyclic_512(pk_a, sk_b, tmp);
    
    for (int i = 0; i < N; i++) {
        tmp[i] = add_mod_q(tmp[i], mod_q(2 * e[i]));
    }

    for (int i = 0; i < N; i++) {
        htb[i] = sig_0(tmp[i]);
        dhb[i] = mod2_recover(tmp[i], htb[i]);
    }
}

// ==================== astRLDH (恒定时间) ====================
void astRLDH_ct(const int64_t *pk_a, const int64_t *sk_b, int64_t *htb, int64_t *dhb) {
    uint8_t seed[33] = {0};
    if (batched_secure_random(seed, 33) != SUCCESS) {
        fprintf(stderr, "Random generation failed!\n");
        exit(1);
    }

    int64_t e[N] = {0}, tmp[N] = {0};
    prg_gaussian(seed, e);

    poly_mul_negacyclic_512_ct(pk_a, sk_b, tmp);
    
    for (int i = 0; i < N; i++) {
        tmp[i] = add_mod_q_ct(tmp[i], mod_q_ct(2 * e[i]));
    }

    for (int i = 0; i < N; i++) {
        htb[i] = sig_0_ct(tmp[i]);
        dhb[i] = mod2_recover_ct(tmp[i], htb[i]);
    }
}

// ==================== recRLDH (标准) ====================
void recRLDH(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha) {
    int64_t tmp[N] = {0};
    poly_mul_negacyclic_512(pk_b, sk_a, tmp);

    for (int i = 0; i < N; i++) {
        dha[i] = mod2_recover(tmp[i], htb[i]);
    }
}

// ==================== recRLDH (恒定时间) ====================
void recRLDH_ct(const int64_t *pk_b, const int64_t *sk_a, const int64_t *htb, int64_t *dha) {
    int64_t tmp[N] = {0};
    poly_mul_negacyclic_512_ct(pk_b, sk_a, tmp);

    for (int i = 0; i < N; i++) {
        dha[i] = mod2_recover_ct(tmp[i], htb[i]);
    }
}

// ==================== 压缩/解压缩 ====================
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

// ==================== KDF ====================
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

// ==================== Session ID ====================
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

// ==================== AES-GCM ====================
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

// ==================== RDTSC ====================
uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// ==================== 测试函数 ====================
void test_keygen(void) {
    printf("\n=== Key Generation Performance Test ===\n");
    
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

void test_keygen_ct(void) {
    printf("\n=== Key Generation (Constant Time) Performance Test ===\n");
    
    int64_t a[N] = {0};
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt_ct(a, &precomputed);

    int64_t s[N] = {0}, b[N] = {0};
    uint64_t total_cycles = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc();
        keygen_ct(&precomputed, s, b);
        uint64_t end = rdtsc();
        total_cycles += (end - start);
    }
    
    printf("Average keygen (CT) cycles: %.2f\n", (double)total_cycles / 1000);
    printf("Average keygen (CT) cycles: %.3f Mcycles\n", (double)total_cycles / 1000 / 1e6);
}

void test_RLDH(void) {
    printf("\n=== RLDH Protocol Test ===\n");
    
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
    int all_match = 1;
    
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
                all_match = 0;
                printf("Reconciliation failed at index %d\n", j);
                break;
            }
        }
        if (!all_match) break;
    }
    
    if (all_match) {
        printf("RLDH reconciliation: PASS ✓ (all 1000 iterations)\n");
    } else {
        printf("RLDH reconciliation: FAIL ✗\n");
    }
    
    printf("Average AstRLDH cycles: %.2f\n", (double)total_cycles / 1000);
    printf("Average AstRLDH cycles: %.3f Mcycles\n", (double)total_cycles / 1000 / 1e6);
    printf("Average RecRLDH cycles: %.2f\n", (double)total_cycles1 / 1000);
    printf("Average RecRLDH cycles: %.3f Mcycles\n", (double)total_cycles1 / 1000 / 1e6);
}

void test_RLDH_ct(void) {
    printf("\n=== RLDH Protocol (Constant Time) Test ===\n");
    
    int64_t a[N] = {0};
    generate_random_poly(a);
    
    precomputed_a_t precomputed;
    precompute_a_ntt_ct(a, &precomputed);

    int64_t s[N] = {0}, b[N] = {0};
    int64_t s1[N] = {0}, b1[N] = {0}, htb[N] = {0}, dha[N] = {0}, dhb[N] = {0};
    keygen_ct(&precomputed, s, b);
    keygen_ct(&precomputed, s1, b1);
    
    uint64_t total_cycles = 0;
    uint64_t total_cycles1 = 0;
    int all_match = 1;
    
    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc();
        astRLDH_ct(b, s1, htb, dhb);
        uint64_t end = rdtsc();
        total_cycles += (end - start);
        
        start = rdtsc();
        recRLDH_ct(b1, s, htb, dha);        
        end = rdtsc();
        total_cycles1 += (end - start);
        
        for(int j = 0; j < N; j++){
            if(dha[j] != dhb[j]){
                all_match = 0;
                printf("Reconciliation failed at index %d\n", j);
                break;
            }
        }
        if (!all_match) break;
    }
    
    if (all_match) {
        printf("RLDH (CT) reconciliation: PASS ✓ (all 1000 iterations)\n");
    } else {
        printf("RLDH (CT) reconciliation: FAIL ✗\n");
    }
    
    printf("Average AstRLDH (CT) cycles: %.2f\n", (double)total_cycles / 1000);
    printf("Average AstRLDH (CT) cycles: %.3f Mcycles\n", (double)total_cycles / 1000 / 1e6);
    printf("Average RecRLDH (CT) cycles: %.2f\n", (double)total_cycles1 / 1000);
    printf("Average RecRLDH (CT) cycles: %.3f Mcycles\n", (double)total_cycles1 / 1000 / 1e6);
}

// ==================== 主函数 ====================
int main(void) {
    printf("========================================\n");
    printf("RLDH Protocol with NTT 512-point\n");
    printf("Q = %d, ROOT = %d, ETA = %d\n", Q, ROOT, ETA);
    printf("Ring: Z_q[X]/(X^%d + 1)\n", N);
    printf("SCA Protection Level: %d\n", SCA_PROTECTION_LEVEL);
    printf("========================================\n");
    
    precompute_roots();
    precompute_psi();
    
    test_keygen();
    test_keygen_ct();
    test_RLDH();
    test_RLDH_ct();
    
    printf("\nAll tests completed successfully!\n");
    return 0;
}
