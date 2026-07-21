/*
 * gpu_GMM.cu
 * Gaussian Mixture Model - Hybrid CPU/GPU (CUDA) Version
 *
 * Compile (Linux):
 *   nvcc -O2 -arch=sm_75 -o gpuGMM gpu_GMM.cu common.c $(gdal-config --cflags) $(gdal-config --libs)
 *
 * Compile (MSVC on Windows / OSGeo4W):
 *   nvcc -O2 -arch=sm_75 -o gpuGMM.exe gpu_GMM.cu common.c -I"C:\OSGeo4W\apps\gdal-dev\include" -L"C:\OSGeo4W\apps\gdal-dev\lib" -lgdal_i
 *
 * Run:
 *   ./gpuGMM <image_filepath> <k> [max_iter] [convergence] [seed] [kmeans_init_iter]
 *   Example: ./gpuGMM sample.tif 4 100 0.001 12345 10
 */

#include <cuda_runtime.h>
#include "common.h"

__global__ void     e_step_kernel(const float * __restrict__ data_bm, float * __restrict__ post, double* __restrict__ d_block_ll, int n_pixels, size_t stride, int n_bands, int k, int has_nodata, float nodata_value);
__global__ void     m_step_means_kernel(const float* __restrict__ data_bm, const float* __restrict__ post, double* __restrict__ d_sum_w, double* __restrict__ d_sum_x, int n_pixels, size_t stride, int n_bands, int k, int has_nodata, float nodata_value, int blocks_per_cluster);
__global__ void     m_step_cov_kernel(const float* __restrict__ data_bm, const float* __restrict__ post, double* __restrict__ d_sum_xx, int n_pixels, size_t stride, int n_bands, int k, int has_nodata, float nodata_value, int blocks_per_cluster);
__global__ void     hard_labels_kernel(const float * __restrict__ post, const float * __restrict__ data_bm, int * __restrict__ labels, int n_pixels, size_t stride, int n_bands, int k, int has_nodata, float nodata_value);

static void         upload_components_to_constant(const GMMModel *model);
static float        finalize_mstep_means(GMMModel *model, const Image *img, const double *h_sum_w, const double *h_sum_x, int k, int nb);
static float        finalize_mstep_cov(GMMModel *model, const double *h_sum_w, const double *h_sum_xx, int n_valid, int k, int nb);
void                gmm_run(GMMModel *model, const Image *img, const GMMConfig *cfg);

#define CUDA_CHECK_ERROR(call) do {                                              \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
        fprintf(stderr, "[CUDA] %s:%d %s\n",                                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));                     \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

// Memoria constante en GPU
__constant__ float c_means[MAX_K * MAX_BANDS];
__constant__ float c_covar_inv[MAX_K * MAX_BANDS * MAX_BANDS];
__constant__ float c_log_det[MAX_K];
__constant__ float c_log_priori[MAX_K];

// Copia medias, covar_inv, log_det y log_priori a memoria constante
static void upload_components_to_constant(const GMMModel *model)
{
    static float h_means[MAX_K * MAX_BANDS];
    static float h_inv  [MAX_K * MAX_BANDS * MAX_BANDS];
    static float h_logd [MAX_K];
    static float h_logp [MAX_K];

    int k  = model->k;
    int nb = model->n_bands;

    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &model->components[c];
        memcpy(&h_means[c * nb], g->mean, nb * sizeof(float));
        memcpy(&h_inv[c * nb * nb], g->covar_inv, nb * nb * sizeof(float));
        h_logd[c] = logf(fmaxf(g->covar_det, FLT_MIN));
        h_logp[c] = logf(fmaxf(g->priori, FLT_MIN));
    }

    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_means, h_means, (size_t)k * nb * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_covar_inv, h_inv, (size_t)k * nb * nb * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_log_det, h_logd, (size_t)k * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_log_priori, h_logp, (size_t)k * sizeof(float)));
}

// Un thread por pixel, calculamos la posteriori de cada pixel y su contribucion
// al log-likelihood
__global__ void e_step_kernel(const float * __restrict__ data_bm, 
    float * __restrict__ post, double* __restrict__ d_block_ll, 
    int n_pixels, size_t stride, int n_bands, int k, int has_nodata, 
    float nodata_value)
{
    extern __shared__ double s_ll[];
    int tid = threadIdx.x;
    int p = blockIdx.x * blockDim.x + tid;

    s_ll[tid] = 0.0;

    if (p < n_pixels) {
        float x[MAX_BANDS];
        int is_nd = has_nodata;
        for (int b = 0; b < n_bands; b++) {
            float v = data_bm[(size_t)b * stride + p];
            x[b] = v;
            if (is_nd && v != nodata_value) is_nd = 0;
        }

        if (is_nd) {
            for (int c = 0; c < k; c++)
                post[(size_t)c * n_pixels + p] = 0.0f;
        } else {
            float log_probs[MAX_K];
            float two_pi_term = (float)n_bands * logf(2.0f * (float)M_PI);

            for (int c = 0; c < k; c++) {
                int mean_base = c * n_bands;
                int cov_base  = c * n_bands * n_bands;

                // quad = (x-μ)ᵀ Σ⁻¹ (x-μ)
                float quad = 0.0f;
                for (int i = 0; i < n_bands; i++) {
                    float di  = x[i] - c_means[mean_base + i];
                    float tmp = 0.0f;
                    for (int j = 0; j < n_bands; j++) {
                        float dj = x[j] - c_means[mean_base + j];
                        tmp += c_covar_inv[cov_base + i * n_bands + j] * dj;
                    }
                    quad += di * tmp;
                }

                // log_probs = log(π_k) - ½ [ d·log(2π) + log|Σ| + (x-μ)ᵀ Σ⁻¹ (x-μ) ]
                log_probs[c] = c_log_priori[c]
                             - 0.5f * (two_pi_term + c_log_det[c] + quad);
            }

            // log‑sum‑exp
            float lse_max = log_probs[0];
            for (int c = 1; c < k; c++)
                if (log_probs[c] > lse_max) lse_max = log_probs[c];

            float lse_sum = 0.0f;
            for (int c = 0; c < k; c++)
                lse_sum += expf(log_probs[c] - lse_max);

            float log_norm = lse_max + logf(lse_sum);

            // Guardamos la log-likelihood del pixel
            s_ll[tid] = (double)log_norm;

            for (int c = 0; c < k; c++)
                post[(size_t)c * n_pixels + p] = expf(log_probs[c] - log_norm);
        }
    }
    __syncthreads();

    // Reduccion paralela
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_ll[tid] += s_ll[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_block_ll[blockIdx.x] = s_ll[0];
}

// blocks_per_cluster warps por componente gaussiana, acumulamos sus sumas
__global__ void m_step_means_kernel(const float* __restrict__ data_bm,
    const float* __restrict__ post, double* __restrict__ d_sum_w,
    double* __restrict__ d_sum_x, int n_pixels, size_t stride, 
    int n_bands, int k, int has_nodata, float nodata_value, 
    int blocks_per_cluster)
{
    int c = blockIdx.x;
    int slice = blockIdx.y;
    int tid = threadIdx.x;
    int nb = n_bands;
    int loop_stride = blocks_per_cluster * blockDim.x;

    double sum_w = 0.0;
    double sum_x[MAX_BANDS] = {0.0};

    for (int p = slice * blockDim.x + tid; p < n_pixels; p += loop_stride) {
        float r = post[(size_t)c * n_pixels + p];
        if (r == 0.0f) continue;

        float x[MAX_BANDS];
        bool valid = true;
        if (has_nodata) {
            valid = false;
            for (int b = 0; b < nb; b++) {
                float v = data_bm[(size_t)b * stride + p];
                x[b] = v;
                if (v != nodata_value) valid = true;
            }
        } else {
            for (int b = 0; b < nb; b++)
                x[b] = data_bm[(size_t)b * stride + p];
        }
        if (!valid) continue;

        // Acumulamos en sum_w y sum_x
        double dr = (double)r;
        sum_w += dr;
        for (int b = 0; b < nb; b++)
            sum_x[b] += dr * (double)x[b];
    }

    // Los 32 threads del warp hacen un atomicAdd a la misma direccion,
    // por tanto el hardware los fusiona en un solo atomicAdd
    atomicAdd(&d_sum_w[c], sum_w);
    for (int b = 0; b < nb; b++)
        atomicAdd(&d_sum_x[c * nb + b], sum_x[b]);
}

// blocks_per_cluster warps por componente gaussiana
__global__ void m_step_cov_kernel(const float* __restrict__ data_bm,
    const float* __restrict__ post, double* __restrict__ d_sum_xx,
    int n_pixels, size_t stride, int n_bands, int k, int has_nodata, 
    float nodata_value, int blocks_per_cluster)
{
    int c = blockIdx.x;
    int slice = blockIdx.y;
    int tid = threadIdx.x;
    int nb = n_bands;
    int ut_len = nb * (nb + 1) / 2;
    int loop_stride = blocks_per_cluster * blockDim.x;

    double sum_xx[MAX_BANDS * (MAX_BANDS + 1) / 2] = {0.0};

    for (int p = slice * blockDim.x + tid; p < n_pixels; p += loop_stride) {
        float r = post[(size_t)c * n_pixels + p];
        if (r == 0.0f) continue;

        float x[MAX_BANDS];
        bool valid = true;
        if (has_nodata) {
            valid = false;
            for (int b = 0; b < nb; b++) {
                float v = data_bm[(size_t)b * stride + p];
                x[b] = v;
                if (v != nodata_value) valid = true;
            }
        } else {
            for (int b = 0; b < nb; b++)
                x[b] = data_bm[(size_t)b * stride + p];
        }
        if (!valid) continue;

        // sum_xx = Σ post * (x-μ) * (x-μ)ᵀ
        double dr = (double)r;
        for (int i = 0; i < nb; i++) {
            double di = (double)x[i] - (double)c_means[c * nb + i];
            for (int j = i; j < nb; j++) {
                double dj = (double)x[j] - (double)c_means[c * nb + j];
                int idx = i * (2 * nb - i + 1) / 2 + (j - i);
                sum_xx[idx] += dr * di * dj;
            }
        }
    }

    // Los 32 threads del warp hacen un atomicAdd a la misma direccion,
    // por tanto el hardware los fusiona en un solo atomicAdd
    for (int i = 0; i < ut_len; i++)
        atomicAdd(&d_sum_xx[c * ut_len + i], sum_xx[i]);
}

// Actualizamos las medias en host (CPU)
static float finalize_mstep_means(GMMModel *model, const Image *img,
    const double *h_sum_w, const double *h_sum_x, int k, int nb)
{
    float max_shift = 0.0f;

    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &model->components[c];

        // Componentes practicamente vacias -> les asignamos un pixel aleatorio
        if (h_sum_w[c] < DBL_MIN) {
            int rp = get_random_pixel(img);

            for (int b = 0; b < nb; b++)
                g->mean[b] = img->data[(size_t)rp * nb + b];

            memset(g->covar, 0, nb * nb * sizeof(float));
            for (int b = 0; b < nb; b++)
                g->covar[b * nb + b] = MIN_COVAR;
            compute_covar_derived(g, nb);
            g->priori = 1.0f / k;
            continue;
        }

        // Actualizamos la media
        double inv_w = 1.0 / h_sum_w[c];
        for (int i = 0; i < nb; i++) {
            float new_mean = (float)(h_sum_x[c * nb + i] * inv_w);
            float d = fabsf(new_mean - g->mean[i]);
            if (d > max_shift) max_shift = d;
            g->mean[i] = new_mean;
        }
    }

    return max_shift;
}

// Actualizamos las covarianzas en host (CPU)
static float finalize_mstep_cov(GMMModel *model, const double *h_sum_w, 
    const double *h_sum_xx, int n_valid, int k, int nb)
{
    int ut_len = nb * (nb + 1) / 2;
    float max_shift = 0.0f;

    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &model->components[c];

        // Saltamos componentes vacias
        if (h_sum_w[c] < DBL_MIN)
            continue;

        double inv_w = 1.0 / h_sum_w[c];
        float new_cov[MAX_BANDS * MAX_BANDS];

        for (int i = 0; i < nb; i++) {
            for (int j = i; j < nb; j++) {
                int idx_ut = c * ut_len + i * (2 * nb - i + 1) / 2 + (j - i);
                double cov_ij = h_sum_xx[idx_ut] * inv_w;
                if (i == j) cov_ij += MIN_COVAR;
                float val = (float)cov_ij;
                new_cov[i * nb + j] = val;
                new_cov[j * nb + i] = val; // simetria triangular de la matriz de cov.
                float d = fabsf(val - g->covar[i * nb + j]);
                if (d > max_shift) max_shift = d;
            }
        }
        memcpy(g->covar, new_cov, nb * nb * sizeof(float));
        g->priori = (float)(h_sum_w[c] / (double)n_valid);

        compute_covar_derived(g, nb);
    }

    // Renormalizamos prioris
    float sum_p = 0.0f;
    for (int c = 0; c < k; c++) sum_p += model->components[c].priori;
    if (sum_p > 0.0f)
        for (int c = 0; c < k; c++) model->components[c].priori /= sum_p;
    else
        for (int c = 0; c < k; c++) model->components[c].priori = 1.0f / k;

    return max_shift;
}

// Obtenemos la mayor posteriori que define a que cluster pertenece cada pixel,
// lanzando un thread por pixel
__global__ void hard_labels_kernel(const float * __restrict__ post, 
    const float * __restrict__ data_bm, int * __restrict__ labels, 
    int n_pixels, size_t stride, int n_bands, int k, int has_nodata, 
    float nodata_value)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    if (has_nodata) {
        int all_nd = 1;
        for (int b = 0; b < n_bands; b++) {
            if (data_bm[(size_t)b * stride + p] != nodata_value) { all_nd = 0; break; }
        }
        if (all_nd) { labels[p] = -1; return; }
    }

    int   best = 0;
    float bval = post[(size_t)0 * n_pixels + p];
    for (int c = 1; c < k; c++) {
        float v = post[(size_t)c * n_pixels + p];
        if (v > bval) { bval = v; best = c; }
    }
    labels[p] = best;
}

void gmm_run(GMMModel *model, const Image *img, const GMMConfig *cfg)
{
    if (img->n_bands != model->n_bands) {
        fprintf(stderr, "[GMM]    Error: image has %d bands but model expects %d.\n",
                img->n_bands, model->n_bands);
        exit(EXIT_FAILURE);
    }
    if (img->n_bands > MAX_BANDS) {
        fprintf(stderr, "[GMM]    Error: n_bands (%d) > MAX_BANDS (%d). Increase MAX_BANDS and recompile.\n",
                img->n_bands, MAX_BANDS);
        exit(EXIT_FAILURE);
    }

    int k = model->k;
    int nb = model->n_bands;
    int np = img->n_pixels;
    int grid_px = (np + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shmem_size = BLOCK_SIZE * sizeof(double);

    int blocks_per_cluster = 512;
    dim3 grid_m(k, blocks_per_cluster);
    dim3 block_m(32);

    // Convertimos datos a banda-mayor
    size_t padded_np;
    float *h_data_bm = transpose_to_band_major(img, &padded_np);
    if (!h_data_bm) {
        fprintf(stderr, "Error: cannot allocate band-major buffer\n"); exit(1);
    }

    // Contamos pixeles no validos
    int n_valid = 0;
    for (int p = 0; p < np; p++)
        if (!is_nodata(&img->data[(size_t)p * nb], img)) n_valid++;

    // Alocacion de memoria en dispositivo
    int ut_len = nb * (nb + 1) / 2;
    float  *d_data_bm = NULL;
    float  *d_post = NULL;
    double *d_block_ll = NULL;
    int    *d_labels = NULL;
    double *d_sum_w = NULL;
    double *d_sum_x = NULL;
    double *d_sum_xx = NULL;

    CUDA_CHECK_ERROR(cudaMalloc(&d_data_bm, padded_np * nb * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_post, (size_t)np * k * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_block_ll, (size_t)grid_px * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_labels, (size_t)np * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum_w, (size_t)k * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum_x, (size_t)k * nb * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum_xx, (size_t)k * ut_len * sizeof(double)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_data_bm, h_data_bm, padded_np * nb * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data_bm);

    // Subimos datos al dispositivo
    upload_components_to_constant(model);

    // Alocacion de memoria en host
    double *h_sum_w = (double*)malloc((size_t)k * sizeof(double));
    double *h_sum_x = (double*)malloc((size_t)k * nb * sizeof(double));
    double *h_sum_xx = (double*)malloc((size_t)k * ut_len * sizeof(double));
    double *h_block_ll = (double*)malloc((size_t)grid_px * sizeof(double));

    double prev_ll = -DBL_MAX;
    int converged = 0;
    int last_iter = 0;

    // Lanzamos el timer
    cudaEvent_t iter_start, iter_stop;
    cudaEventCreate(&iter_start);
    cudaEventCreate(&iter_stop);
    cudaEventRecord(iter_start, 0);

    // Expectation-Maximization
    for (int iter = 0; iter < cfg->max_iter; iter++) {
        last_iter = iter + 1;

        // E-step
        e_step_kernel<<<grid_px, BLOCK_SIZE, shmem_size>>>(
            d_data_bm, d_post, d_block_ll, np, padded_np, nb, k, img->has_nodata, img->nodata_value);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Copiamos log-likelihoods parciales y reducimos
        CUDA_CHECK_ERROR(cudaMemcpy(h_block_ll, d_block_ll, (size_t)grid_px * sizeof(double), cudaMemcpyDeviceToHost));
        double ll = 0.0;
        for (int i = 0; i < grid_px; i++)
            ll += h_block_ll[i];

        // M-step (medias)
        CUDA_CHECK_ERROR(cudaMemset(d_sum_w, 0, (size_t)k * sizeof(double)));
        CUDA_CHECK_ERROR(cudaMemset(d_sum_x, 0, (size_t)k * nb * sizeof(double)));

        m_step_means_kernel<<<grid_m, block_m>>>(
            d_data_bm, d_post,
            d_sum_w, d_sum_x,
            np, padded_np, nb, k, img->has_nodata, img->nodata_value, blocks_per_cluster);
        CUDA_CHECK_ERROR(cudaGetLastError());

        CUDA_CHECK_ERROR(cudaMemcpy(h_sum_w, d_sum_w, (size_t)k * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(h_sum_x, d_sum_x, (size_t)k * nb * sizeof(double), cudaMemcpyDeviceToHost));

        // Calculamos el maximo desplazamiento de medias
        float shift_means = finalize_mstep_means(model, img, h_sum_w, h_sum_x, k, nb);

        // Subimos nuevas medias a dispositivo
        upload_components_to_constant(model);

        // M-step (matriz de covarianzas)
        CUDA_CHECK_ERROR(cudaMemset(d_sum_xx, 0, (size_t)k * ut_len * sizeof(double)));

        m_step_cov_kernel<<<grid_m, block_m>>>(
            d_data_bm, d_post, d_sum_xx,
            np, padded_np, nb, k, img->has_nodata, img->nodata_value,
            blocks_per_cluster);
        CUDA_CHECK_ERROR(cudaGetLastError());

        CUDA_CHECK_ERROR(cudaMemcpy(h_sum_xx, d_sum_xx, (size_t)k * ut_len * sizeof(double), cudaMemcpyDeviceToHost));

        // Calculamos el maximo desplazamiento de covarianzas
        float shift_cov = finalize_mstep_cov(model, h_sum_w, h_sum_xx, n_valid, k, nb);
        float max_shift = fmaxf(shift_means, shift_cov);

        // Subimos parametros finales a dispositivo
        upload_components_to_constant(model);

        double ll_change = (iter == 0) ? 0.0 : fabs((ll - prev_ll) / fmax(fabs(ll), 1.0));

        printf("[GMM]    iter %3d | log-likelihood = %.4f | ll delta = %.7f | max shift = %.6f\n",
               last_iter, ll, ll_change, max_shift);

        // DEBUG
        for (int c = 0; c < k; c++) {
            if (!isfinite(model->components[c].covar_det) || model->components[c].covar_det <= 0.0f)
                fprintf(stderr, "[M-step] component %d has bad det=%.6e\n", c, model->components[c].covar_det);
            float sum = 0.0f;
            for (int b = 0; b < nb; b++) sum += model->components[c].mean[b];
            if (!isfinite(sum))
                fprintf(stderr, "[M-step] component %d has NaN/Inf in mean\n", c);
        }

        if (iter > 0 && ll < prev_ll - 1.0) {
            fprintf(stderr, "[GMM]    WARNING: LL decreased by %.6f at iter %d - EM monotonicity violated.\n",
                    prev_ll - ll, last_iter);
            for (int c = 0; c < k; c++)
                fprintf(stderr, "         component %d: priori=%.6f  det=%.6e\n",
                        c, model->components[c].priori, model->components[c].covar_det);

            fprintf(stderr, "         posterior sums: ");
            for (int c = 0; c < k; c++) fprintf(stderr, "[%d]=%.1f ", c, h_sum_w[c]);
            fprintf(stderr, "\n");
        }

        for (int c = 0; c < k; c++) {
            if (model->components[c].priori < (1.0f / k) * 0.01f)
                fprintf(stderr, "         component %d has suspiciously low priori (%.6f).\n",
                        c, model->components[c].priori);
        }

        if (iter > 0 && ll_change < cfg->convergence) {
            printf("[GMM]    Converged at iteration %d.\n", last_iter);
            converged = 1;
            break;
        }
        prev_ll = ll;
    }

    if (!converged)
        printf("[GMM]    Reached maximum iterations (%d).\n", cfg->max_iter);

    // Ultimo e-step para actualizar las posterioris
    e_step_kernel<<<grid_px, BLOCK_SIZE, shmem_size>>>(
        d_data_bm, d_post, d_block_ll, np, padded_np, nb, k, img->has_nodata, img->nodata_value);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Copiamos centroides y etiquetas finales al host
    hard_labels_kernel<<<grid_px, BLOCK_SIZE>>>(
        d_post, d_data_bm, d_labels, np, padded_np, nb, k, img->has_nodata, img->nodata_value);
    CUDA_CHECK_ERROR(cudaGetLastError());

    model->labels = (int*)malloc((size_t)np * sizeof(int));
    CUDA_CHECK_ERROR(cudaMemcpy(model->labels, d_labels, (size_t)np * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(iter_stop, 0);
    cudaEventSynchronize(iter_stop);
    float iter_time = 0;
    cudaEventElapsedTime(&iter_time, iter_start, iter_stop);
    printf("\n[Time]   EM loop elapsed: %.4f seconds (GPU + host overhead)\n", iter_time / 1000.0f);

    cudaEventDestroy(iter_start);
    cudaEventDestroy(iter_stop);
    
    cudaFree(d_data_bm);
    cudaFree(d_post);
    cudaFree(d_block_ll);
    cudaFree(d_labels);
    cudaFree(d_sum_w);
    cudaFree(d_sum_x);
    cudaFree(d_sum_xx);
    free(h_sum_w); free(h_sum_x); free(h_sum_xx); free(h_block_ll);
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 6) {
        fprintf(stderr, "Error: GPU compute capability %d.%d is too low. Need >= 6.0 for double atomicAdd.\n",
                prop.major, prop.minor);
        exit(EXIT_FAILURE);
    }

    if (argc < 3) {
        fprintf(stderr, "Usage:   %s <image_filepath> <k> [max_iter] [convergence] [seed] [k-means_init_iter]\n", argv[0]);
        fprintf(stderr, "Example: %s sample.tif 4 100 0.003 12345 10\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    int k = atoi(argv[2]);
    if (k <= 0) {
        fprintf(stderr, "'k' must be a positive integer.\n");
        return 1;
    }
    if (k > MAX_K) {
        fprintf(stderr, "'k' cannot exceed %d (color palette / constant-memory limit).\n", MAX_K);
        return 1;
    }

    GMMConfig cfg;
    cfg.max_iter = 200;
    cfg.convergence = 1e-3f;
    unsigned int seed = 12345;
    lcg_set_seed(seed);
    cfg.kmeans_init_iter = 10;

    if (argc >= 4) {
        if (atoi(argv[3]) <= 0) {
            fprintf(stderr, "Invalid max_iterations: %d (must be > 0)\n", atoi(argv[3]));
            return 1;
        }
        cfg.max_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        if ((float)atof(argv[4]) <= 0) {
            fprintf(stderr, "Invalid convergence: %f (must be > 0)\n", (float)atof(argv[4]));
            return 1;
        }
        cfg.convergence = (float)atof(argv[4]);
    }
    if (argc >= 6) {
        seed = (unsigned int)atoi(argv[5]);
        lcg_set_seed(seed);
    }
    if (argc >= 7) {
        if (atoi(argv[6]) <= 0) {
            fprintf(stderr, "Invalid k-means_init_iter: %d (must be > 0)\n", atoi(argv[6]));
            return 1;
        }
        cfg.kmeans_init_iter = atoi(argv[6]);
    }

    GDALAllRegister();

    printf("=== GMM - CPU/GPU Version ===\n");
    printf("[Config] k=%d | image=\"%s\"\n", k, filename);
    printf("[Config] max_iter=%d | convergence=%.7f | seed=%u | k-means_init_iter=%d\n\n",
           cfg.max_iter, cfg.convergence, seed, cfg.kmeans_init_iter);

    Image *img = load_image(filename);
    if (!img) return 1;

    GMMModel *model = gmm_init(k, img->n_bands);

    clock_t init_start = clock();
    init_params_kmeans(model, img, cfg.kmeans_init_iter);
    clock_t init_end = clock();

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    gmm_run(model, img, &cfg);

    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    printf("[Time]   Total elapsed: %.4f seconds (GPU + host overhead)\n", ms / 1000.0f);
    printf("[Time]   Initialization elapsed: %.4f seconds\n\n", (double)(init_end - init_start) / CLOCKS_PER_SEC);

    print_components(model);
    printf("\n");
    print_cluster_sizes(model->labels, img->n_pixels, k);
    printf("\n");
    save_csv(model->labels, img->n_pixels, "gmm_output.csv");
    save_image(model->labels, img->n_pixels, model->k, "gmm_output.tif", filename);

    gmm_free(model);
    free_image(img);
    return 0;
}