/*
 * gpu_Kmeans.cu
 * K-Means - Hybrid CPU/GPU (CUDA) Version
 *
 * Compile (Linux):
 *   nvcc -O2 -arch=sm_75 -o gpuKM gpu_Kmeans.cu common.c $(gdal-config --cflags) $(gdal-config --libs)
 *
 * Compile (MSVC on Windows / OSGeo4W):
 *   nvcc -O2 -arch=sm_75 -o gpuKM.exe gpu_Kmeans.cu common.c -I"C:\OSGeo4W\apps\gdal-dev\include" -L"C:\OSGeo4W\apps\gdal-dev\lib" -lgdal_i
 *
 * Run:
 *   ./gpuKM <image_filepath> <number_of_clusters> [init: random|pp] [max_iterations] [convergence] [seed]
 *   Example: ./gpuKM sample.tif 5 pp 100 0.0003 12345
 */

#include <cuda_runtime.h>
#include "common.h"

__global__ void     assign_labels_kernel(const float * __restrict__ data_bm, int * __restrict__ labels, int n_pixels, size_t stride, int n_bands, int k, float nodata_value, int has_nodata);
__global__ void     accumulate_kernel(const float* __restrict__ data_bm, const int*   __restrict__ labels, double* __restrict__ d_partial_sums, int* __restrict__ d_partial_counts, int n_pixels, size_t stride, int n_bands, int k);
__global__ void     reduce_kernel(const double* __restrict__ d_partial_sums, const int*    __restrict__ d_partial_counts, double* d_sums, int*    d_counts, int n_blocks_stride, int k, int n_bands);
__global__ void     finalize_centroids_kernel(float *centroids, const double *sums, const int *counts, int k, int n_bands);

static void         compute_shift(const float *prev, const float *now, int n, float *shift);
void                kmeans_run(KMeansModel *model, const Image *img, const KMeansConfig *cfg);

#define CUDA_CHECK_ERROR(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "[CUDA] %s:%d %s\n",                               \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// Memoria constante en GPU
__constant__ float c_centroids[MAX_K * MAX_BANDS];

// 1 hilo por píxel, compara con cada centroide (c_centroids)
__global__ void assign_labels_kernel(const float * __restrict__ data_bm, 
    int * __restrict__ labels, int n_pixels, size_t stride, int n_bands, int k, 
    float nodata_value, int has_nodata)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;

    if (has_nodata) {
        int valid = 0;
        for (int b = 0; b < n_bands; b++) {
            if (data_bm[(size_t)b * stride + p] != nodata_value) {
                valid = 1;
                break;
            }
        }
        if (!valid) {
            labels[p] = -1;
            return;
        }
    }

    float best_d = FLT_MAX;
    int   best_c = 0;

    for (int c = 0; c < k; c++) {
        float d = 0.0f;
        for (int b = 0; b < n_bands; b++) {
            float v = data_bm[(size_t)b * stride + p];
            float diff = v - c_centroids[c * n_bands + b];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best_c = c; }
    }
    labels[p] = best_c;
}

__global__ void accumulate_kernel(const float* __restrict__ data_bm,
    const int* __restrict__ labels, double* __restrict__ d_partial_sums,
    int* __restrict__ d_partial_counts, int n_pixels, size_t stride, 
    int n_bands, int k)
{
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int slots = k * n_bands;

    extern __shared__ float s_mem[];
    // Sumas acumuladas por warp: warps_per_block * k * n_bands floats
    float* s_sums = s_mem;
    // Contadores acumulados por warp: warps_per_block * k ints
    int* s_counts = (int*)&s_sums[warps_per_block * slots];

    float* sums = &s_sums[warp_id * slots];
    int* counts = &s_counts[warp_id * k];

    for (int i = lane; i < slots; i += 32) sums[i] = 0.0;
    for (int i = lane; i < k; i += 32) counts[i] = 0;
    __syncwarp();

    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n_pixels; p += blockDim.x * gridDim.x) {
        int c = labels[p];
        if (c < 0) continue;
        atomicAdd(&counts[c], 1);
        for (int b = 0; b < n_bands; b++)
            atomicAdd(&sums[c * n_bands + b], data_bm[(size_t)b * stride + p]);
    }
    __syncthreads();

    // Combinamos acumuladores de warps en el acumulador del bloque
    if (warp_id == 0) {
        for (int i = lane; i < slots; i += 32) {
            double total = 0.0;
            for (int w = 0; w < warps_per_block; w++)
                total += (double)s_sums[w * slots + i];
            d_partial_sums[blockIdx.x * slots + i] = total;
        }
        for (int i = lane; i < k; i += 32) {
            int total = 0;
            for (int w = 0; w < warps_per_block; w++)
                total += s_counts[w * k + i];
            d_partial_counts[blockIdx.x * k + i] = total;
        }
    }
}

// Reduce las sumas y conteos parciales, un thread por banda por cluster
__global__ void reduce_kernel(const double* __restrict__ d_partial_sums,
    const int*    __restrict__ d_partial_counts, double* d_sums, int*    d_counts,
    int n_blocks_stride, int k, int n_bands)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * n_bands) return;

    int c = idx / n_bands;
    int b = idx % n_bands;

    double sum_val = 0.0;
    for (int blk = 0; blk < n_blocks_stride; blk++)
        sum_val += d_partial_sums[blk * k * n_bands + idx];
    d_sums[idx] = sum_val;

    // Solo un thread por cluster escribe en memoria global
    if (b == 0) {
        int cnt = 0;
        for (int blk = 0; blk < n_blocks_stride; blk++)
            cnt += d_partial_counts[blk * k + c];
        d_counts[c] = cnt;
    }
}

// Actualizacion de los nuevos centroides, un thread por banda por cluster
__global__ void finalize_centroids_kernel(float *centroids, const double *sums, 
    const int *counts, int k, int n_bands)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * n_bands) return;
    int c = idx / n_bands;
    int cnt = counts[c];
    if (cnt == 0) return;

    centroids[idx] = (float)(sums[idx] / (double)cnt);
}

void compute_shift(const float *prev, const float *now, int n, float *shift)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(now[i] - prev[i]);
        if (d > s) s = d;
    }
    *shift = s;
}

void kmeans_run(KMeansModel *model, const Image *img, const KMeansConfig *cfg)
{
    int k = model->k;
    int nb = model->n_bands;
    int np = img->n_pixels;

    if (nb > MAX_BANDS) {
        fprintf(stderr, "Error: n_bands (%d) > MAX_BANDS (%d). Increase MAX_BANDS and recompile.\n", nb, MAX_BANDS);
        exit(EXIT_FAILURE);
    }

    model->labels = (int*)malloc((size_t)np * sizeof(int));

    // Convertimos datos a banda-mayor
    size_t padded_np;
    float *h_data_bm = transpose_to_band_major(img, &padded_np);
    if (!h_data_bm) { fprintf(stderr, "Error: cannot allocate band-major buffer\n"); exit(1); }

    // Geometria de lanzamiento de kernels
    int grid_px = (np + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n_cent = k * nb;
    int grid_cf = (n_cent + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int n_sm;
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0); // 30 SMs para RTX 2060 (Turing, CC 7.5)
    int n_blocks_stride = n_sm * 64;
    dim3 block_acc(256);
    dim3 grid_acc(n_blocks_stride);
    int warps_per_block = block_acc.x / 32;
    size_t shmem = warps_per_block * (k * nb * sizeof(float) + k * sizeof(int));

    // Alocacion de memoria en dispositivo
    float  *d_data_bm = NULL;
    float  *d_centroids = NULL;
    int    *d_labels = NULL;
    double *d_sums = NULL;
    int    *d_counts = NULL;
    double *d_partial_sums = NULL;
    int    *d_partial_counts = NULL;

    CUDA_CHECK_ERROR(cudaMalloc(&d_data_bm, padded_np * nb * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_centroids, (size_t)k  * nb * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_labels, (size_t)np * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sums, (size_t)k  * nb * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_counts, (size_t)k  * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_partial_sums, (size_t)n_blocks_stride * k * nb * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_partial_counts, (size_t)n_blocks_stride * k * sizeof(int)));

    // Subimos datos al dispositivo
    CUDA_CHECK_ERROR(cudaMemcpy(d_data_bm, h_data_bm, padded_np * nb * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_centroids, model->centroids, (size_t)k * nb * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_centroids, model->centroids, (size_t)k * nb * sizeof(float)));
    free(h_data_bm);

    // Alocacion de memoria en host
    float *h_centroids_prev = NULL;
    h_centroids_prev = (float*)malloc((size_t)k * nb * sizeof(float));
    memcpy(h_centroids_prev, model->centroids, (size_t)k * nb * sizeof(float));

    int last_iter = 0;
    int converged = 0;

    // Lanzamos el timer
    cudaEvent_t iter_start, iter_stop;
    cudaEventCreate(&iter_start);
    cudaEventCreate(&iter_stop);
    cudaEventRecord(iter_start, 0);

    // Bucle de asignacion-actualizacion
    for (int iter = 0; iter < cfg->max_iter; iter++) {
        last_iter = iter + 1;

        // Asignamos centroides
        assign_labels_kernel<<<grid_px, BLOCK_SIZE>>>(
            d_data_bm, d_labels, np, padded_np, nb, k, img->nodata_value, img->has_nodata);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Acumulacion de sumas y conteos parciales
        accumulate_kernel<<<grid_acc, block_acc, shmem>>>(
            d_data_bm, d_labels,
            d_partial_sums, d_partial_counts,
            np, padded_np, nb, k);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Reduccion de las sumas y conteos parciales
        int reduce_grid = (k * nb + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_kernel<<<reduce_grid, BLOCK_SIZE>>>(
            d_partial_sums, d_partial_counts,
            d_sums, d_counts,
            n_blocks_stride, k, nb);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Actualizacion de los centroides
        finalize_centroids_kernel<<<grid_cf, BLOCK_SIZE>>>(
            d_centroids, d_sums, d_counts, k, nb);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Copiamos nuevos centroides del dispositivo al host (de GPU a CPU)
        CUDA_CHECK_ERROR(cudaMemcpy(model->centroids, d_centroids, (size_t)k * nb * sizeof(float), cudaMemcpyDeviceToHost));

        // Calculamos el maximo desplazamiento
        float shift;
        compute_shift(h_centroids_prev, model->centroids, k * nb, &shift);
        memcpy(h_centroids_prev, model->centroids, (size_t)k * nb * sizeof(float));

        // Actualizamos la memoria constante para la prox. iteracion
        CUDA_CHECK_ERROR(cudaMemcpyToSymbol(c_centroids, d_centroids, (size_t)k * nb * sizeof(float), 0, cudaMemcpyDeviceToDevice));

        printf("[K-Means] iter %3d | max centroid shift = %.7f\n", last_iter, shift);

        if (shift < cfg->convergence) {
            converged = 1;
            printf("[K-Means] Converged at iteration %d.\n", last_iter);
            break;
        }
    }

    if (!converged)
        printf("[K-Means] Reached maximum iterations (%d). Stopping now.\n", cfg->max_iter);

    // Copiamos centroides y etiquetas finales al host
    CUDA_CHECK_ERROR(cudaMemcpy(model->centroids, d_centroids, (size_t)k * nb * sizeof(float), cudaMemcpyDeviceToHost));
    assign_labels_kernel<<<grid_px, BLOCK_SIZE>>>(
        d_data_bm, d_labels, np, padded_np, nb, k, img->nodata_value, img->has_nodata);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaMemcpy(model->labels, d_labels, (size_t)np * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(iter_stop, 0);
    cudaEventSynchronize(iter_stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, iter_start, iter_stop);
    printf("\n[Time]   K-means loop elapsed: %.4f seconds (GPU + host overhead)\n", ms / 1000.0f);

    cudaEventDestroy(iter_start);
    cudaEventDestroy(iter_stop);

    if (h_centroids_prev) free(h_centroids_prev);
    cudaFree(d_data_bm);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_partial_sums);
    cudaFree(d_partial_counts);
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 3) {
        fprintf(stderr, "Usage:   %s <image_filepath> <number_of_clusters> [init: \"random\"(default)|\"pp\"] [max_iterations] [convergence] [seed]\n", argv[0]);
        fprintf(stderr, "Example: %s sample.tif 4 random 100 0.0003 12345\n", argv[0]);
        fprintf(stderr, "Example: %s sample.tif 4 pp 100 0.0003 12345\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];

    int k = atoi(argv[2]);
    if (k <= 0) {
        fprintf(stderr, "'k' number of clusters must be a positive integer.\n");
        return 1;
    }
    if (k > MAX_K) {
        fprintf(stderr, "'k' cannot exceed %d (color palette / constant-memory limit).\n", MAX_K);
        return 1;
    }

    KMeansConfig cfg; // config por defecto
    cfg.init = 0;
    cfg.max_iter = 200;
    cfg.convergence = 1e-3f;
    unsigned int seed = 12345;
    lcg_set_seed(seed);

    if (argc >= 4) {
        if (strcmp(argv[3], "pp") == 0)
            cfg.init = 1;
        else if (strcmp(argv[3], "random") == 0)
            cfg.init = 0;
        else {
            fprintf(stderr, "Invalid init method: %s (use 'random' or 'pp')\n", argv[3]);
            return 1;
        }
    }

    if (argc >= 5) {
        if(atoi(argv[4]) <= 0) {
            fprintf(stderr, "Invalid max_iterations: %d (must be > 0)\n", atoi(argv[4]));
            return 1;
        } else {
            cfg.max_iter = atoi(argv[4]);
        }
    }

    if (argc >= 6) {
        if((float)atof(argv[5]) <= 0) {
            fprintf(stderr, "Invalid convergence: %f (must be > 0)\n", (float)atof(argv[5]));
            return 1;
        } else {
            cfg.convergence = (float)atof(argv[5]);
        }
    }

    if (argc >= 7) {
        seed = (unsigned int)atoi(argv[6]);
        lcg_set_seed(seed);
    }

    GDALAllRegister();

    printf("=== K-Means - CPU/GPU Version ===\n");
    printf(cfg.init == 0 ? "[Config] Random initialization.\n" : "[Config] K-means++ initialization.\n");
    printf("[Config] k=%d | image_filepath=\"%s\"\n", k, filename);
    printf("[Config] max_iter=%d | convergence=%.7f | seed=%u\n\n", cfg.max_iter, cfg.convergence, seed);

    Image *img = load_image(filename);
    if (!img) return 1;

    KMeansModel *model = kmeans_init(k, img->n_bands);

    clock_t init_start = clock();
    if (cfg.init == 1)
        // init == 1 -> Inicializacion con k-means++
        init_centroids_pp(model, img);
    else
        // init == 0 -> Inicializacion aleatoria
        init_centroids_random(model, img);
    clock_t init_end = clock();

    // Medimos el tiempo de ejecucion total
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    kmeans_run(model, img, &cfg);

    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    double init_time = (double)(init_end - init_start) / CLOCKS_PER_SEC;
    printf("[Time]   Total elapsed: %.4f seconds (GPU + host overhead)\n", ms / 1000.0f);
    printf("[Time]   Initialization elapsed: %.4f seconds\n\n", init_time);

    print_centroids(model);
    printf("\n");
    print_cluster_sizes(model->labels, img->n_pixels, k);
    printf("\n");
    save_csv(model->labels, img->n_pixels, "kmeans_output.csv");
    save_image(model->labels, img->n_pixels, model->k, "kmeans_output.tif", filename);

    kmeans_free(model);
    free_image(img);
    return 0;
}