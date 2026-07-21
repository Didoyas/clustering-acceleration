/*
 * cpu_Kmeans.c
 * K-Means - CPU Version
 *
 * Compile (Linux):
 * Compile with: gcc -O2 -o cpuKM cpu_Kmeans.c common.c -lm $(gdal-config --cflags) $(gdal-config --libs)
 *      or with: gcc -O2 -o cpuKM cpu_Kmeans.c common.c -lm -lgdal
 *
 * Compile (MSVC on Windows / OSGeo4W):
 * Compile with: cl cpu_Kmeans.c common.c /O2 /I"C:\OSGeo4W\apps\gdal-dev\include" /link /LIBPATH:"C:\OSGeo4W\apps\gdal-dev\lib" gdal_i.lib /out:cpuKM.exe
 *
 * Run:      ./cpuKM <image_filepath> <number_of_clusters> [init: "random"(default)|"pp"] [max_iterations] [convergence] [seed]
 * Example for random init:  ./cpuKM sample.tif 5 random 100 0.0003 12345
 * Example for k-means++ init:  ./cpuKM sample.tif 5 pp 100 0.0003 12345
 */

#include "common.h"

void         assign_labels(KMeansModel *model, const Image *img);
float        update_centroids(KMeansModel *model, const Image *img, double *new_centroids, int *counts);
void         kmeans_run(KMeansModel *model, const Image *img, const KMeansConfig *cfg);

// Clasificacion de pixeles segun el algoritmo k-means
void assign_labels(KMeansModel *model, const Image *img)
{
    for (int p = 0; p < img->n_pixels; p++) {
        const float *pixel = &img->data[p * img->n_bands];
        float best_dist = FLT_MAX;
        int best_k = 0;

        if (is_nodata(pixel, img)) {
            model->labels[p] = -1;
            continue;
        }

        for (int c = 0; c < model->k; c++) {
            const float *centroid = &model->centroids[c * model->n_bands];
            float d = euclidean_distance_sq(pixel, centroid, img->n_bands);

            if (d < best_dist) {
                best_dist = d;
                best_k = c;
            }
        }
        model->labels[p] = best_k;
    }
}

// Calculamos nuevos centroides de cada cluster
float update_centroids(KMeansModel *model, const Image *img, double *new_centroids, int *counts)
{
    int k = model->k;
    int n_bands = model->n_bands;

    memset(new_centroids, 0, (size_t)k * n_bands * sizeof(double));
    memset(counts, 0, k * sizeof(int));

    for (int p = 0; p < img->n_pixels; p++) {
        int c = model->labels[p];
        if (c == -1) continue;
        counts[c]++;
        for (int b = 0; b < n_bands; b++)
            new_centroids[c * n_bands + b] += (double)img->data[p * n_bands + b];
    }

    double max_shift = 0.0;
    for (int c = 0; c < k; c++) {
        if (counts[c] == 0) {
            for (int b = 0; b < n_bands; b++)
                new_centroids[c * n_bands + b] = (double)model->centroids[c * n_bands + b];
            continue;
        }
        for (int b = 0; b < n_bands; b++) {
            new_centroids[c * n_bands + b] /= counts[c];
            double diff = fabs((float)new_centroids[c * n_bands + b]
                    - model->centroids[c * n_bands + b]);
            if (diff > max_shift) max_shift = diff;
        }
    }

    for (int c = 0; c < k; c++) {
        for (int b = 0; b < n_bands; b++) {
            model->centroids[c * n_bands + b] = (float)new_centroids[c * n_bands + b];
        }
    }

    return (float)max_shift;
}

// Loop del algoritmo k-means
void kmeans_run(KMeansModel *model, const Image *img, const KMeansConfig *cfg)
{
    model->labels = malloc(img->n_pixels * sizeof(int));
    double *new_centroids = calloc((size_t)model->k * model->n_bands, sizeof(double));
    int *counts = calloc(model->k, sizeof(int));
    if (!new_centroids || !counts) {
        fprintf(stderr, "[K-Means] Memory allocation failed for centroid update (new_centroids and counts).\n");
        free(new_centroids);
        free(counts);
        exit(EXIT_FAILURE);
    }

    clock_t iter_start = clock();
    for (int iter = 0; iter < cfg->max_iter; iter++) {
        assign_labels(model, img);
        float shift = update_centroids(model, img, new_centroids, counts);

        printf("[K-Means] iter %3d | max centroid shift = %.7f\n", iter + 1, shift);

        if (shift < cfg->convergence) {
            printf("[K-Means] Converged at iteration %d.\n", iter + 1);
            assign_labels(model, img);
            clock_t iter_end = clock();
            double iter_time = (double)(iter_end - iter_start) / CLOCKS_PER_SEC;
            printf("\n[Time]   K-means loop elapsed: %.4f seconds\n", iter_time);
            free(new_centroids);
            free(counts);
            return;
        }
    }
    assign_labels(model, img);

    printf("[K-Means] Reached maximum iterations (%d). Stopping now.\n", cfg->max_iter);

    clock_t iter_end = clock();
    double iter_time = (double)(iter_end - iter_start) / CLOCKS_PER_SEC;
    printf("\n[Time]   K-means loop elapsed: %.4f seconds\n", iter_time);

    free(new_centroids);
    free(counts);
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
        fprintf(stderr, "number_of_clusters must be a positive integer.\n");
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

    printf("=== K-Means - CPU Version ===\n");
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

    clock_t t_start = clock();
    kmeans_run(model, img, &cfg);
    clock_t t_end   = clock();

    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    double init_time = (double)(init_end - init_start) / CLOCKS_PER_SEC;
    printf("[Time]   Total elapsed: %.4f seconds\n", elapsed);
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